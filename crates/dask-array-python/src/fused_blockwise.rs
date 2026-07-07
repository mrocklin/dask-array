//! Binary-record layer for Python-validated fused blockwise tasks.
//!
//! Python builds and validates the block-independent `_FusedSubgraph` callable.
//! Rust only expands the output grid into ordinary call tasks whose arguments
//! are the per-output source block dependencies Python already resolved from
//! each fused `_execute_subgraph` task.
//!
//! Most fused layers pass only source-block dependency refs. A layer that bakes
//! its own `block_id` into the subgraph (map_overlap's `_trim`, stencils,
//! `block_info`) is made block-independent on the Python side by lifting that
//! literal into a per-block *seed*: an ordinary value (a nested int tuple such as
//! `(block_id, numblocks)`) that `_execute_subgraph` feeds into a holed subgraph
//! slot. Those seeds arrive as `seed_slots`, appended to each task's args AFTER
//! the dependency refs — matching the `inkeys` order Python assigns (sources
//! first, then seeds). A layer with no lifted literals passes empty seed lists,
//! so its records are byte-identical to before.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask, Num};

#[pyclass]
pub struct FusedBlockwiseLayer {
    name: String,
    func: Py<PyAny>,
    empty_kwargs: Py<PyAny>,
    dep_names: Vec<String>,
    dep_slots: Vec<Vec<(usize, Vec<u32>)>>,
    /// Per-block lifted-literal seed args, appended after the dep refs. Each is a
    /// nested int structure pre-converted to an owned `ArgSlot` (no Python refs),
    /// so expansion stays pure Rust. Empty for the common (no block_id) case.
    seed_slots: Vec<Vec<ArgSlot>>,
    numblocks: Vec<usize>,
}

/// Convert a per-block seed value (a nested structure of ints, e.g.
/// `(block_id, numblocks)`) into an owned `ArgSlot`. A plain int becomes a
/// `Scalar`, a flat int sequence an `IntTuple` (rendering as a real tuple), and a
/// nested sequence a `List` of the same. Rejects anything that isn't int-shaped
/// so a mis-inferred seed declines to the tuples path rather than emitting a
/// wrong graph.
///
/// Fidelity note: only the innermost flat int runs stay tuples; a *nested*
/// sequence's outer container renders as a Python list, not a tuple (the grammar
/// has no tuple-of-slots element). Harmless for the consumers that reach the seed
/// path — `_trim` and the stencil shapes only index/iterate the seed — but a
/// hypothetical consumer doing `isinstance(seed, tuple)` on an outer container
/// would see the difference. The Python side (`_seed_spec`) only lifts such
/// literals, so it owns that contract.
fn seed_to_slot(obj: &Bound<'_, PyAny>) -> PyResult<ArgSlot> {
    // bool is an int subclass in Python; seeds are coordinates, never bools.
    if obj.is_instance_of::<pyo3::types::PyBool>() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "seed value must be an int or nested int sequence, got bool",
        ));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(ArgSlot::Scalar(Num::Int(i)));
    }
    let items: Vec<Bound<'_, PyAny>> = if let Ok(t) = obj.cast::<PyTuple>() {
        t.iter().collect()
    } else if let Ok(l) = obj.cast::<PyList>() {
        l.iter().collect()
    } else {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "seed value must be an int or nested int sequence",
        ));
    };
    // A flat int tuple renders as a real tuple; a nested one as a List of the
    // same (a List indexes/unpacks like a tuple, which is all the consumers do).
    let all_int = items
        .iter()
        .all(|it| !it.is_instance_of::<pyo3::types::PyBool>() && it.extract::<i64>().is_ok());
    if all_int {
        let ints: Vec<i64> = items.iter().map(|it| it.extract::<i64>().unwrap()).collect();
        return Ok(ArgSlot::IntTuple(ints));
    }
    let mut nested: Vec<ArgSlot> = Vec::with_capacity(items.len());
    for it in &items {
        nested.push(seed_to_slot(it)?);
    }
    Ok(ArgSlot::List(nested))
}

#[pymethods]
impl FusedBlockwiseLayer {
    #[new]
    fn new(
        py: Python<'_>,
        name: String,
        func: Py<PyAny>,
        numblocks: Vec<usize>,
        dep_names: Vec<String>,
        dep_slots: Vec<Vec<(usize, Vec<u32>)>>,
        seed_slots: Vec<Vec<Py<PyAny>>>,
    ) -> PyResult<Self> {
        let expected: usize = if numblocks.is_empty() {
            1
        } else {
            numblocks.iter().product()
        };
        // `seed_slots` is either empty (the common no-block_id case: no seeds for
        // any block) or one list per output block.
        if dep_slots.len() != expected || (!seed_slots.is_empty() && seed_slots.len() != expected) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dep/seed slots length does not match output blocks",
            ));
        }
        for slots in &dep_slots {
            for (dep_idx, _) in slots {
                if *dep_idx >= dep_names.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "dep slot index out of range",
                    ));
                }
            }
        }
        // Pre-convert each block's seed values into owned int-shaped ArgSlots up
        // front, so expansion is pure Rust and a non-int seed is rejected here.
        let seed_slots: Vec<Vec<ArgSlot>> = seed_slots
            .into_iter()
            .map(|block| {
                block
                    .into_iter()
                    .map(|obj| seed_to_slot(obj.bind(py)))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            name,
            func,
            empty_kwargs: PyDict::new(py).unbind().into_any(),
            dep_names,
            dep_slots,
            seed_slots,
            numblocks,
        })
    }

    fn to_dask_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        to_dask_graph(py, &self.expand())
    }

    fn to_task_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_task_records(py, &self.expand())
    }

    fn to_records_chunk<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        crate::common::to_records_chunk(py, &self.expand())
    }
}

impl FusedBlockwiseLayer {
    fn expand(&self) -> Expanded<'_> {
        let ndim = self.numblocks.len();
        let total: usize = if ndim == 0 {
            1
        } else {
            self.numblocks.iter().product()
        };
        let mut tasks = Vec::with_capacity(total);
        let mut coord = vec![0u32; ndim];

        for task_idx in 0..total {
            // Source dependency refs first, then any lifted-literal seeds — the
            // args order Python assigned when building `inkeys`.
            let mut slots: Vec<ArgSlot> = self.dep_slots[task_idx]
                .iter()
                .map(|(dep_idx, dep_coord)| ArgSlot::Dep {
                    name_idx: *dep_idx,
                    coord: dep_coord.clone(),
                })
                .collect();
            if let Some(seeds) = self.seed_slots.get(task_idx) {
                slots.extend(seeds.iter().cloned());
            }
            tasks.push(NeutralTask {
                name_idx: 0,
                coord: coord.clone(),
                compute: Compute::Call { func_idx: 0 },
                slots,
            });

            for d in (0..ndim).rev() {
                coord[d] += 1;
                if (coord[d] as usize) < self.numblocks[d] {
                    break;
                }
                coord[d] = 0;
            }
        }

        Expanded {
            names: vec![&self.name],
            funcs: vec![&self.func],
            kwargs: &self.empty_kwargs,
            literals: &[],
            dep_names: &self.dep_names,
            tasks,
        }
    }
}
