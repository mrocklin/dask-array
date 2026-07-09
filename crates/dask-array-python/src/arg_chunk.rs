//! ArgChunk layer: the per-block chunk step of an arg reduction (`argmin`/
//! `argmax`), non-ravel case.
//!
//! The reference `_layer` emits, for every input block index `k` (C order over
//! `x.numblocks`), a task `(chunk_func, (x.name, *k), axis, off)` — i.e.
//! `chunk_func(x_block, axis, off)`. `axis` is shared across all blocks; `off`
//! is the per-block start offset along the reduction axis
//! (`pluck(axis[0], offsets)` in the legacy non-ravel branch). So it fits the
//! neutral form exactly: one shared func, output coord == input coord (identity,
//! like coarsen), with three args per task — a single `ArgSlot::Dep` (the input
//! block), one `ArgSlot::IntTuple` (the shared `axis`, always a tuple of ints —
//! `arg_chunk` only does `len(axis)`/`axis[0]`, so a plain int-tuple is
//! behaviour-identical and, unlike a `Literal`, expressible in binary records),
//! and one `ArgSlot::Scalar(Num::Int(off))` (the per-block offset).
//!
//! Two offset shapes: the non-ravel (`axis=k`) case carries a per-block scalar
//! (`ArgSlot::Scalar`); the ravel (`axis=None`) case carries the nested
//! `(per-dim offsets, full shape)` the ravel `arg_chunk` unpacks, built as an
//! `ArgSlot::List([IntTuple(offsets), IntTuple(shape)])` (a list tuple-unpacks
//! like the original tuple, so no nested-tuple arg type is needed).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask, Num};

#[pyclass]
pub struct ArgChunkLayer {
    /// Output layer name.
    name: String,
    /// The shared `chunk_func` (the partialed arg-chunk function).
    func: Py<PyAny>,
    /// Shared kwargs applied to every call — empty.
    kwargs: Py<PyAny>,
    /// The shared reduction `axis`, a tuple of ints, emitted per task as an
    /// `ArgSlot::IntTuple`.
    axis: Vec<i64>,
    /// Name of the single input dependency.
    dep_names: Vec<String>,
    /// Number of blocks per dimension (input grid; arg-chunk preserves the grid,
    /// so it is also the output grid).
    numblocks: Vec<usize>,
    /// Ravel (full / `axis=None`) reduction? Then the offset is the nested
    /// `(per-dim offsets, full shape)` tuple rather than a scalar.
    ravel: bool,
    /// Non-ravel: per-block start offset along the reduction axis (C order).
    offs: Vec<i64>,
    /// Ravel: per-block per-dimension start offsets (C order, one inner Vec of
    /// length ndim per block).
    offset_tuples: Vec<Vec<i64>>,
    /// Ravel: the full array shape (shared across blocks).
    shape: Vec<i64>,
}

#[pymethods]
impl ArgChunkLayer {
    /// `axis`: shared per-block arg (the reduction axes tuple). `dep_name`: the
    /// input array name. `numblocks`: blocks per dimension. `offs`: the per-block
    /// offsets, already computed in Python (one per block, C order), mirroring the
    /// legacy non-ravel `pluck(axis[0], offsets)`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        func: Py<PyAny>,
        kwargs: Py<PyAny>,
        axis: Vec<i64>,
        dep_name: String,
        numblocks: Vec<usize>,
        ravel: bool,
        offs: Vec<i64>,
        offset_tuples: Vec<Vec<i64>>,
        shape: Vec<i64>,
    ) -> Self {
        Self {
            name,
            func,
            kwargs,
            axis,
            dep_names: vec![dep_name],
            numblocks,
            ravel,
            offs,
            offset_tuples,
            shape,
        }
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

impl ArgChunkLayer {
    /// Build the neutral form. One task per block, output coord == input coord, a
    /// single dependency at the same coord plus the shared `axis` int-tuple and the
    /// per-block offset scalar. Iterates blocks in C order to match
    /// `itertools.product` (last axis fastest), so `offs[p]` lines up with the
    /// `p`-th block.
    fn expand(&self) -> Expanded<'_> {
        let ndim = self.numblocks.len();
        let total: usize = if ndim == 0 {
            1
        } else {
            self.numblocks.iter().product()
        };
        let mut tasks = Vec::with_capacity(total);
        let mut coord = vec![0u32; ndim];

        for p in 0..total {
            // The per-block offset: a scalar along the reduction axis (non-ravel),
            // or the nested `(per-dim offsets, full shape)` the ravel arg_chunk
            // unpacks (a List is fine — it tuple-unpacks like a tuple).
            let off_slot = if self.ravel {
                ArgSlot::List(vec![
                    ArgSlot::IntTuple(self.offset_tuples[p].clone()),
                    ArgSlot::IntTuple(self.shape.clone()),
                ])
            } else {
                ArgSlot::Scalar(Num::Int(self.offs[p]))
            };
            tasks.push(NeutralTask {
                nbytes: 0,
                name_idx: 0,
                coord: coord.clone(),
                compute: Compute::Call { func_idx: 0 },
                // chunk_func(x_block, axis, off)
                slots: vec![
                    ArgSlot::Dep {
                        name_idx: 0,
                        coord: coord.clone(),
                    },
                    ArgSlot::IntTuple(self.axis.clone()), // axis
                    off_slot,
                ],
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
            kwargs: &self.kwargs,
            literals: &[],
            dep_names: &self.dep_names,
            tasks,
        }
    }
}
