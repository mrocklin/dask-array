//! Stack layer: stack multiple arrays along a new axis.
//!
//! Each output block at coord `c` picks input array `dep_names[c[axis]]`
//! and source block coord `c` with the new axis position removed. The task
//! is `np.expand_dims(source_block, axis)`. The axis is an integer scalar, so
//! stack layers can use binary records instead of falling back for a literal
//! getitem indexer.
//!
//! `dep_names` holds one entry per input array; `name_idx` in each `Dep` slot
//! is `c[axis]` — so the new-axis coordinate directly selects the input array.
//! The output block grid is `self.out_numblocks` (the new axis has `n_arrays`
//! blocks of size 1).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask, Num};

#[pyclass]
pub struct StackLayer {
    /// Output layer name.
    name: String,
    /// `np.expand_dims` function (shared for every task).
    func: Py<PyAny>,
    /// Empty kwargs dict (stack tasks take no kwargs).
    kwargs: Py<PyAny>,
    /// Names of the input arrays, one per stacked array; `dep_names[i]` is the
    /// array selected when the new-axis coordinate equals `i`.
    dep_names: Vec<String>,
    /// Number of blocks per OUTPUT dimension (C order). Length = input ndim + 1.
    out_numblocks: Vec<usize>,
    /// The position of the new axis in the output coordinate (0-based).
    axis: usize,
}

#[pymethods]
impl StackLayer {
    #[new]
    fn new(
        name: String,
        func: Py<PyAny>,
        kwargs: Py<PyAny>,
        dep_names: Vec<String>,
        out_numblocks: Vec<usize>,
        axis: usize,
    ) -> Self {
        Self {
            name,
            func,
            kwargs,
            dep_names,
            out_numblocks,
            axis,
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

impl StackLayer {
    /// Expand into the neutral form.
    ///
    /// Iterates output blocks in C order. For each output coord `c`:
    /// - `c[axis]` selects the input array (`dep_names[c[axis]]`).
    /// - The source block coord is `c` with position `axis` removed.
    /// - Slots: `[Dep{name_idx: c[axis], coord: source_coord}, Scalar(axis)]`.
    fn expand(&self) -> Expanded<'_> {
        let out_ndim = self.out_numblocks.len();
        let total: usize = if out_ndim == 0 {
            1
        } else {
            self.out_numblocks.iter().product()
        };
        let mut tasks = Vec::with_capacity(total);
        let mut out_coord = vec![0u32; out_ndim];

        for _ in 0..total {
            // The new-axis index selects the input array.
            let name_idx = out_coord[self.axis] as usize;

            // Source block coord: output coord with the new axis position removed.
            let mut src_coord = Vec::with_capacity(out_ndim - 1);
            for (d, &c) in out_coord.iter().enumerate() {
                if d != self.axis {
                    src_coord.push(c);
                }
            }

            tasks.push(NeutralTask {
                name_idx: 0,
                coord: out_coord.clone(),
                compute: Compute::Call { func_idx: 0 },
                slots: vec![
                    // dep_names[name_idx] is the input array for this block.
                    ArgSlot::Dep {
                        name_idx,
                        coord: src_coord,
                    },
                    ArgSlot::Scalar(Num::Int(self.axis as i64)),
                ],
            });

            // Advance out_coord in C order.
            for d in (0..out_ndim).rev() {
                out_coord[d] += 1;
                if (out_coord[d] as usize) < self.out_numblocks[d] {
                    break;
                }
                out_coord[d] = 0;
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
