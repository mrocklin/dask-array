//! Basic slicing layer (`SliceSlicesIntegers`).
//!
//! Per output block the task is `getitem(input_block, index_tuple)`, or an
//! `Alias` to the input block when the index is all full-slices and
//! getitem-optimization is allowed. The intricate per-dimension index math
//! (`_slice_1d`: irregular chunks, arbitrary and negative steps, integer-drop)
//! stays in tested Python and runs once per dimension (O(n_blocks)); this layer
//! does the O(n_tasks) cartesian-product expansion over the per-dimension
//! block slices.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{
    to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, IndexElem, NeutralTask,
};

/// One sliced dimension, as resolved by Python's `_slice_1d`.
struct Dim {
    /// The dimension's index is an integer, so it is dropped from the output.
    is_integer: bool,
    /// The output block order is reversed (a negative-step slice).
    reverse: bool,
    /// Input block index per position (the `_slice_1d` dict keys, sorted).
    blocks: Vec<u32>,
    /// `(start, stop, step)` per position; for an integer dim, `start` holds the
    /// (block-relative) integer index.
    elems: Vec<(Option<i64>, Option<i64>, Option<i64>)>,
}

#[pyclass]
pub struct SliceLayer {
    name: String,
    getitem: Py<PyAny>,
    kwargs: Py<PyAny>,
    dep_name: String,
    allow_opt: bool,
    dims: Vec<Dim>,
}

#[pymethods]
impl SliceLayer {
    /// `dims`: per output dimension `(is_integer, reverse, blocks, elems)`, with
    /// `blocks`/`elems` aligned to the sorted `_slice_1d` result.
    #[new]
    fn new(
        name: String,
        getitem: Py<PyAny>,
        kwargs: Py<PyAny>,
        dep_name: String,
        allow_opt: bool,
        dims: Vec<(
            bool,
            bool,
            Vec<u32>,
            Vec<(Option<i64>, Option<i64>, Option<i64>)>,
        )>,
    ) -> Self {
        let dims = dims
            .into_iter()
            .map(|(is_integer, reverse, blocks, elems)| Dim {
                is_integer,
                reverse,
                blocks,
                elems,
            })
            .collect();
        Self {
            name,
            getitem,
            kwargs,
            dep_name,
            allow_opt,
            dims,
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

impl SliceLayer {
    fn expand(&self) -> Expanded<'_> {
        let ndim = self.dims.len();
        let npos: Vec<usize> = self.dims.iter().map(|d| d.blocks.len()).collect();
        let total: usize = npos.iter().product(); // empty product = 1 (a 0-d slice)

        let mut tasks: Vec<NeutralTask> = Vec::with_capacity(total);
        let mut pos = vec![0usize; ndim];
        for _ in 0..total {
            let mut in_coord = vec![0u32; ndim];
            let mut out_coord: Vec<u32> = Vec::with_capacity(ndim);
            let mut elems: Vec<IndexElem> = Vec::with_capacity(ndim);
            let mut all_full = true;
            for (d, dim) in self.dims.iter().enumerate() {
                let p = pos[d];
                in_coord[d] = dim.blocks[p];
                let (start, stop, step) = dim.elems[p];
                if dim.is_integer {
                    // An integer index drops the dim and is never a full slice.
                    elems.push(IndexElem::Int(start.expect("integer index")));
                    all_full = false;
                } else {
                    out_coord.push(if dim.reverse {
                        (npos[d] - 1 - p) as u32
                    } else {
                        p as u32
                    });
                    if start.is_some() || stop.is_some() || step.is_some() {
                        all_full = false;
                    }
                    elems.push(IndexElem::Slice { start, stop, step });
                }
            }

            if self.allow_opt && all_full {
                tasks.push(NeutralTask {
                    name_idx: 0,
                    coord: out_coord,
                    compute: Compute::Alias,
                    slots: vec![ArgSlot::Dep {
                        name_idx: 0,
                        coord: in_coord,
                    }],
                });
            } else {
                tasks.push(NeutralTask {
                    name_idx: 0,
                    coord: out_coord,
                    compute: Compute::Call { func_idx: 0 },
                    slots: vec![
                        ArgSlot::Dep {
                            name_idx: 0,
                            coord: in_coord,
                        },
                        ArgSlot::Index(elems),
                    ],
                });
            }

            // Advance the position counter (last dim fastest); output keys are
            // explicit, so the iteration order itself doesn't affect the result.
            for d in (0..ndim).rev() {
                pos[d] += 1;
                if pos[d] < npos[d] {
                    break;
                }
                pos[d] = 0;
            }
        }

        Expanded {
            names: vec![self.name.as_str()],
            funcs: vec![&self.getitem],
            kwargs: &self.kwargs,
            literals: &[],
            dep_names: std::slice::from_ref(&self.dep_name),
            tasks,
        }
    }
}
