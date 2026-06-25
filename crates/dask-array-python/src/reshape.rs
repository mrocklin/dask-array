//! Reshape layer: per-block `M.reshape(in_block, out_shape)`.
//!
//! Both `ReshapeLowered._layer` and `ReshapeBlockwise._layer` build the same
//! subgraph: each input block maps 1:1, **by C-order position**, to one output
//! block, reshaped to that output block's shape. The Python `zip(out_keys,
//! in_keys, shapes)` pairs the p-th C-order coord of the (reshaped) input grid
//! with the p-th coord of the output grid and the p-th precomputed output shape.
//! The two grids have different shapes but the SAME total block count (the
//! reshape is lowered/rechunked so this holds), so we walk both coords in
//! lockstep, advancing each in C order (last axis fastest) over its own
//! numblocks.
//!
//! Per the recipe: Python plans (computes the output chunk grid + per-block
//! shapes, reusing dask's tested `reshape_rechunk`), Rust expands the
//! O(n_tasks) mapping. The shared `func` is `dask.utils.M.reshape` (calls
//! `block.reshape(shape)`); `kwargs` is empty.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask};

#[pyclass]
pub struct ReshapeLayer {
    /// Name of the output array.
    name: String,
    /// `dask.utils.M.reshape` — the single shared task function.
    func: PyObject,
    /// Empty dict, stored so `Expanded.kwargs` has a valid reference.
    kwargs: PyObject,
    /// Name of the single input array. Stored as a 1-element `Vec` so it can be
    /// borrowed as `dep_names` (the `ArgSlot::Dep` always uses `name_idx == 0`).
    dep_names: Vec<String>,
    /// Input block counts per dimension (the reshaped/rechunked input grid).
    in_numblocks: Vec<usize>,
    /// Output block counts per dimension.
    out_numblocks: Vec<usize>,
    /// Per-block output shapes, one per C-order position; `out_shapes.len()`
    /// equals the total block count.
    out_shapes: Vec<Vec<i64>>,
}

#[pymethods]
impl ReshapeLayer {
    #[new]
    fn new(
        name: String,
        func: PyObject,
        kwargs: PyObject,
        dep_name: String,
        in_numblocks: Vec<usize>,
        out_numblocks: Vec<usize>,
        out_shapes: Vec<Vec<i64>>,
    ) -> Self {
        Self {
            name,
            func,
            kwargs,
            dep_names: vec![dep_name],
            in_numblocks,
            out_numblocks,
            out_shapes,
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

impl ReshapeLayer {
    /// Pure-Rust expansion. Walks `total` C-order positions, advancing the input
    /// and output coords in lockstep (each in C order over its own numblocks).
    /// At position `p` the task is `M.reshape(in_block@in_coord,
    /// out_shapes[p])`, keyed at `out_coord`. This mirrors the legacy
    /// `zip(out_keys, in_keys, shapes)`: `out_keys`/`in_keys` are the C-order
    /// products of their grids, so the p-th element of each is the p-th coord.
    fn expand(&self) -> Expanded<'_> {
        // Empty product is 1 (0-d / single block), matching the Python product().
        let total: usize = self.in_numblocks.iter().product();
        let in_ndim = self.in_numblocks.len();
        let out_ndim = self.out_numblocks.len();
        let mut tasks = Vec::with_capacity(total);
        let mut in_coord = vec![0u32; in_ndim];
        let mut out_coord = vec![0u32; out_ndim];

        for p in 0..total {
            tasks.push(NeutralTask {
                name_idx: 0,
                coord: out_coord.clone(),
                compute: Compute::Call { func_idx: 0 },
                slots: vec![
                    ArgSlot::Dep {
                        name_idx: 0,
                        coord: in_coord.clone(),
                    },
                    ArgSlot::IntTuple(self.out_shapes[p].clone()),
                ],
            });

            // Advance both coords in C order (last axis fastest), each over its
            // own grid. They step in lockstep — both are the p-th coord.
            for d in (0..in_ndim).rev() {
                in_coord[d] += 1;
                if (in_coord[d] as usize) < self.in_numblocks[d] {
                    break;
                }
                in_coord[d] = 0;
            }
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
