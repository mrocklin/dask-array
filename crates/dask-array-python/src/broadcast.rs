//! BroadcastTo layer.
//!
//! Each output block `(i0, i1, ..., i_{ndim-1})` is
//! `np.broadcast_to(input_block, chunk_shape)` where:
//!
//! - `ndim_new` leading output axes have no corresponding input axis (they were
//!   added by the broadcast); they carry no dependency coordinate.
//! - For each input axis `d` (which maps to output axis `ndim_new + d`):
//!   - if the input has a single block on that axis (`broadcast_dim[d]` = true),
//!     the input coord is always 0;
//!   - otherwise the input coord equals the output coord at position `ndim_new + d`.
//! - `chunk_shape` is the per-output-block shape (a tuple of ints) →
//!   `ArgSlot::IntTuple`.
//!
//! Slots per task: `[Dep{input, input_coord}, IntTuple(chunk_shape)]`.
//! Single func (`np.broadcast_to`), no kwargs, no literals.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask};

#[pyclass]
pub struct BroadcastLayer {
    name: String,
    func: PyObject,
    kwargs: PyObject,
    /// Name of the input array.
    dep_name: String,
    /// Per-output-block chunk sizes: `out_chunks[d][i]` = size of the i-th
    /// block on output dimension `d`. `out_chunks.len()` is total output ndim.
    out_chunks: Vec<Vec<i64>>,
    /// Number of new leading axes (not present in the input).
    ndim_new: usize,
    /// Per input dimension: true if the input has a single block (broadcast
    /// from size-1), false if the output coord is passed through.
    /// `broadcast_dim.len()` == input ndim == `out_chunks.len() - ndim_new`.
    broadcast_dim: Vec<bool>,
}

#[pymethods]
impl BroadcastLayer {
    #[new]
    fn new(
        name: String,
        func: PyObject,
        kwargs: PyObject,
        dep_name: String,
        out_chunks: Vec<Vec<i64>>,
        ndim_new: usize,
        broadcast_dim: Vec<bool>,
    ) -> Self {
        Self {
            name,
            func,
            kwargs,
            dep_name,
            out_chunks,
            ndim_new,
            broadcast_dim,
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

impl BroadcastLayer {
    fn expand(&self) -> Expanded<'_> {
        let out_ndim = self.out_chunks.len();
        let in_ndim = self.broadcast_dim.len();
        // number of blocks per output dimension
        let numblocks: Vec<usize> = self.out_chunks.iter().map(|c| c.len()).collect();
        let total: usize = if out_ndim == 0 {
            1
        } else {
            numblocks.iter().product()
        };

        let mut tasks = Vec::with_capacity(total);
        let mut coord = vec![0u32; out_ndim];

        for _ in 0..total {
            // Build input coord: only input dims contribute.
            // Output dims [0..ndim_new] are new; output dims [ndim_new..] map
            // to input dims [0..in_ndim].
            let in_coord: Vec<u32> = (0..in_ndim)
                .map(|d| {
                    if self.broadcast_dim[d] {
                        0
                    } else {
                        coord[self.ndim_new + d]
                    }
                })
                .collect();

            // Chunk shape for this output block.
            let chunk_shape: Vec<i64> = (0..out_ndim)
                .map(|d| self.out_chunks[d][coord[d] as usize])
                .collect();

            tasks.push(NeutralTask {
                name_idx: 0,
                coord: coord.clone(),
                compute: Compute::Call { func_idx: 0 },
                slots: vec![
                    ArgSlot::Dep {
                        name_idx: 0,
                        coord: in_coord,
                    },
                    ArgSlot::IntTuple(chunk_shape),
                ],
            });

            // Increment coord in C order (last axis fastest).
            for d in (0..out_ndim).rev() {
                coord[d] += 1;
                if (coord[d] as usize) < numblocks[d] {
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
            dep_names: std::slice::from_ref(&self.dep_name),
            tasks,
        }
    }
}
