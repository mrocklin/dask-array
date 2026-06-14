//! Squeeze layer: remove size-1 axes from each input block.
//!
//! Each output block at coord `(i, j, ...)` corresponds to the input block at a
//! coord derived by inserting `0` at each squeezed axis. The shared func is
//! `np.squeeze` with `axis=chunk_axis` (the tuple of squeezed axes, same for
//! every block). Shared kwargs carry `{"axis": chunk_axis}`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask};

#[pyclass]
pub struct SqueezeLayer {
    /// Output layer name.
    name: String,
    /// `np.squeeze` (or equivalent partial).
    func: PyObject,
    /// `{"axis": chunk_axis}` — shared across all tasks.
    kwargs: PyObject,
    /// Name of the single input dependency.
    dep_name: String,
    /// Number of output blocks per dimension (len = output ndim).
    numblocks: Vec<usize>,
    /// Input ndim = output ndim + number of squeezed axes.
    input_ndim: usize,
    /// Sorted list of squeezed axes (indices into the INPUT dimensions).
    /// Each of these is an axis that is always 0 in the input coord.
    axis_set: Vec<usize>,
}

#[pymethods]
impl SqueezeLayer {
    /// `axis_set`: sorted list of squeezed (size-1) INPUT dimension indices.
    /// `numblocks`: number of blocks per OUTPUT dimension.
    #[new]
    fn new(
        name: String,
        func: PyObject,
        kwargs: PyObject,
        dep_name: String,
        numblocks: Vec<usize>,
        input_ndim: usize,
        axis_set: Vec<usize>,
    ) -> Self {
        Self { name, func, kwargs, dep_name, numblocks, input_ndim, axis_set }
    }

    fn to_dask_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        to_dask_graph(py, &self.expand())
    }

    fn to_task_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_task_records(py, &self.expand())
    }
}

impl SqueezeLayer {
    /// Build the neutral form. For each output block coord `(i, j, ...)`,
    /// reconstruct the input coord by inserting `0` at each squeezed axis.
    fn expand(&self) -> Expanded<'_> {
        let out_ndim = self.numblocks.len();
        let total: usize = if out_ndim == 0 { 1 } else { self.numblocks.iter().product() };
        let mut tasks = Vec::with_capacity(total);
        let mut out_coord = vec![0u32; out_ndim];

        for _ in 0..total {
            // Build input coord: walk input dims; insert 0 at squeezed axes,
            // copy from out_coord at kept axes.
            let mut in_coord = Vec::with_capacity(self.input_ndim);
            let mut out_pos = 0usize;
            let mut axis_iter = self.axis_set.iter().peekable();
            for in_dim in 0..self.input_ndim {
                if axis_iter.peek() == Some(&&in_dim) {
                    in_coord.push(0u32);
                    axis_iter.next();
                } else {
                    in_coord.push(out_coord[out_pos]);
                    out_pos += 1;
                }
            }

            tasks.push(NeutralTask {
                name_idx: 0,
                coord: out_coord.clone(),
                compute: Compute::Call { func_idx: 0 },
                slots: vec![ArgSlot::Dep { name_idx: 0, coord: in_coord }],
            });

            // Advance out_coord in C order.
            for d in (0..out_ndim).rev() {
                out_coord[d] += 1;
                if (out_coord[d] as usize) < self.numblocks[d] {
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
            dep_names: std::slice::from_ref(&self.dep_name),
            tasks,
        }
    }
}
