//! No-dependency creation layer (ones/zeros/empty/full).
//!
//! Each output block is `func(block_shape)`, where `block_shape` is read off
//! the chunk sizes. The shared `func` (a partial carrying dtype/meta/kwargs)
//! and `kwargs` are Python objects stored once; the per-block shape is a raw
//! Rust int tuple until a converter materializes it.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Expanded};

#[pyclass]
pub struct CreationLayer {
    name: String,
    func: PyObject,
    kwargs: PyObject,
    /// Chunk sizes per dimension; `chunks[d].len()` is the block count on `d`.
    chunks: Vec<Vec<i64>>,
}

#[pymethods]
impl CreationLayer {
    #[new]
    fn new(name: String, func: PyObject, kwargs: PyObject, chunks: Vec<Vec<i64>>) -> Self {
        Self { name, func, kwargs, chunks }
    }

    fn to_dask_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        to_dask_graph(py, &self.expand())
    }

    fn to_task_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_task_records(py, &self.expand())
    }
}

impl CreationLayer {
    fn expand(&self) -> Expanded<'_> {
        let ndim = self.chunks.len();
        let numblocks: Vec<usize> = self.chunks.iter().map(|c| c.len()).collect();
        let total: usize = if ndim == 0 { 1 } else { numblocks.iter().product() };
        let mut tasks = Vec::with_capacity(total);
        let mut coord = vec![0u32; ndim];

        for _ in 0..total {
            let shape: Vec<i64> = (0..ndim).map(|d| self.chunks[d][coord[d] as usize]).collect();
            tasks.push((coord.clone(), vec![ArgSlot::IntTuple(shape)]));

            for d in (0..ndim).rev() {
                coord[d] += 1;
                if (coord[d] as usize) < numblocks[d] {
                    break;
                }
                coord[d] = 0;
            }
        }

        Expanded {
            name: &self.name,
            func: &self.func,
            kwargs: &self.kwargs,
            literals: &[],
            dep_names: &[],
            tasks,
        }
    }
}
