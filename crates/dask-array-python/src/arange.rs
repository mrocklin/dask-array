//! Arange layer (`da.arange`): 1-D indexed creation.
//!
//! Each block computes `arange(blockstart, blockstop, step, size, dtype)` for
//! its slice of the range; blocks differ only in the per-block scalars
//! `blockstart`/`blockstop`/`size`, and there are no dependencies. The per-block
//! start/stop arithmetic (`start + elem_count*step`, which must preserve the
//! int/float type of `start`/`step`) stays in Python's `_frisky_layer`, exactly
//! mirroring the legacy `_layer`; Rust slots the resulting scalars into the
//! per-block args via [`ArgSlot::Scalar`]. `step`, `dtype`, and `like` are baked
//! into the shared Python function so the binary records path has no literal
//! Python args.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask, Num};

#[pyclass]
pub struct ArangeLayer {
    name: String,
    /// `functools.partial(chunk.arange, like=like)` — the shared chunk function.
    func: Py<PyAny>,
    /// Shared kwargs (empty — everything is positional / baked into the partial).
    kwargs: Py<PyAny>,
    blockstarts: Vec<Num>,
    blockstops: Vec<Num>,
    sizes: Vec<i64>,
}

#[pymethods]
impl ArangeLayer {
    /// `step`, `dtype`: shared per-block args. `blockstarts`/`blockstops`/`sizes`:
    /// the per-block scalars, already computed in Python (aligned, one per block).
    #[new]
    fn new(
        name: String,
        func: Py<PyAny>,
        kwargs: Py<PyAny>,
        blockstarts: Vec<Num>,
        blockstops: Vec<Num>,
        sizes: Vec<i64>,
    ) -> Self {
        Self {
            name,
            func,
            kwargs,
            blockstarts,
            blockstops,
            sizes,
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

impl ArangeLayer {
    fn expand(&self) -> Expanded<'_> {
        let tasks = (0..self.sizes.len())
            .map(|i| NeutralTask {
                nbytes: 0,
                name_idx: 0,
                coord: vec![i as u32],
                compute: Compute::Call { func_idx: 0 },
                // arange_bound(blockstart, blockstop, size)
                slots: vec![
                    ArgSlot::Scalar(self.blockstarts[i]),
                    ArgSlot::Scalar(self.blockstops[i]),
                    ArgSlot::Scalar(Num::Int(self.sizes[i])),
                ],
            })
            .collect();

        Expanded {
            names: vec![&self.name],
            funcs: vec![&self.func],
            kwargs: &self.kwargs,
            literals: &[],
            dep_names: &[],
            tasks,
        }
    }
}
