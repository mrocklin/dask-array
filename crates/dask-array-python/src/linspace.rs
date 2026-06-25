//! Linspace layer (`da.linspace`): 1-D indexed creation.
//!
//! Each block computes `linspace(blockstart, blockstop, size)` for its slice of
//! the interval; `endpoint` and `dtype` are baked into the shared chunk function
//! (a `functools.partial`), so blocks differ only in the three per-block scalars
//! `blockstart`/`blockstop`/`size`, and there are no dependencies. The per-block
//! arithmetic — a running `blockstart` that advances by `step*bs` each block, and
//! a `blockstop` that uses `bs - 1` elements when `endpoint` is set — stays in
//! Python's `_frisky_layer`, exactly mirroring the legacy `_layer` (which also
//! preserves the int/float type of `start`/`step`). Rust slots the resulting
//! scalars into the per-block args via [`ArgSlot::Scalar`].

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask, Num};

#[pyclass]
pub struct LinspaceLayer {
    name: String,
    /// `functools.partial(chunk.linspace, endpoint=..., dtype=...)` — the shared
    /// chunk function.
    func: PyObject,
    /// Shared kwargs (empty — `endpoint`/`dtype` are baked into the partial).
    kwargs: PyObject,
    blockstarts: Vec<Num>,
    blockstops: Vec<Num>,
    sizes: Vec<i64>,
}

#[pymethods]
impl LinspaceLayer {
    /// `blockstarts`/`blockstops`/`sizes`: the per-block scalars, already computed
    /// in Python (aligned, one per block; the legacy `_layer` arithmetic).
    #[new]
    fn new(
        name: String,
        func: PyObject,
        kwargs: PyObject,
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

impl LinspaceLayer {
    fn expand(&self) -> Expanded<'_> {
        let tasks = (0..self.sizes.len())
            .map(|i| NeutralTask {
                name_idx: 0,
                coord: vec![i as u32],
                compute: Compute::Call { func_idx: 0 },
                // linspace(blockstart, blockstop, size)
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
