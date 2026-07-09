//! Coarsen layer: apply a fixed reduction over fixed-size neighborhoods, per
//! block, on an already-aligned input grid.
//!
//! This is the simplest blockwise shape. The reference `_layer` emits, for every
//! input block index `in_idx`, a task `Task((name, *in_idx), func, TaskRef((x,
//! *in_idx)))` where `func = functools.partial(chunk.coarsen, reduction,
//! axes=axes, trim_excess=..., **kwargs)`. Everything is baked into the single
//! shared `partial`, so there is one func, one dependency per task, and the
//! output coord equals the input coord (identity map). It fits the neutral form
//! exactly: each task is `Compute::Call { func_idx: 0 }` with a single
//! `ArgSlot::Dep { name_idx: 0, coord }`, and shared `kwargs` is the (typically
//! empty) dict.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask};

#[pyclass]
pub struct CoarsenLayer {
    /// Output layer name.
    name: String,
    /// The shared `functools.partial(chunk.coarsen, reduction, axes=..., ...)`.
    func: Py<PyAny>,
    /// Shared kwargs applied to every call — usually empty (everything is baked
    /// into the partial).
    kwargs: Py<PyAny>,
    /// Name of the single input dependency.
    dep_name: String,
    /// Number of blocks per dimension. Coarsen preserves the block grid (one
    /// output block per input block), so this is both the input and output grid.
    numblocks: Vec<usize>,
}

#[pymethods]
impl CoarsenLayer {
    /// `func`: the shared `functools.partial`. `kwargs`: shared dict (usually
    /// empty). `numblocks`: number of blocks per dimension (identical input and
    /// output grid).
    #[new]
    fn new(
        name: String,
        func: Py<PyAny>,
        kwargs: Py<PyAny>,
        dep_name: String,
        numblocks: Vec<usize>,
    ) -> Self {
        Self {
            name,
            func,
            kwargs,
            dep_name,
            numblocks,
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

impl CoarsenLayer {
    /// Build the neutral form. One task per block, output coord == input coord,
    /// a single dependency at the same coord. Iterates blocks in C order to match
    /// `itertools.product`.
    fn expand(&self) -> Expanded<'_> {
        let ndim = self.numblocks.len();
        let total: usize = if ndim == 0 {
            1
        } else {
            self.numblocks.iter().product()
        };
        let mut tasks = Vec::with_capacity(total);
        let mut coord = vec![0u32; ndim];

        for _ in 0..total {
            tasks.push(NeutralTask {
                nbytes: 0,
                name_idx: 0,
                coord: coord.clone(),
                compute: Compute::Call { func_idx: 0 },
                slots: vec![ArgSlot::Dep {
                    name_idx: 0,
                    coord: coord.clone(),
                }],
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
            dep_names: std::slice::from_ref(&self.dep_name),
            tasks,
        }
    }
}
