//! Binary-record layer for Python-validated fused blockwise tasks.
//!
//! Python builds and validates the block-independent `_FusedSubgraph` callable.
//! Rust only expands the output grid into ordinary call tasks whose arguments
//! are the per-output source block dependencies Python already resolved from
//! each fused `_execute_subgraph` task.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask};

#[pyclass]
pub struct FusedBlockwiseLayer {
    name: String,
    func: Py<PyAny>,
    empty_kwargs: Py<PyAny>,
    dep_names: Vec<String>,
    dep_slots: Vec<Vec<(usize, Vec<u32>)>>,
    numblocks: Vec<usize>,
}

#[pymethods]
impl FusedBlockwiseLayer {
    #[new]
    fn new(
        py: Python<'_>,
        name: String,
        func: Py<PyAny>,
        numblocks: Vec<usize>,
        dep_names: Vec<String>,
        dep_slots: Vec<Vec<(usize, Vec<u32>)>>,
    ) -> PyResult<Self> {
        let expected: usize = if numblocks.is_empty() {
            1
        } else {
            numblocks.iter().product()
        };
        if dep_slots.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dep slots length does not match output blocks",
            ));
        }
        for slots in &dep_slots {
            for (dep_idx, _) in slots {
                if *dep_idx >= dep_names.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "dep slot index out of range",
                    ));
                }
            }
        }

        Ok(Self {
            name,
            func,
            empty_kwargs: PyDict::new(py).unbind().into_any(),
            dep_names,
            dep_slots,
            numblocks,
        })
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

impl FusedBlockwiseLayer {
    fn expand(&self) -> Expanded<'_> {
        let ndim = self.numblocks.len();
        let total: usize = if ndim == 0 {
            1
        } else {
            self.numblocks.iter().product()
        };
        let mut tasks = Vec::with_capacity(total);
        let mut coord = vec![0u32; ndim];

        for task_idx in 0..total {
            let slots = self.dep_slots[task_idx]
                .iter()
                .map(|(dep_idx, dep_coord)| ArgSlot::Dep {
                    name_idx: *dep_idx,
                    coord: dep_coord.clone(),
                })
                .collect();
            tasks.push(NeutralTask {
                name_idx: 0,
                coord: coord.clone(),
                compute: Compute::Call { func_idx: 0 },
                slots,
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
            kwargs: &self.empty_kwargs,
            literals: &[],
            dep_names: &self.dep_names,
            tasks,
        }
    }
}
