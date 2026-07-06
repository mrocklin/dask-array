//! Binary-record layer for Python-validated fused blockwise tasks.
//!
//! Python builds and validates the block-independent `_FusedSubgraph` callable.
//! Rust only expands the output grid into ordinary call tasks whose arguments
//! are broadcast-mapped source block dependencies.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask};

#[pyclass]
pub struct FusedBlockwiseLayer {
    name: String,
    func: Py<PyAny>,
    empty_kwargs: Py<PyAny>,
    dep_names: Vec<String>,
    dep_numblocks: Vec<Vec<usize>>,
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
        sources: Vec<(String, Vec<usize>)>,
    ) -> PyResult<Self> {
        for (_, source_numblocks) in &sources {
            if source_numblocks.len() > numblocks.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "source has more dimensions than output",
                ));
            }
        }
        let (dep_names, dep_numblocks): (Vec<_>, Vec<_>) = sources.into_iter().unzip();
        Ok(Self {
            name,
            func,
            empty_kwargs: PyDict::new(py).unbind().into_any(),
            dep_names,
            dep_numblocks,
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
    fn broadcast_block_id(source_numblocks: &[usize], out_coord: &[u32]) -> Vec<u32> {
        let offset = out_coord.len() - source_numblocks.len();
        source_numblocks
            .iter()
            .enumerate()
            .map(|(i, &n)| if n == 1 { 0 } else { out_coord[offset + i] })
            .collect()
    }

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
            let slots = self
                .dep_numblocks
                .iter()
                .enumerate()
                .map(|(dep_idx, source_numblocks)| ArgSlot::Dep {
                    name_idx: dep_idx,
                    coord: Self::broadcast_block_id(source_numblocks, &coord),
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
