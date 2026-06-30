//! Diag layers (`da.diag`, k=0): build a diagonal matrix from a 1-D array
//! (`Diag1D`) or extract the diagonal of a square-block 2-D array
//! (`Diag2DSimple`).
//!
//! `Diag1D` is the **per-task-kwargs** exemplar. Its `(i, j)` block grid is
//! square (one block per 1-D input block on each axis); the diagonal blocks
//! `i == j` are `np.diag(x_block_i)` (a plain `Call` with one dep), and the
//! off-diagonal blocks are `np.zeros_like(meta, shape=(m, n))` — `meta` is a
//! shared literal but `shape` is a *per-block keyword*, so those tasks use
//! `Compute::CallKw` to attach `shape` as an `ArgSlot::IntTuple`.
//!
//! `Diag2DSimple` is a plain single-func layer: output 1-D block `i` is
//! `np.diag(x[(i, i)])`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask};

#[pyclass]
pub struct Diag1DLayer {
    name: String,
    /// `[np.diag, np.zeros_like]`.
    funcs: Vec<Py<PyAny>>,
    /// Shared kwargs (empty — the per-block `shape` rides in `CallKw`).
    kwargs: Py<PyAny>,
    /// Shared literals `[meta]` (the 2-D prototype for `np.zeros_like`).
    literals: Vec<Py<PyAny>>,
    /// The single input array name (`ArgSlot::Dep` uses `name_idx == 0`).
    dep_names: Vec<String>,
    /// 1-D input chunk sizes; the block grid is `chunks_1d.len()` square.
    chunks_1d: Vec<i64>,
}

#[pymethods]
impl Diag1DLayer {
    #[new]
    fn new(
        name: String,
        diag_fn: Py<PyAny>,
        zeros_like_fn: Py<PyAny>,
        kwargs: Py<PyAny>,
        meta: Py<PyAny>,
        dep_name: String,
        chunks_1d: Vec<i64>,
    ) -> Self {
        Self {
            name,
            funcs: vec![diag_fn, zeros_like_fn],
            kwargs,
            literals: vec![meta],
            dep_names: vec![dep_name],
            chunks_1d,
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

impl Diag1DLayer {
    fn expand(&self) -> Expanded<'_> {
        let n = self.chunks_1d.len();
        let mut tasks = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let coord = vec![i as u32, j as u32];
                if i == j {
                    // np.diag(x_block_i)
                    tasks.push(NeutralTask {
                        name_idx: 0,
                        coord,
                        compute: Compute::Call { func_idx: 0 },
                        slots: vec![ArgSlot::Dep {
                            name_idx: 0,
                            coord: vec![i as u32],
                        }],
                    });
                } else {
                    // np.zeros_like(meta, shape=(chunks_1d[i], chunks_1d[j]))
                    tasks.push(NeutralTask {
                        name_idx: 0,
                        coord,
                        compute: Compute::CallKw {
                            func_idx: 1,
                            kwargs: vec![(
                                "shape".to_string(),
                                ArgSlot::IntTuple(vec![self.chunks_1d[i], self.chunks_1d[j]]),
                            )],
                        },
                        slots: vec![ArgSlot::Literal(0)],
                    });
                }
            }
        }

        Expanded {
            names: vec![&self.name],
            funcs: self.funcs.iter().collect(),
            kwargs: &self.kwargs,
            literals: &self.literals,
            dep_names: &self.dep_names,
            tasks,
        }
    }
}

#[pyclass]
pub struct Diag2DSimpleLayer {
    name: String,
    func: Py<PyAny>,
    kwargs: Py<PyAny>,
    dep_names: Vec<String>,
    /// Number of diagonal blocks (the 2-D input is square in blocks).
    nblocks: usize,
}

#[pymethods]
impl Diag2DSimpleLayer {
    #[new]
    fn new(
        name: String,
        diag_fn: Py<PyAny>,
        kwargs: Py<PyAny>,
        dep_name: String,
        nblocks: usize,
    ) -> Self {
        Self {
            name,
            func: diag_fn,
            kwargs,
            dep_names: vec![dep_name],
            nblocks,
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

impl Diag2DSimpleLayer {
    fn expand(&self) -> Expanded<'_> {
        let tasks = (0..self.nblocks)
            .map(|i| NeutralTask {
                name_idx: 0,
                coord: vec![i as u32],
                compute: Compute::Call { func_idx: 0 },
                // np.diag(x[(i, i)])
                slots: vec![ArgSlot::Dep {
                    name_idx: 0,
                    coord: vec![i as u32, i as u32],
                }],
            })
            .collect();

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
