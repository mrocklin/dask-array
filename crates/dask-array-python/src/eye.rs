//! Eye layer (`da.eye`): 2-D indexed creation with a per-block choice of func.
//!
//! Each output block `(i, j)` is either on/near the `k`-diagonal — then it calls
//! `np.eye(vchunk, hchunk, local_k, dtype)` — or it's an all-zeros block, calling
//! `np.zeros((vchunk, hchunk), dtype)`. The diagonal test and `local_k` are pure
//! integer arithmetic over the `(i, j)` grid, so this is the *multi-func* shape
//! (like rechunk's split/merge): two entries in `funcs`, each task picks one via
//! [`Compute::Call`]. `dtype` is shared across blocks, so it rides in `literals`;
//! there are no dependencies.
//!
//! Mirrors the legacy `_layer` in `dask_array/creation/_eye.py` exactly: the
//! `(j - i - 1) * chunk_size <= k <= (j - i + 1) * chunk_size` bounds and the
//! `local_k = k - (j - i) * chunk_size` offset. `chunk_size` is `chunks[0][0]`
//! (the first vertical chunk size) per the expr's `_chunk_size`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask, Num};

#[pyclass]
pub struct EyeLayer {
    name: String,
    /// `[np.eye, np.zeros]`; `Compute::Call{func_idx}` picks per block.
    funcs: Vec<Py<PyAny>>,
    /// Shared kwargs (empty — everything is positional).
    kwargs: Py<PyAny>,
    /// Shared literals `[dtype]`, referenced by `ArgSlot::Literal(0)`.
    literals: Vec<Py<PyAny>>,
    /// Vertical (row) chunk sizes; `vchunks.len()` is the block count on dim 0.
    vchunks: Vec<i64>,
    /// Horizontal (column) chunk sizes; `hchunks.len()` is the block count on dim 1.
    hchunks: Vec<i64>,
    /// Diagonal positioning scale (`chunks[0][0]`).
    chunk_size: i64,
    /// Diagonal index (0 = main diagonal).
    k: i64,
}

#[pymethods]
impl EyeLayer {
    /// `eye_fn`/`zeros_fn`: `np.eye`/`np.zeros` (the two per-block funcs). `dtype`:
    /// shared per-block literal. `vchunks`/`hchunks`/`chunk_size`/`k`: the grid
    /// params; the diagonal test runs in Rust.
    #[new]
    fn new(
        name: String,
        eye_fn: Py<PyAny>,
        zeros_fn: Py<PyAny>,
        kwargs: Py<PyAny>,
        dtype: Py<PyAny>,
        vchunks: Vec<i64>,
        hchunks: Vec<i64>,
        chunk_size: i64,
        k: i64,
    ) -> Self {
        Self {
            name,
            funcs: vec![eye_fn, zeros_fn],
            kwargs,
            literals: vec![dtype],
            vchunks,
            hchunks,
            chunk_size,
            k,
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

impl EyeLayer {
    fn expand(&self) -> Expanded<'_> {
        let mut tasks = Vec::with_capacity(self.vchunks.len() * self.hchunks.len());
        for (i, &vchunk) in self.vchunks.iter().enumerate() {
            for (j, &hchunk) in self.hchunks.iter().enumerate() {
                // Block (i, j) holds part of the k-diagonal iff
                //   (j - i - 1) * chunk_size <= k <= (j - i + 1) * chunk_size
                let diag = (j as i64) - (i as i64);
                let on_diag = (diag - 1) * self.chunk_size <= self.k
                    && self.k <= (diag + 1) * self.chunk_size;
                let (func_idx, slots) = if on_diag {
                    let local_k = self.k - diag * self.chunk_size;
                    // np.eye(vchunk, hchunk, local_k, dtype)
                    (
                        0,
                        vec![
                            ArgSlot::Scalar(Num::Int(vchunk)),
                            ArgSlot::Scalar(Num::Int(hchunk)),
                            ArgSlot::Scalar(Num::Int(local_k)),
                            ArgSlot::Literal(0), // dtype
                        ],
                    )
                } else {
                    // np.zeros((vchunk, hchunk), dtype)
                    (
                        1,
                        vec![
                            ArgSlot::IntTuple(vec![vchunk, hchunk]),
                            ArgSlot::Literal(0), // dtype
                        ],
                    )
                };
                tasks.push(NeutralTask {
                    nbytes: 0,
                    name_idx: 0,
                    coord: vec![i as u32, j as u32],
                    compute: Compute::Call { func_idx },
                    slots,
                });
            }
        }

        Expanded {
            names: vec![&self.name],
            funcs: self.funcs.iter().collect(),
            kwargs: &self.kwargs,
            literals: &self.literals,
            dep_names: &[],
            tasks,
        }
    }
}
