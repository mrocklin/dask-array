//! No-dependency `from_map` layer.
//!
//! Each output block is `_map_block(func, value, block_shape, kwargs)` — a single
//! call over a per-block Python `value` (a delayed-call bundle, or a user datum),
//! reshaped to the block's chunk shape. It mirrors `FromMap._layer` exactly; the
//! win over the generic `GraphRecordsLayer` fallback is the O(n_tasks) expansion
//! in Rust rather than lowering a legacy dict and translating it.
//!
//! Like `from_array` (and unlike the computed layers), the per-block `value`,
//! the block `func`, and the `kwargs` are arbitrary Python objects — carried as
//! shared literals in `common`'s neutral task. So there is no `to_records_chunk`
//! here: a literal isn't expressible in the Python-object-free binary grammar,
//! so this layer stays on the plain-records path (which is why a `from_map`
//! group carries no shape/chunks/dtype metadata, same as `from_array`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask};

#[pyclass]
pub struct FromMapLayer {
    name: String,
    /// The shared task func, `dask_array.io._from_map._map_block`.
    map_block: Py<PyAny>,
    /// Shared `Call` kwargs — always empty (`_map_block` takes only positionals).
    /// One dict handle shared by every record (records are read-only once built,
    /// same as `from_array`); the fallback builds a fresh `{}` per record.
    empty_kwargs: Py<PyAny>,
    /// Shared literals: `[func, kwargs, value_0, ..., value_{n-1}]`, so the block
    /// func is slot `Literal(0)`, the user kwargs `Literal(1)`, and block `i`'s
    /// value `Literal(2 + i)` (values in row-major / C order, matching `coord`).
    literals: Vec<Py<PyAny>>,
    /// Chunk sizes per dimension; `chunks[d].len()` is the block count on `d`.
    chunks: Vec<Vec<i64>>,
}

#[pymethods]
impl FromMapLayer {
    #[new]
    fn new(
        py: Python<'_>,
        name: String,
        map_block: Py<PyAny>,
        func: Py<PyAny>,
        kwargs: Py<PyAny>,
        values: Vec<Py<PyAny>>,
        chunks: Vec<Vec<i64>>,
    ) -> PyResult<Self> {
        // One value per block. `from_map` already enforces this, but guard here
        // so a bad direct construction (or a future merge rule that miscomputes
        // the grid) is a clean error, not an out-of-bounds panic across FFI.
        let n_blocks: usize = chunks.iter().map(|c| c.len()).product();
        if values.len() != n_blocks {
            return Err(PyValueError::new_err(format!(
                "from_map: {} values for {} blocks (chunk grid {:?})",
                values.len(),
                n_blocks,
                chunks.iter().map(|c| c.len()).collect::<Vec<_>>(),
            )));
        }
        let mut literals = Vec::with_capacity(values.len() + 2);
        literals.push(func);
        literals.push(kwargs);
        literals.extend(values);
        Ok(Self {
            name,
            map_block,
            empty_kwargs: PyDict::new(py).into(),
            literals,
            chunks,
        })
    }

    fn to_dask_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        to_dask_graph(py, &self.expand())
    }

    fn to_task_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_task_records(py, &self.expand())
    }
}

impl FromMapLayer {
    fn expand(&self) -> Expanded<'_> {
        let ndim = self.chunks.len();
        let numblocks: Vec<usize> = self.chunks.iter().map(|c| c.len()).collect();
        let total: usize = if ndim == 0 {
            1
        } else {
            numblocks.iter().product()
        };
        let mut tasks = Vec::with_capacity(total);
        let mut coord = vec![0u32; ndim];

        for i in 0..total {
            let shape: Vec<i64> = (0..ndim)
                .map(|d| self.chunks[d][coord[d] as usize])
                .collect();
            tasks.push(NeutralTask {
                name_idx: 0,
                coord: coord.clone(),
                compute: Compute::Call { func_idx: 0 },
                slots: vec![
                    ArgSlot::Literal(0),       // func
                    ArgSlot::Literal(2 + i),   // value for this block (C order)
                    ArgSlot::IntTuple(shape),  // block (chunk) shape
                    ArgSlot::Literal(1),       // kwargs
                ],
            });

            for d in (0..ndim).rev() {
                coord[d] += 1;
                if (coord[d] as usize) < numblocks[d] {
                    break;
                }
                coord[d] = 0;
            }
        }

        Expanded {
            names: vec![&self.name],
            funcs: vec![&self.map_block],
            kwargs: &self.empty_kwargs,
            literals: &self.literals,
            dep_names: &[],
            tasks,
        }
    }
}
