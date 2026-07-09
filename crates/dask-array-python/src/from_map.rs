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
//! shared literals in `common`'s neutral task. So `FromMapLayer` has no
//! `to_records_chunk`: a literal isn't expressible in the Python-object-free
//! binary grammar, so it stays on the plain-records path.
//!
//! `FromMapBinaryLayer` is the pure-Rust variant for the common special case a
//! coalesced `concatenate(from_delayed(...))` produces: every block calls one
//! *shared* function (hoisted out into the layer's pickled-once `func`) and the
//! only per-block variation is arguments the binary grammar CAN hold — scalars,
//! strings (filenames!), int-tuples, and lists thereof. `FromMap._frisky_layer`
//! picks this layer when it applies and `FromMapLayer` otherwise. The per-block
//! args arrive pre-classified from Python as `(tag, payload)` descriptors
//! (`parse_slot`), so Rust just slots them and reuses the shared converters.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask, Num};

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
                nbytes: 0,
                name_idx: 0,
                coord: coord.clone(),
                compute: Compute::Call { func_idx: 0 },
                slots: vec![
                    ArgSlot::Literal(0),      // func
                    ArgSlot::Literal(2 + i),  // value for this block (C order)
                    ArgSlot::IntTuple(shape), // block (chunk) shape
                    ArgSlot::Literal(1),      // kwargs
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

/// Parse one Python arg descriptor `(tag, payload)` into an [`ArgSlot`]. The
/// `FromMap._frisky_layer` binary path classifies each per-block call argument in
/// Python (where the isinstance checks are natural) and hands Rust these tagged
/// pairs, so the mapping here is 1:1 and unambiguous:
///   `"i"` int scalar · `"f"` float scalar · `"s"` string · `"t"` int-tuple
///   (rebuilds a Python tuple) · `"l"` list (recursive; rebuilds a Python list).
fn parse_slot(d: &Bound<'_, PyAny>) -> PyResult<ArgSlot> {
    let tup = d.cast::<PyTuple>().map_err(|_| {
        PyValueError::new_err("from_map arg descriptor must be a (tag, payload) tuple")
    })?;
    let tag: String = tup.get_item(0)?.extract()?;
    let payload = tup.get_item(1)?;
    match tag.as_str() {
        "i" => Ok(ArgSlot::Scalar(Num::Int(payload.extract()?))),
        "f" => Ok(ArgSlot::Scalar(Num::Float(payload.extract()?))),
        "s" => Ok(ArgSlot::Str(payload.extract()?)),
        "t" => Ok(ArgSlot::IntTuple(payload.extract()?)),
        "l" => {
            let items = payload.cast::<PyList>().map_err(|_| {
                PyValueError::new_err("from_map 'l' descriptor payload must be a list")
            })?;
            let slots = items
                .iter()
                .map(|x| parse_slot(&x))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(ArgSlot::List(slots))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown from_map arg descriptor tag {other:?}"
        ))),
    }
}

/// Pure-Rust `from_map` for the shared-function special case (see the module
/// docstring). Each block is `func(args_list, block_shape)` where `func` is the
/// shared block wrapper (`dask_array.io._from_map._apply_args`, carrying the
/// hoisted user function + kwargs bound as `kwargs`), `args_list` is a
/// [`ArgSlot::List`] of that block's binary-expressible call args, and
/// `block_shape` an [`ArgSlot::IntTuple`]. No per-block Python literal → this
/// layer emits a binary `to_records_chunk`.
#[pyclass]
pub struct FromMapBinaryLayer {
    name: String,
    /// `_apply_args` — the shared block func. Its `_fn` / `_fkwargs` (the hoisted
    /// user function and its kwargs) ride in `kwargs`, bound once via
    /// `functools.partial` when the chunk is serialized.
    func: Py<PyAny>,
    /// Shared kwargs `{"_fn": <hoisted func>, "_fkwargs": <user kwargs>}`.
    kwargs: Py<PyAny>,
    /// Per-block call args (C order, matching the row-major block iteration),
    /// pre-classified into slots.
    block_args: Vec<Vec<ArgSlot>>,
    /// Chunk sizes per dimension; `chunks[d].len()` is the block count on `d`.
    chunks: Vec<Vec<i64>>,
}

#[pymethods]
impl FromMapBinaryLayer {
    #[new]
    fn new(
        py: Python<'_>,
        name: String,
        func: Py<PyAny>,
        kwargs: Py<PyAny>,
        block_args: Vec<Py<PyAny>>,
        chunks: Vec<Vec<i64>>,
    ) -> PyResult<Self> {
        let n_blocks: usize = if chunks.is_empty() {
            1
        } else {
            chunks.iter().map(|c| c.len()).product()
        };
        if block_args.len() != n_blocks {
            return Err(PyValueError::new_err(format!(
                "from_map: {} per-block arg lists for {} blocks",
                block_args.len(),
                n_blocks,
            )));
        }
        let mut parsed = Vec::with_capacity(block_args.len());
        for obj in &block_args {
            let lst = obj.bind(py).cast::<PyList>().map_err(|_| {
                PyValueError::new_err("from_map per-block args must be a list of descriptors")
            })?;
            let slots = lst
                .iter()
                .map(|d| parse_slot(&d))
                .collect::<PyResult<Vec<_>>>()?;
            parsed.push(slots);
        }
        Ok(Self {
            name,
            func,
            kwargs,
            block_args: parsed,
            chunks,
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

impl FromMapBinaryLayer {
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
                nbytes: 0,
                name_idx: 0,
                coord: coord.clone(),
                compute: Compute::Call { func_idx: 0 },
                // _apply_args(args_list, block_shape)  [_fn / _fkwargs ride in kwargs]
                slots: vec![
                    ArgSlot::List(self.block_args[i].clone()),
                    ArgSlot::IntTuple(shape),
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
            funcs: vec![&self.func],
            kwargs: &self.kwargs,
            literals: &[],
            dep_names: &[],
            tasks,
        }
    }
}
