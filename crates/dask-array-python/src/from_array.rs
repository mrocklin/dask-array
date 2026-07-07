//! Array-like `from_array` data-source layer (the getter path).
//!
//! `from_array(arr, chunks=...)` over a generic slicing target (zarr, h5py,
//! icechunk, or any object with `.shape`/`.dtype`/`__getitem__`) reads each
//! output block by slicing the source array. This layer mirrors dask's
//! `graph_from_arraylike(inline_array=False)` exactly: the source array is placed
//! once as a data node (key `"original-<name>"`, value the array) and every block
//! task is `getitem(TaskRef("original-<name>"), block_slice)` — or, with
//! `inline_array=True`, the array embedded directly in each task.
//!
//! Unlike the computed layers this is a *source*, so the per-block work is numpy
//! slicing (inherently Python). What Rust accelerates is the O(n_tasks)
//! expansion: building the cartesian product of per-dimension slices and the key
//! strings without re-lowering a legacy dict through dask's converters (the ~8x
//! tax the generic `GraphRecordsLayer` pays). The plain-ndarray case is handled
//! by the eager-slice `FromArrayLayer` in Python; this layer is the array-like
//! getter case.
//!
//! It does not go through `common.rs`'s neutral-task machinery: that keys every
//! task by a `(name, *coord)` tuple, whereas the data node here has a bare-string
//! key and a literal value. A bespoke expansion is flatter than teaching the
//! shared converters a from_array-only concept.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString, PyTuple};

use crate::common::{ArgSlot, Compute, Expanded, IndexElem, NeutralTask};

/// Per-dimension chunk sizes and the region offset (start) applied to that
/// dimension's slices. `offset` is 0 when there is no `_region`.
struct Dim {
    sizes: Vec<i64>,
    offset: i64,
}

#[pyclass]
pub struct FromArrayGetterLayer {
    name: String,
    /// The source array, placed once as the data node.
    array: Py<PyAny>,
    /// `getter` / `getter_nofancy` — the per-block slicing function.
    getitem: Py<PyAny>,
    dims: Vec<Dim>,
    /// Embed the array in each task (`inline_array=True`) instead of a data node.
    inline_array: bool,
    /// `Some((asarray, lock))` emits the 5-arg getter call `(arr, slc, asarray,
    /// lock)`; `None` the 3-arg call `(arr, slc)` (the asarray=True/no-lock common
    /// case), matching `graph_from_arraylike`'s kwargs branch.
    extra_args: Option<(bool, bool)>,
}

impl FromArrayGetterLayer {
    fn original_name(&self) -> String {
        format!("original-{}", self.name)
    }

    /// Build the per-dimension `slice(start, stop)` objects (region offset
    /// applied), as Python objects, once — block slices index into these.
    fn dim_slices<'py>(
        &self,
        slice_cls: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
        let mut out = Vec::with_capacity(self.dims.len());
        for dim in &self.dims {
            let mut start = dim.offset;
            let mut slices = Vec::with_capacity(dim.sizes.len());
            for &sz in &dim.sizes {
                let stop = start + sz;
                slices.push(slice_cls.call1((start, stop))?);
                start = stop;
            }
            out.push(slices);
        }
        Ok(out)
    }

    /// The block index tuple `(start_i_0, ..., start_i_n)` in product order.
    /// `numblocks` is the per-dim block count.
    fn each_block(numblocks: &[usize]) -> impl Iterator<Item = Vec<usize>> + '_ {
        let total: usize = numblocks.iter().product();
        let ndim = numblocks.len();
        (0..total).map(move |flat| {
            let mut coord = vec![0usize; ndim];
            let mut rem = flat;
            for d in (0..ndim).rev() {
                coord[d] = rem % numblocks[d];
                rem /= numblocks[d];
            }
            coord
        })
    }
}

#[pymethods]
impl FromArrayGetterLayer {
    /// `dims`: per dimension `(chunk_sizes, region_offset)`. `extra_args`:
    /// `Some((asarray, lock))` for the 5-arg getter call, else `None`.
    #[new]
    #[pyo3(signature = (name, array, getitem, dims, inline_array, extra_args=None))]
    fn new(
        name: String,
        array: Py<PyAny>,
        getitem: Py<PyAny>,
        dims: Vec<(Vec<i64>, i64)>,
        inline_array: bool,
        extra_args: Option<(bool, bool)>,
    ) -> Self {
        let dims = dims
            .into_iter()
            .map(|(sizes, offset)| Dim { sizes, offset })
            .collect();
        Self {
            name,
            array,
            getitem,
            dims,
            inline_array,
            extra_args,
        }
    }

    /// The legacy/correctness path: `{key: dask Task | data}`, matching
    /// `graph_from_arraylike(inline_array=...)`. The data node holds the bare
    /// array (a legacy HLG data node); each block is a `dask._task_spec.Task`.
    fn to_dask_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let ts = py.import("dask._task_spec")?;
        let task_cls = ts.getattr("Task")?;
        let taskref_cls = ts.getattr("TaskRef")?;
        let slice_cls = py.import("builtins")?.getattr("slice")?;
        let getter = self.getitem.bind(py);

        let dsk = PyDict::new(py);
        let dim_slices = self.dim_slices(&slice_cls)?;
        let numblocks: Vec<usize> = self.dims.iter().map(|d| d.sizes.len()).collect();
        let original = self.original_name();

        // The argument referencing the source array is identical across every
        // block (a TaskRef to the one data node, or the inlined array), so build
        // it once and clone the handle per task.
        let arr_arg: Bound<'py, PyAny> = if self.inline_array {
            self.array.bind(py).clone()
        } else {
            dsk.set_item(PyString::new(py, &original), self.array.bind(py))?;
            taskref_cls.call1((PyString::new(py, &original),))?
        };

        for coord in Self::each_block(&numblocks) {
            let key = key_tuple(py, &self.name, &coord)?;
            let idx = index_tuple(py, &dim_slices, &coord)?;
            let mut call: Vec<Bound<'py, PyAny>> = vec![
                key.clone().into_any(),
                getter.clone(),
                arr_arg.clone(),
                idx.into_any(),
            ];
            if let Some((asarray, lock)) = self.extra_args {
                call.push(asarray.into_pyobject(py)?.to_owned().into_any());
                call.push(lock.into_pyobject(py)?.to_owned().into_any());
            }
            let t = task_cls.call1(PyTuple::new(py, call)?)?;
            dsk.set_item(key, t)?;
        }
        Ok(dsk)
    }

    /// The Frisky fast path: flat `(key, func, args, kwargs, deps)` records.
    /// The data node is a `toolz.identity` task over the bare array (Frisky has
    /// no data node); each block is a `getter` call whose array arg is a
    /// `TaskRef` (a dep), matching `GraphRecordsLayer`'s output for this case.
    fn to_task_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let taskref_cls = py.import("dask._task_spec")?.getattr("TaskRef")?;
        let slice_cls = py.import("builtins")?.getattr("slice")?;
        let empty_kwargs = PyDict::new(py);
        let getter = self.getitem.bind(py);

        let records = PyList::empty(py);
        let dim_slices = self.dim_slices(&slice_cls)?;
        let numblocks: Vec<usize> = self.dims.iter().map(|d| d.sizes.len()).collect();
        let original = self.original_name();

        // The array arg is identical across every block (one shared TaskRef to the
        // data node, or the inlined array), so build it once and clone the handle
        // per task — the per-task TaskRef construction was the dominant cost.
        let inline = self.inline_array;
        let arr_arg: Bound<'py, PyAny> = if inline {
            self.array.bind(py).clone()
        } else {
            // Data node: identity over the bare array, no deps. Key is the bare
            // string (str() of a string is itself).
            let identity = py.import("toolz")?.getattr("identity")?;
            let dn_args = PyTuple::new(py, [self.array.bind(py).clone()])?;
            records.append((
                original.clone(),
                identity,
                dn_args,
                empty_kwargs.clone(),
                PyList::empty(py),
            ))?;
            taskref_cls.call1((PyString::new(py, &original),))?
        };

        for coord in Self::each_block(&numblocks) {
            let key = key_string(&self.name, &coord);
            let idx = index_tuple(py, &dim_slices, &coord)?;
            // Fresh deps list per record (avoids any shared-mutable-state risk).
            let deps = if inline {
                PyList::empty(py)
            } else {
                PyList::new(py, [&original])?
            };
            let mut args: Vec<Bound<'py, PyAny>> = vec![arr_arg.clone(), idx.into_any()];
            if let Some((asarray, lock)) = self.extra_args {
                args.push(asarray.into_pyobject(py)?.to_owned().into_any());
                args.push(lock.into_pyobject(py)?.to_owned().into_any());
            }
            let args_tuple = PyTuple::new(py, args)?;
            records.append((key, getter.clone(), args_tuple, empty_kwargs.clone(), deps))?;
        }
        Ok(records)
    }

    /// The Frisky binary-records fast path: one LAYER chunk for the N getter
    /// tasks, built through the shared `common` machinery. The source array is
    /// NOT in the chunk — it ships once as a plain "holder" record the Python
    /// layer emits (`chunk_side_records`), which each getter references via a
    /// `Dep` slot with an empty coord (key `('original-<name>',)`). The per-block
    /// slice is an `Index` slot derived from the chunk sizes. Declines (falls back
    /// to `to_task_records`) when the array is inlined per task or the 5-arg
    /// getter (asarray/lock) is needed — neither is expressible as a shared chunk.
    fn to_records_chunk<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        if self.inline_array || self.extra_args.is_some() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "from_array getter: inline_array / extra getter args are not on the binary path",
            ));
        }
        let dep_names = vec![self.original_name()];
        let empty_kwargs: Py<PyAny> = PyDict::new(py).unbind().into_any();
        let numblocks: Vec<usize> = self.dims.iter().map(|d| d.sizes.len()).collect();
        let total: usize = numblocks.iter().product();
        let mut tasks: Vec<NeutralTask> = Vec::with_capacity(total);
        for coord in Self::each_block(&numblocks) {
            let mut index: Vec<IndexElem> = Vec::with_capacity(self.dims.len());
            for (d, dim) in self.dims.iter().enumerate() {
                let start: i64 = dim.offset + dim.sizes[..coord[d]].iter().sum::<i64>();
                let stop = start + dim.sizes[coord[d]];
                index.push(IndexElem::Slice {
                    start: Some(start),
                    stop: Some(stop),
                    step: None,
                });
            }
            tasks.push(NeutralTask {
                name_idx: 0,
                coord: coord.iter().map(|&c| c as u32).collect(),
                compute: Compute::Call { func_idx: 0 },
                slots: vec![
                    ArgSlot::Dep {
                        name_idx: 0,
                        coord: Vec::new(),
                    },
                    ArgSlot::Index(index),
                ],
            });
        }
        let exp = Expanded {
            names: vec![&self.name],
            funcs: vec![&self.getitem],
            kwargs: &empty_kwargs,
            literals: &[],
            dep_names: &dep_names,
            tasks,
        };
        crate::common::to_records_chunk(py, &exp)
    }
}

/// `(name, *coord)` as a Python tuple key.
fn key_tuple<'py>(py: Python<'py>, name: &str, coord: &[usize]) -> PyResult<Bound<'py, PyTuple>> {
    let mut elems: Vec<Bound<'py, PyAny>> = Vec::with_capacity(coord.len() + 1);
    elems.push(PyString::new(py, name).into_any());
    for &c in coord {
        elems.push(c.into_pyobject(py)?.into_any());
    }
    PyTuple::new(py, elems)
}

/// `str((name, *coord))`, matching Python's tuple repr (Frisky keys by string).
fn key_string(name: &str, coord: &[usize]) -> String {
    if coord.is_empty() {
        format!("('{name}',)")
    } else {
        let inner: Vec<String> = coord.iter().map(|c| c.to_string()).collect();
        format!("('{name}', {})", inner.join(", "))
    }
}

/// The per-block index tuple `(dim_slices[0][coord[0]], ...)`.
fn index_tuple<'py>(
    py: Python<'py>,
    dim_slices: &[Vec<Bound<'py, PyAny>>],
    coord: &[usize],
) -> PyResult<Bound<'py, PyTuple>> {
    let elems: Vec<Bound<'py, PyAny>> = coord
        .iter()
        .enumerate()
        .map(|(d, &i)| dim_slices[d][i].clone())
        .collect();
    PyTuple::new(py, elems)
}
