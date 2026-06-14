//! Shared scaffolding for the layer modules.
//!
//! A layer expands (pure Rust) into an [`Expanded`] form: the shared per-layer
//! Python payloads (`funcs`, `kwargs`, literal values) plus a `Vec` of
//! [`NeutralTask`] records that are entirely raw Rust — an output key
//! (`name_idx` + coord), which function to call, and the argument
//! [`ArgSlot`]s, where a dependency is a `(dep_name_index, coord)` and a
//! literal is an index into the shared literals. Nothing per-task is a Python
//! object.
//!
//! Two generic converters consume that form: [`to_dask_graph`] builds a dask
//! task graph (the correctness path, validated by the test suite) and
//! [`to_task_records`] builds the plain `(key, func, args, kwargs, deps)`
//! records the Frisky client serializes. New layer kinds only implement the
//! expansion; the converters are shared.
//!
//! The form covers three task structures: the *fast path* (one shared func,
//! flat args, keys `(name, *coord)`) used by blockwise/creation; *nested /
//! variable-length dep args* ([`ArgSlot::List`]) used by reductions and
//! rechunk's `concatenate3`; and *per-task func + free-form keys* (multiple
//! `funcs`, multiple `names`, [`Compute::Alias`]) used by rechunk's
//! split/merge/alias tasks within one layer.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

/// One positional argument of a task, resolved per output block.
pub enum ArgSlot {
    /// Index into the layer's shared literal values.
    Literal(usize),
    /// A dependency: its key is `(dep_names[name_idx], *coord)`.
    Dep { name_idx: usize, coord: Vec<u32> },
    /// A per-block computed integer tuple (e.g. a creation block shape).
    IntTuple(Vec<i64>),
    /// A nested list of args (a reduction's lol_tuples, a rechunk's
    /// `concatenate3` argument). In the dask path it becomes a
    /// `dask._task_spec.List` so embedded `TaskRef`s register as dependencies;
    /// in the records path a plain Python list (the worker resolves
    /// placeholders nested in lists).
    List(Vec<ArgSlot>),
    /// A per-block getitem index tuple — each element a `slice(start, stop,
    /// step)` (Python `None` for a missing bound) or an integer (which drops
    /// that dimension). Used by rechunk's split slices and basic slicing.
    Index(Vec<IndexElem>),
}

/// One element of a getitem index tuple.
pub enum IndexElem {
    /// `slice(start, stop, step)`; a `None` maps to Python `None`.
    Slice { start: Option<i64>, stop: Option<i64>, step: Option<i64> },
    /// An integer index (drops the dimension).
    Int(i64),
}

/// What a task computes.
pub enum Compute {
    /// Apply `funcs[func_idx]` to the args (with the layer's shared kwargs).
    Call { func_idx: usize },
    /// An alias: the key resolves directly to its single dependency (`slots[0]`,
    /// a [`ArgSlot::Dep`]). The dask path emits a `dask._task_spec.Alias`; the
    /// records path emits a `toolz.identity` task (Frisky has no alias node).
    Alias,
}

/// One task in the expanded subgraph. Its key is `(names[name_idx], *coord)`.
pub struct NeutralTask {
    pub name_idx: usize,
    pub coord: Vec<u32>,
    pub compute: Compute,
    pub slots: Vec<ArgSlot>,
}

/// A layer's fully expanded subgraph. Shared payloads borrow from the layer;
/// the per-task records are owned and raw Rust.
pub struct Expanded<'a> {
    /// Names of the keys this layer produces; [`NeutralTask::name_idx`] indexes.
    pub names: Vec<&'a str>,
    /// Distinct task functions; [`Compute::Call`]'s `func_idx` indexes.
    pub funcs: Vec<&'a PyObject>,
    /// Shared keyword arguments applied to every `Call` (usually empty).
    pub kwargs: &'a PyObject,
    pub literals: &'a [PyObject],
    /// Names of dependencies referenced by [`ArgSlot::Dep`] (external arrays and
    /// this layer's own intermediates); the slot's `name_idx` indexes.
    pub dep_names: &'a [String],
    pub tasks: Vec<NeutralTask>,
}

/// Build the Python tuple key `(name, *coord)`.
fn key_tuple<'py>(py: Python<'py>, name: &str, coord: &[u32]) -> PyResult<Bound<'py, PyTuple>> {
    let mut elems: Vec<Bound<'py, PyAny>> = Vec::with_capacity(coord.len() + 1);
    elems.push(PyString::new(py, name).into_any());
    for c in coord {
        elems.push((*c as usize).into_pyobject(py)?.into_any());
    }
    PyTuple::new(py, elems)
}

/// Build the key string `str((name, *coord))`, matching Python's tuple repr so
/// it agrees with keys/deps produced elsewhere (Frisky keys tasks by string).
fn key_string(name: &str, coord: &[u32]) -> String {
    if coord.is_empty() {
        format!("('{name}',)")
    } else {
        let inner: Vec<String> = coord.iter().map(|c| c.to_string()).collect();
        format!("('{name}', {})", inner.join(", "))
    }
}

fn int_tuple<'py>(py: Python<'py>, vals: &[i64]) -> PyResult<Bound<'py, PyTuple>> {
    let mut elems: Vec<Bound<'py, PyAny>> = Vec::with_capacity(vals.len());
    for v in vals {
        elems.push(v.into_pyobject(py)?.into_any());
    }
    PyTuple::new(py, elems)
}

fn opt_obj<'py>(py: Python<'py>, v: Option<i64>) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Some(x) => x.into_pyobject(py)?.into_any(),
        None => py.None().into_bound(py),
    })
}

/// Build a getitem index tuple — `slice(start, stop, step)` or an int per
/// element. `slice_cls` is Python's `slice` builtin (fetched once per convert).
fn index_tuple<'py>(
    py: Python<'py>,
    elems: &[IndexElem],
    slice_cls: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut out: Vec<Bound<'py, PyAny>> = Vec::with_capacity(elems.len());
    for e in elems {
        let item = match e {
            IndexElem::Slice { start, stop, step } => {
                slice_cls.call1((opt_obj(py, *start)?, opt_obj(py, *stop)?, opt_obj(py, *step)?))?
            }
            IndexElem::Int(i) => i.into_pyobject(py)?.into_any(),
        };
        out.push(item);
    }
    PyTuple::new(py, out)
}

/// Build one task argument for the dask path. A dependency becomes a `TaskRef`,
/// a nested list a `dask._task_spec.List` (so its embedded `TaskRef`s register
/// as dependencies).
fn build_arg_dask<'py>(
    py: Python<'py>,
    exp: &Expanded,
    slot: &ArgSlot,
    taskref_cls: &Bound<'py, PyAny>,
    list_cls: &Bound<'py, PyAny>,
    slice_cls: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match slot {
        ArgSlot::Literal(i) => Ok(exp.literals[*i].bind(py).clone()),
        ArgSlot::Dep { name_idx, coord } => {
            let dep_key = key_tuple(py, &exp.dep_names[*name_idx], coord)?;
            Ok(taskref_cls.call1((dep_key,))?)
        }
        ArgSlot::IntTuple(v) => Ok(int_tuple(py, v)?.into_any()),
        ArgSlot::Index(s) => Ok(index_tuple(py, s, slice_cls)?.into_any()),
        ArgSlot::List(items) => {
            let mut elems: Vec<Bound<'py, PyAny>> = Vec::with_capacity(items.len());
            for it in items {
                elems.push(build_arg_dask(py, exp, it, taskref_cls, list_cls, slice_cls)?);
            }
            // dask._task_spec.List(*elems)
            Ok(list_cls.call1(PyTuple::new(py, elems)?)?)
        }
    }
}

/// Generic correctness/legacy path: convert the expanded form to a dask task
/// graph (`{key: dask._task_spec.Task | Alias}`). Builds `Task`/`TaskRef`/`List`
/// via pyo3, so dask-array's existing single-threaded test suite validates the
/// Rust expansion. This is the only place that materializes Python per task, and
/// only for the dask path.
pub fn to_dask_graph<'py>(py: Python<'py>, exp: &Expanded) -> PyResult<Bound<'py, PyDict>> {
    let ts = py.import("dask._task_spec")?;
    let task_cls = ts.getattr("Task")?;
    let taskref_cls = ts.getattr("TaskRef")?;
    let list_cls = ts.getattr("List")?;
    let alias_cls = ts.getattr("Alias")?;
    let slice_cls = py.import("builtins")?.getattr("slice")?;
    let kwargs = exp.kwargs.bind(py).downcast::<PyDict>()?.clone();

    let dsk = PyDict::new(py);
    for task in &exp.tasks {
        let key = key_tuple(py, exp.names[task.name_idx], &task.coord)?;
        match &task.compute {
            Compute::Alias => {
                let src = match &task.slots[0] {
                    ArgSlot::Dep { name_idx, coord } => key_tuple(py, &exp.dep_names[*name_idx], coord)?,
                    _ => return Err(PyValueError::new_err("alias source must be a Dep")),
                };
                let alias = alias_cls.call1((key.clone(), src))?;
                dsk.set_item(key, alias)?;
            }
            Compute::Call { func_idx } => {
                let func = exp.funcs[*func_idx].bind(py);
                let mut call: Vec<Bound<'py, PyAny>> = Vec::with_capacity(task.slots.len() + 2);
                call.push(key.clone().into_any());
                call.push(func.clone());
                for slot in &task.slots {
                    call.push(build_arg_dask(py, exp, slot, &taskref_cls, &list_cls, &slice_cls)?);
                }
                let t = task_cls.call(PyTuple::new(py, call)?, Some(&kwargs))?;
                dsk.set_item(key, t)?;
            }
        }
    }
    Ok(dsk)
}

/// Build one task argument for the records path, collecting dependency key
/// strings into `deps`. A dependency becomes a `TaskRef` (Frisky converts it to
/// its own placeholder at submit time); a nested list a plain Python list (the
/// worker resolves placeholders nested in lists).
fn build_arg_rec<'py>(
    py: Python<'py>,
    exp: &Expanded,
    slot: &ArgSlot,
    taskref_cls: &Bound<'py, PyAny>,
    slice_cls: &Bound<'py, PyAny>,
    deps: &mut Vec<String>,
) -> PyResult<Bound<'py, PyAny>> {
    match slot {
        ArgSlot::Literal(i) => Ok(exp.literals[*i].bind(py).clone()),
        ArgSlot::Dep { name_idx, coord } => {
            let dep_key = key_tuple(py, &exp.dep_names[*name_idx], coord)?;
            deps.push(key_string(&exp.dep_names[*name_idx], coord));
            Ok(taskref_cls.call1((dep_key,))?)
        }
        ArgSlot::IntTuple(v) => Ok(int_tuple(py, v)?.into_any()),
        ArgSlot::Index(s) => Ok(index_tuple(py, s, slice_cls)?.into_any()),
        ArgSlot::List(items) => {
            let out = PyList::empty(py);
            for it in items {
                out.append(build_arg_rec(py, exp, it, taskref_cls, slice_cls, deps)?)?;
            }
            Ok(out.into_any())
        }
    }
}

/// Generic Frisky path: convert the expanded form to a plain list of task
/// records the Frisky client serializes as it would any task graph. Each record
/// is `(key, func, args, kwargs, deps)`:
///   - `key`  — `str((name, *coord))` (built in Rust).
///   - `func` — one of the layer's shared functions (an alias uses
///     `toolz.identity`); `kwargs` — the layer's shared dict.
///   - `args` — a Python tuple; a dependency is a dask `TaskRef(dep_key_tuple)`
///     (Frisky converts it to its own placeholder at submit time), a literal is
///     the shared Python value, nested deps are plain Python lists.
///   - `deps` — the dependency key strings (built in Rust).
///
/// This mirrors a dask Task per task (no shared template, no packed buffers, no
/// pickle tricks); each task is fully self-contained, so any layer kind can emit
/// records the same way and the Frisky client stays generic.
pub fn to_task_records<'py>(py: Python<'py>, exp: &Expanded) -> PyResult<Bound<'py, PyList>> {
    let taskref_cls = py.import("dask._task_spec")?.getattr("TaskRef")?;
    let slice_cls = py.import("builtins")?.getattr("slice")?;
    let kwargs = exp.kwargs.bind(py);
    let mut identity: Option<Bound<'py, PyAny>> = None;

    let records = PyList::empty(py);
    for task in &exp.tasks {
        let key = key_string(exp.names[task.name_idx], &task.coord);
        let mut deps: Vec<String> = Vec::new();
        match &task.compute {
            Compute::Alias => {
                let (src_name, src_coord) = match &task.slots[0] {
                    ArgSlot::Dep { name_idx, coord } => (&exp.dep_names[*name_idx], coord),
                    _ => return Err(PyValueError::new_err("alias source must be a Dep")),
                };
                let dep_key = key_tuple(py, src_name, src_coord)?;
                deps.push(key_string(src_name, src_coord));
                let func = match &identity {
                    Some(f) => f.clone(),
                    None => {
                        let f = py.import("toolz")?.getattr("identity")?;
                        identity = Some(f.clone());
                        f
                    }
                };
                let args = PyTuple::new(py, [taskref_cls.call1((dep_key,))?])?;
                records.append((key, func, args, kwargs.clone(), deps))?;
            }
            Compute::Call { func_idx } => {
                let func = exp.funcs[*func_idx].bind(py);
                let mut args: Vec<Bound<'py, PyAny>> = Vec::with_capacity(task.slots.len());
                for slot in &task.slots {
                    args.push(build_arg_rec(py, exp, slot, &taskref_cls, &slice_cls, &mut deps)?);
                }
                let args_tuple = PyTuple::new(py, args)?;
                records.append((key, func.clone(), args_tuple, kwargs.clone(), deps))?;
            }
        }
    }
    Ok(records)
}
