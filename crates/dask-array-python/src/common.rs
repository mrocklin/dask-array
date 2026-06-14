//! Shared scaffolding for the layer modules.
//!
//! A layer expands (pure Rust) into an [`Expanded`] form: the shared per-layer
//! Python payloads (`func`, `kwargs`, literal values) plus a `Vec` of per-task
//! records that are entirely raw Rust — output coordinate + [`ArgSlot`]s, where
//! a dependency is a `(dep_name_index, coord)` and a literal is an index into
//! the shared literals. Nothing per-task is a Python object.
//!
//! Two generic converters consume that form: [`to_dask_graph`] builds a dask
//! task graph (the correctness path, validated by the test suite) and
//! [`to_frisky_tasks`] builds the compact form the Frisky client serializes.
//! New layer kinds only implement the expansion; the converters are shared.

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
}

/// A layer's fully expanded subgraph. Shared payloads borrow from the layer;
/// the per-task records are owned and raw Rust.
pub struct Expanded<'a> {
    pub name: &'a str,
    pub func: &'a PyObject,
    pub kwargs: &'a PyObject,
    pub literals: &'a [PyObject],
    pub dep_names: &'a [String],
    pub tasks: Vec<(Vec<u32>, Vec<ArgSlot>)>,
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

/// Generic correctness/legacy path: convert the expanded form to a dask task
/// graph (`{key: dask._task_spec.Task}`). Builds `Task`/`TaskRef` via pyo3, so
/// dask-array's existing single-threaded test suite validates the Rust
/// expansion. This is the only place that materializes Python per task, and
/// only for the dask path.
pub fn to_dask_graph<'py>(py: Python<'py>, exp: &Expanded) -> PyResult<Bound<'py, PyDict>> {
    let ts = py.import("dask._task_spec")?;
    let task_cls = ts.getattr("Task")?;
    let taskref_cls = ts.getattr("TaskRef")?;
    let func = exp.func.bind(py);
    let kwargs = exp.kwargs.bind(py).downcast::<PyDict>()?.clone();

    let dsk = PyDict::new(py);
    for (coord, slots) in &exp.tasks {
        let key = key_tuple(py, exp.name, coord)?;
        let mut call: Vec<Bound<'py, PyAny>> = Vec::with_capacity(slots.len() + 2);
        call.push(key.clone().into_any());
        call.push(func.clone());
        for slot in slots {
            match slot {
                ArgSlot::Literal(i) => call.push(exp.literals[*i].bind(py).clone()),
                ArgSlot::Dep { name_idx, coord } => {
                    let dep_key = key_tuple(py, &exp.dep_names[*name_idx], coord)?;
                    call.push(taskref_cls.call1((dep_key,))?);
                }
                ArgSlot::IntTuple(v) => call.push(int_tuple(py, v)?.into_any()),
            }
        }
        let task = task_cls.call(PyTuple::new(py, call)?, Some(&kwargs))?;
        dsk.set_item(key, task)?;
    }
    Ok(dsk)
}

/// Generic Frisky path: convert the expanded form to a plain list of task
/// records the Frisky client serializes as it would any task graph. Each record
/// is `(key, func, args, kwargs, deps)`:
///   - `key`  — `str((name, *coord))` (built in Rust).
///   - `func` / `kwargs` — the layer's shared Python objects (one ref per task).
///   - `args` — a Python tuple; a dependency is a dask `TaskRef(dep_key_tuple)`
///     (dask-native — Frisky converts it to its own placeholder at submit time),
///     a literal is the shared Python value, a per-block value is an int tuple.
///   - `deps` — the dependency key strings (built in Rust).
///
/// This mirrors a dask Task per task (no shared template, no packed buffers, no
/// pickle tricks); each task is fully self-contained, so any layer kind can emit
/// records the same way and the Frisky client stays generic.
pub fn to_task_records<'py>(py: Python<'py>, exp: &Expanded) -> PyResult<Bound<'py, PyList>> {
    let taskref_cls = py.import("dask._task_spec")?.getattr("TaskRef")?;
    let func = exp.func.bind(py);
    let kwargs = exp.kwargs.bind(py);

    let records = PyList::empty(py);
    for (coord, slots) in &exp.tasks {
        let key = key_string(exp.name, coord);
        let mut args: Vec<Bound<'py, PyAny>> = Vec::with_capacity(slots.len());
        let mut deps: Vec<String> = Vec::new();
        for slot in slots {
            match slot {
                ArgSlot::Literal(i) => args.push(exp.literals[*i].bind(py).clone()),
                ArgSlot::Dep { name_idx, coord } => {
                    let dep_key = key_tuple(py, &exp.dep_names[*name_idx], coord)?;
                    args.push(taskref_cls.call1((dep_key,))?);
                    deps.push(key_string(&exp.dep_names[*name_idx], coord));
                }
                ArgSlot::IntTuple(v) => args.push(int_tuple(py, v)?.into_any()),
            }
        }
        let args_tuple = PyTuple::new(py, args)?;
        records.append((key, func.clone(), args_tuple, kwargs.clone(), deps))?;
    }
    Ok(records)
}
