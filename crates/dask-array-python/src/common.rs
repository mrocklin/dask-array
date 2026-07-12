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

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString, PyTuple};

/// One positional argument of a task, resolved per output block.
#[derive(Clone)]
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
    /// A per-block scalar (int or float) computed per output block — e.g. an
    /// arange block's start/stop, or an eye block's diagonal offset. Used by the
    /// indexed-creation layers, whose blocks differ only in such scalars.
    Scalar(Num),
    /// A per-block string — e.g. a filename passed to a shared loader function.
    /// Rebuilds a Python `str`; carried as its own slot so string-arg `from_map` /
    /// coalesced-`from_delayed` blocks stay pure-Rust instead of falling back.
    Str(String),
}

/// A scalar carried in an [`ArgSlot::Scalar`]. The int/float split is preserved
/// so the chunk function (`np.arange`, `np.eye`, …) sees the right Python type.
#[derive(Clone, Copy, FromPyObject)]
pub enum Num {
    Int(i64),
    Float(f64),
}

/// One element of a getitem index tuple.
#[derive(Clone)]
pub enum IndexElem {
    /// `slice(start, stop, step)`; a `None` maps to Python `None`.
    Slice {
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    },
    /// An integer index (drops the dimension).
    Int(i64),
}

/// What a task computes.
pub enum Compute {
    /// Apply `funcs[func_idx]` to the args (with the layer's shared kwargs).
    Call { func_idx: usize },
    /// Like [`Compute::Call`] but with extra per-task keyword arguments, merged
    /// over the layer's shared `kwargs` (per-task wins on a name clash). Each
    /// value is an [`ArgSlot`], so a kwarg can be a literal, scalar, int-tuple,
    /// nested list or dependency. Used where blocks differ by a keyword — e.g.
    /// `diag`'s off-diagonal `np.zeros_like(meta, shape=(m, n))`.
    CallKw {
        func_idx: usize,
        kwargs: Vec<(String, ArgSlot)>,
    },
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
    /// Expected output size in bytes (0 = unknown). Layers whose helper tasks
    /// aren't grid-shaped (rechunk splits, overlap halo slices, scan carries…)
    /// fill this at expansion time, where the exact extents are in hand; plain
    /// output-grid tasks may leave it 0 and rely on the collector's
    /// `stamp_expected_nbytes_chunk` pass, which fills zeros from the
    /// expression's `chunks`/`dtype` and never overwrites a nonzero value.
    pub nbytes: i64,
}

/// `itemsize * prod(sizes)` with saturation; 0 (unknown) when any factor is
/// non-positive — the consumer treats 0 as "no estimate", never as free-by-fiat.
pub fn grid_nbytes(itemsize: i64, sizes: impl IntoIterator<Item = i64>) -> i64 {
    if itemsize <= 0 {
        return 0;
    }
    let mut nbytes = itemsize as i128;
    for size in sizes {
        if size <= 0 {
            return 0;
        }
        nbytes *= size as i128;
        if nbytes > i64::MAX as i128 {
            return i64::MAX;
        }
    }
    nbytes as i64
}

/// A layer's fully expanded subgraph. Shared payloads borrow from the layer;
/// the per-task records are owned and raw Rust.
pub struct Expanded<'a> {
    /// Names of the keys this layer produces; [`NeutralTask::name_idx`] indexes.
    pub names: Vec<&'a str>,
    /// Distinct task functions; [`Compute::Call`]'s `func_idx` indexes.
    pub funcs: Vec<&'a Py<PyAny>>,
    /// Shared keyword arguments applied to every `Call` (usually empty).
    pub kwargs: &'a Py<PyAny>,
    pub literals: &'a [Py<PyAny>],
    /// Names of dependencies referenced by [`ArgSlot::Dep`] (external arrays and
    /// this layer's own intermediates); the slot's `name_idx` indexes.
    pub dep_names: &'a [String],
    pub tasks: Vec<NeutralTask>,
}

/// Build the Python tuple key `(name, *coord)`.
pub fn key_tuple<'py>(py: Python<'py>, name: &str, coord: &[u32]) -> PyResult<Bound<'py, PyTuple>> {
    let mut elems: Vec<Bound<'py, PyAny>> = Vec::with_capacity(coord.len() + 1);
    elems.push(PyString::new(py, name).into_any());
    for c in coord {
        elems.push((*c as usize).into_pyobject(py)?.into_any());
    }
    PyTuple::new(py, elems)
}

/// Build the key string `str((name, *coord))`, matching Python's tuple repr so
/// it agrees with keys/deps produced elsewhere (Frisky keys tasks by string).
pub fn key_string(name: &str, coord: &[u32]) -> String {
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

fn num_obj<'py>(py: Python<'py>, n: Num) -> PyResult<Bound<'py, PyAny>> {
    Ok(match n {
        Num::Int(i) => i.into_pyobject(py)?.into_any(),
        Num::Float(f) => f.into_pyobject(py)?.into_any(),
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
            IndexElem::Slice { start, stop, step } => slice_cls.call1((
                opt_obj(py, *start)?,
                opt_obj(py, *stop)?,
                opt_obj(py, *step)?,
            ))?,
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
        ArgSlot::Scalar(n) => num_obj(py, *n),
        ArgSlot::Str(s) => Ok(PyString::new(py, s).into_any()),
        ArgSlot::List(items) => {
            let mut elems: Vec<Bound<'py, PyAny>> = Vec::with_capacity(items.len());
            for it in items {
                elems.push(build_arg_dask(
                    py,
                    exp,
                    it,
                    taskref_cls,
                    list_cls,
                    slice_cls,
                )?);
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
    let kwargs = exp.kwargs.bind(py).cast::<PyDict>()?.clone();

    let dsk = PyDict::new(py);
    for task in &exp.tasks {
        let key = key_tuple(py, exp.names[task.name_idx], &task.coord)?;
        match &task.compute {
            Compute::Alias => {
                let src = match &task.slots[0] {
                    ArgSlot::Dep { name_idx, coord } => {
                        key_tuple(py, &exp.dep_names[*name_idx], coord)?
                    }
                    _ => return Err(PyValueError::new_err("alias source must be a Dep")),
                };
                let alias = alias_cls.call1((key.clone(), src))?;
                dsk.set_item(key, alias)?;
            }
            Compute::Call { func_idx } | Compute::CallKw { func_idx, .. } => {
                let func = exp.funcs[*func_idx].bind(py);
                let mut call: Vec<Bound<'py, PyAny>> = Vec::with_capacity(task.slots.len() + 2);
                call.push(key.clone().into_any());
                call.push(func.clone());
                for slot in &task.slots {
                    call.push(build_arg_dask(
                        py,
                        exp,
                        slot,
                        &taskref_cls,
                        &list_cls,
                        &slice_cls,
                    )?);
                }
                // Per-task kwargs (CallKw) merge over the shared dict; plain
                // Call uses the shared dict directly.
                let kw = if let Compute::CallKw { kwargs: per, .. } = &task.compute {
                    let d = kwargs.copy()?;
                    for (name, slot) in per {
                        d.set_item(
                            name,
                            build_arg_dask(py, exp, slot, &taskref_cls, &list_cls, &slice_cls)?,
                        )?;
                    }
                    d
                } else {
                    kwargs.clone()
                };
                let t = task_cls.call(PyTuple::new(py, call)?, Some(&kw))?;
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
        ArgSlot::Scalar(n) => num_obj(py, *n),
        ArgSlot::Str(s) => Ok(PyString::new(py, s).into_any()),
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
            Compute::Call { func_idx } | Compute::CallKw { func_idx, .. } => {
                let func = exp.funcs[*func_idx].bind(py);
                let mut args: Vec<Bound<'py, PyAny>> = Vec::with_capacity(task.slots.len());
                for slot in &task.slots {
                    args.push(build_arg_rec(
                        py,
                        exp,
                        slot,
                        &taskref_cls,
                        &slice_cls,
                        &mut deps,
                    )?);
                }
                let args_tuple = PyTuple::new(py, args)?;
                // Per-task kwargs (CallKw) merge over the shared dict — a kwarg
                // value that is a Dep registers in `deps` like any other arg.
                let kw = if let Compute::CallKw { kwargs: per, .. } = &task.compute {
                    let d = kwargs.cast::<PyDict>()?.copy()?;
                    for (name, slot) in per {
                        d.set_item(
                            name,
                            build_arg_rec(py, exp, slot, &taskref_cls, &slice_cls, &mut deps)?,
                        )?;
                    }
                    d.into_any()
                } else {
                    kwargs.clone()
                };
                records.append((key, func.clone(), args_tuple, kw, deps))?;
            }
        }
    }
    Ok(records)
}

// --- Binary records protocol (consumed by frisky `records_proto.rs`) ----------
//
// One layer's `Expanded` serialized to a compact byte chunk so Frisky can build
// `TaskSpec`s in pure Rust, skipping the per-task Python object materialization
// `to_task_records` does. The Frisky bundle frames a sequence of these chunks
// with a framing version + layer count; this function emits exactly one LAYER.
// Each LAYER also self-describes its grammar version (its leading byte) so a
// grammar drift here that isn't matched in `records_proto.rs` is REJECTED (the
// decoder falls back) rather than silently misparsed. Bump
// `RECORDS_PROTOCOL_VERSION` here and the matching `CHUNK_GRAMMAR_VERSION` in
// `records_proto.rs` together on any grammar change. Little-endian.
//
//   LAYER := u8 grammar_version, STRLIST names, STRLIST dep_names, BYTESLIST funcs,
//            u32 n_tasks, TASK*
//   TASK  := u32 name_idx, COORD, i64 expected_nbytes, COMPUTE, u8 n_slots, SLOT*
//   COORD := u8 n, u32*n        COMPUTE := u8 (0 Call{u32 idx} | 2 Alias)
//   SLOT  := u8 tag (0 Dep{u32 name_idx, COORD} | 1 Index{u8 n, ELEM*}
//                    | 2 IntTuple{u8 n, i64*} | 3 List{u32 n, SLOT*}
//                    | 4 Scalar{NUM} | 5 Str{STR})
//   ELEM  := u8 (0 Slice{OPTI64 start,stop,step} | 1 Int{i64})
//   OPTI64:= u8 present, i64?   NUM := u8 (0 Int{i64} | 1 Float{f64})
//   STR := u32 len, utf8        BYTES := u32 len, bytes
//
// `Literal` slots and `CallKw` compute aren't expressible (a literal is an
// arbitrary Python object); this raises `NotImplementedError`, and the caller
// falls back to `to_task_records` for that layer.

/// Grammar version stamped at the head of every LAYER chunk; Frisky's decoder
/// (`records_proto::CHUNK_GRAMMAR_VERSION`) rejects a mismatch and falls back.
/// v2 added the `Str` slot (tag 5) for per-block string args (`from_map`
/// filenames). v3 added per-task `expected_nbytes` after the task coordinate.
/// Bump this and Frisky's `CHUNK_GRAMMAR_VERSION` together.
pub const RECORDS_PROTOCOL_VERSION: u8 = 3;

/// A count that the grammar stores in one byte (coords, slots, index elems —
/// all bounded by ndim / arg arity in practice). A layer that somehow exceeds
/// 255 falls back to `to_task_records` rather than silently truncating.
fn u8_count(n: usize, what: &str) -> PyResult<u8> {
    u8::try_from(n).map_err(|_| {
        pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "{what} count {n} exceeds binary-records u8 limit"
        ))
    })
}

fn w_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn w_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn w_str(buf: &mut Vec<u8>, s: &str) {
    w_u32(buf, s.len() as u32);
    buf.extend_from_slice(s.as_bytes());
}
fn w_bytes(buf: &mut Vec<u8>, b: &[u8]) {
    w_u32(buf, b.len() as u32);
    buf.extend_from_slice(b);
}
fn w_coord(buf: &mut Vec<u8>, coord: &[u32]) -> PyResult<()> {
    buf.push(u8_count(coord.len(), "coord")?);
    for c in coord {
        w_u32(buf, *c);
    }
    Ok(())
}
fn w_opt_i64(buf: &mut Vec<u8>, v: Option<i64>) {
    match v {
        Some(x) => {
            buf.push(1);
            w_i64(buf, x);
        }
        None => buf.push(0),
    }
}

fn w_slot(buf: &mut Vec<u8>, slot: &ArgSlot) -> PyResult<()> {
    match slot {
        ArgSlot::Dep { name_idx, coord } => {
            buf.push(0);
            w_u32(buf, *name_idx as u32);
            w_coord(buf, coord)?;
        }
        ArgSlot::Index(elems) => {
            buf.push(1);
            buf.push(u8_count(elems.len(), "index elems")?);
            for e in elems {
                match e {
                    IndexElem::Slice { start, stop, step } => {
                        buf.push(0);
                        w_opt_i64(buf, *start);
                        w_opt_i64(buf, *stop);
                        w_opt_i64(buf, *step);
                    }
                    IndexElem::Int(i) => {
                        buf.push(1);
                        w_i64(buf, *i);
                    }
                }
            }
        }
        ArgSlot::IntTuple(v) => {
            buf.push(2);
            buf.push(u8_count(v.len(), "int tuple")?);
            for x in v {
                w_i64(buf, *x);
            }
        }
        ArgSlot::List(items) => {
            buf.push(3);
            w_u32(buf, items.len() as u32);
            for it in items {
                w_slot(buf, it)?;
            }
        }
        ArgSlot::Scalar(n) => {
            buf.push(4);
            match n {
                Num::Int(i) => {
                    buf.push(0);
                    w_i64(buf, *i);
                }
                Num::Float(f) => {
                    buf.push(1);
                    buf.extend_from_slice(&f.to_le_bytes());
                }
            }
        }
        ArgSlot::Str(s) => {
            buf.push(5);
            w_str(buf, s);
        }
        ArgSlot::Literal(_) => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "literal arg not expressible in binary records",
            ));
        }
    }
    Ok(())
}

/// Serialize one layer's `Expanded` to a binary records LAYER chunk. The shared
/// functions are pickled once each (with the layer's shared kwargs bound via
/// `functools.partial` when non-empty, matching `to_task_records`'s effective
/// func); the per-task records are raw bytes.
pub fn to_records_chunk<'py>(py: Python<'py>, exp: &Expanded) -> PyResult<Bound<'py, PyBytes>> {
    let kwargs = exp.kwargs.bind(py).cast::<PyDict>()?;
    let kwargs_empty = kwargs.is_empty();
    let cloudpickle_dumps = py.import("cloudpickle")?.getattr("dumps")?;
    let partial = py.import("functools")?.getattr("partial")?;

    let mut buf: Vec<u8> = Vec::new();
    buf.push(RECORDS_PROTOCOL_VERSION);

    w_u32(&mut buf, exp.names.len() as u32);
    for n in &exp.names {
        w_str(&mut buf, n);
    }
    w_u32(&mut buf, exp.dep_names.len() as u32);
    for d in exp.dep_names {
        w_str(&mut buf, d);
    }
    w_u32(&mut buf, exp.funcs.len() as u32);
    for f in &exp.funcs {
        let effective: Bound<'py, PyAny> = if kwargs_empty {
            f.bind(py).clone()
        } else {
            partial.call((f.bind(py),), Some(kwargs))?
        };
        let bytes: Vec<u8> = cloudpickle_dumps.call1((effective,))?.extract()?;
        w_bytes(&mut buf, &bytes);
    }

    w_u32(&mut buf, exp.tasks.len() as u32);
    for task in &exp.tasks {
        w_u32(&mut buf, task.name_idx as u32);
        w_coord(&mut buf, &task.coord)?;
        w_i64(&mut buf, task.nbytes.max(0));
        match &task.compute {
            Compute::Call { func_idx } => {
                buf.push(0);
                w_u32(&mut buf, *func_idx as u32);
            }
            Compute::Alias => buf.push(2),
            Compute::CallKw { .. } => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "per-task kwargs (CallKw) not expressible in binary records",
                ));
            }
        }
        buf.push(u8_count(task.slots.len(), "task slots")?);
        for slot in &task.slots {
            w_slot(&mut buf, slot)?;
        }
    }

    Ok(PyBytes::new(py, &buf))
}

fn short_chunk() -> PyErr {
    PyNotImplementedError::new_err("short binary records chunk")
}

fn take<'a>(data: &'a [u8], pos: &mut usize, n: usize) -> PyResult<&'a [u8]> {
    let end = pos.checked_add(n).ok_or_else(short_chunk)?;
    if end > data.len() {
        return Err(short_chunk());
    }
    let out = &data[*pos..end];
    *pos = end;
    Ok(out)
}

fn r_u8(data: &[u8], pos: &mut usize) -> PyResult<u8> {
    Ok(take(data, pos, 1)?[0])
}

fn r_u32(data: &[u8], pos: &mut usize) -> PyResult<u32> {
    let bytes: [u8; 4] = take(data, pos, 4)?.try_into().map_err(|_| short_chunk())?;
    Ok(u32::from_le_bytes(bytes))
}

fn skip_i64(data: &[u8], pos: &mut usize) -> PyResult<()> {
    take(data, pos, 8)?;
    Ok(())
}

fn r_coord(data: &[u8], pos: &mut usize) -> PyResult<Vec<u32>> {
    let ndim = r_u8(data, pos)? as usize;
    let mut coord = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        coord.push(r_u32(data, pos)?);
    }
    Ok(coord)
}

fn r_string(data: &[u8], pos: &mut usize) -> PyResult<String> {
    let len = r_u32(data, pos)? as usize;
    let bytes = take(data, pos, len)?;
    std::str::from_utf8(bytes)
        .map(|s| s.to_owned())
        .map_err(|e| {
            PyNotImplementedError::new_err(format!("invalid utf8 in binary records chunk: {e}"))
        })
}

fn skip_opt_i64(data: &[u8], pos: &mut usize) -> PyResult<()> {
    if r_u8(data, pos)? != 0 {
        skip_i64(data, pos)?;
    }
    Ok(())
}

fn skip_slot(data: &[u8], pos: &mut usize) -> PyResult<()> {
    match r_u8(data, pos)? {
        0 => {
            // Dep { name_idx, coord }
            r_u32(data, pos)?;
            r_coord(data, pos)?;
        }
        1 => {
            // Index
            for _ in 0..r_u8(data, pos)? {
                match r_u8(data, pos)? {
                    0 => {
                        skip_opt_i64(data, pos)?;
                        skip_opt_i64(data, pos)?;
                        skip_opt_i64(data, pos)?;
                    }
                    1 => skip_i64(data, pos)?,
                    tag => {
                        return Err(PyNotImplementedError::new_err(format!(
                            "unknown index elem tag {tag}"
                        )))
                    }
                }
            }
        }
        2 => {
            // IntTuple
            let n = r_u8(data, pos)? as usize;
            take(data, pos, 8 * n)?;
        }
        3 => {
            // List
            for _ in 0..r_u32(data, pos)? {
                skip_slot(data, pos)?;
            }
        }
        4 => {
            // Scalar
            let num_tag = r_u8(data, pos)?;
            if num_tag != 0 && num_tag != 1 {
                return Err(PyNotImplementedError::new_err(format!(
                    "unknown scalar tag {num_tag}"
                )));
            }
            skip_i64(data, pos)?;
        }
        5 => {
            // Str
            let len = r_u32(data, pos)? as usize;
            take(data, pos, len)?;
        }
        tag => {
            return Err(PyNotImplementedError::new_err(format!(
                "unknown slot tag {tag}"
            )))
        }
    }
    Ok(())
}

fn expected_nbytes_for(
    task_name: &str,
    coord: &[u32],
    output_name: &str,
    chunks: &[Vec<i64>],
    itemsize: i64,
) -> i64 {
    if task_name != output_name || coord.len() != chunks.len() {
        return 0;
    }
    grid_nbytes(
        itemsize,
        coord.iter().enumerate().map(|(axis, &block)| {
            chunks
                .get(axis)
                .and_then(|dim| dim.get(block as usize))
                .copied()
                .unwrap_or(0)
        }),
    )
}

/// Patch the `expected_nbytes` field in a binary records LAYER chunk.
///
/// Layers stamp their non-grid helper tasks (splits, carries, halo slices…) at
/// expansion time; the plain output-grid tasks are left 0 and filled here from
/// the expression's `chunks`/`dtype`, which only Python's collector has. A
/// nonzero stamp is never overwritten. Runs the O(tasks) byte walk in Rust.
pub fn stamp_expected_nbytes_chunk(
    mut data: Vec<u8>,
    output_name: &str,
    chunks: &[Vec<i64>],
    itemsize: i64,
) -> PyResult<Vec<u8>> {
    let mut pos = 0;
    let version = r_u8(&data, &mut pos)?;
    if version != RECORDS_PROTOCOL_VERSION {
        return Ok(data);
    }

    let mut names = Vec::new();
    for _ in 0..r_u32(&data, &mut pos)? {
        names.push(r_string(&data, &mut pos)?);
    }
    for _ in 0..r_u32(&data, &mut pos)? {
        r_string(&data, &mut pos)?;
    }
    for _ in 0..r_u32(&data, &mut pos)? {
        let len = r_u32(&data, &mut pos)? as usize;
        take(&data, &mut pos, len)?;
    }

    for _ in 0..r_u32(&data, &mut pos)? {
        let name_idx = r_u32(&data, &mut pos)? as usize;
        let coord = r_coord(&data, &mut pos)?;
        let task_name = names
            .get(name_idx)
            .ok_or_else(|| PyNotImplementedError::new_err("task name index out of range"))?;

        let offset = pos;
        skip_i64(&data, &mut pos)?;
        // Fill only unknown (zero) stamps: a layer that stamped this task at
        // expansion time (helper geometry only it knows) wins over the generic
        // output-grid estimate.
        let existing = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        if existing == 0 {
            let nbytes = expected_nbytes_for(task_name, &coord, output_name, chunks, itemsize);
            if nbytes > 0 {
                data[offset..offset + 8].copy_from_slice(&nbytes.to_le_bytes());
            }
        }

        match r_u8(&data, &mut pos)? {
            0 => {
                r_u32(&data, &mut pos)?;
            }
            2 => {}
            tag => {
                return Err(PyNotImplementedError::new_err(format!(
                    "unknown compute tag {tag}"
                )))
            }
        }
        for _ in 0..r_u8(&data, &mut pos)? {
            skip_slot(&data, &mut pos)?;
        }
    }

    if pos != data.len() {
        return Err(PyNotImplementedError::new_err(
            "trailing bytes in binary records chunk",
        ));
    }
    Ok(data)
}
