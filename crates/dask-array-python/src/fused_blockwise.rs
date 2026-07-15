//! Binary-record layer for Python-validated fused blockwise tasks.
//!
//! Python builds and validates the block-independent `_FusedSubgraph` callable.
//! Rust only expands the output grid into ordinary call tasks whose arguments
//! are the per-output source block dependencies Python already resolved from
//! each fused `_execute_subgraph` task.
//!
//! Most fused layers pass only source-block dependency refs. A layer that bakes
//! its own `block_id` into the subgraph (map_overlap's `_trim`, stencils,
//! `block_info`) is made block-independent on the Python side by lifting that
//! literal into a per-block *seed*: an ordinary value (a nested int tuple such as
//! `(block_id, numblocks)`) that `_execute_subgraph` feeds into a holed subgraph
//! slot. Those seeds are appended to each task's args AFTER the dependency refs —
//! matching the `inkeys` order Python assigns (sources first, then seeds). A
//! layer with no lifted literals has no seeds, so its records are byte-identical
//! to before.
//!
//! Two ways to receive the per-block wiring:
//!
//!   * [`FusedBlockwiseLayer::new`] takes the fully materialized `dep_slots` /
//!     `seed_slots` (one list per output block). Python's exact fallback specs
//!     (`_fast_spec_uniform` / `_site_based_spec`), which read each block's real
//!     coords, use this — no closed form exists for the shapes they cover.
//!
//!   * [`FusedBlockwiseLayer::from_projections`] takes the *compact* form: a
//!     per-site coordinate PROJECTION (each output-block dim maps to a source-block
//!     dim or a constant) plus per-seed int TEMPLATES. Rust then generates every
//!     block's dep coords and seed values by integer arithmetic in `expand`,
//!     instead of Python materializing an O(blocks) list and shipping it over.
//!     This is the common case (`_analytical_site_spec` / `_seed_spec`, i.e.
//!     elemwise/broadcast/transpose/contractions and overlap/rolling/stencils) —
//!     the arithmetic mirrors Python's `ordered_slots` / `_apply_template`, so the
//!     records are byte-identical to the materialized path.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask, Num};

/// One output-block dimension of a source-coordinate projection: either a fixed
/// block index (`Const`) or "track output block dim `o`" (`Bid`). Mirrors
/// Python's `("const", c)` / `("bid", o)` projection leaves.
enum Coord {
    Const(u32),
    Bid(usize),
}

/// A source SITE's projection: which dependency, and how each of its block dims
/// is derived from the output block id. Evaluated per output block into an
/// [`ArgSlot::Dep`].
struct Projection {
    dep_idx: usize,
    coords: Vec<Coord>,
}

/// An int leaf of a seed template: a constant, an output-block-dim reference
/// (`bid[o]`, a block id), or an output-block CHUNK SIZE (`chunks[a][bid[a]]`, the
/// per-axis size of this output block — e.g. an inner `Ones`/`full` shape whose
/// boundary blocks differ from the interior, as `diff` produces).
enum IntExpr {
    Const(i64),
    Bid(usize),
    Chunk(usize),
}

/// A compiled lifted-literal seed template. Its *shape* is fixed across blocks
/// (only the int leaves vary with the block id), so the tuple-vs-list collapsing
/// that `seed_to_slot` applies to a materialized value is resolved once, here, at
/// construction — a sequence of all int leaves becomes [`Seed::IntTuple`]
/// (rendering a real tuple), anything nested becomes [`Seed::List`]. Evaluating a
/// `Seed` for a block reproduces `seed_to_slot(_apply_template(tmpl, bid))` byte
/// for byte.
enum Seed {
    Scalar(IntExpr),
    IntTuple(Vec<IntExpr>),
    List(Vec<Seed>),
}

/// Where a layer's per-block argument slots come from.
enum SlotSource {
    /// Fully materialized per-block lists — one entry per output block. Dep slots
    /// are `(dep_idx, coord)`; seed slots are owned int-shaped `ArgSlot`s appended
    /// after the deps. Empty `seed_slots` for the common (no block_id) case.
    Materialized {
        dep_slots: Vec<Vec<(usize, Vec<u32>)>>,
        seed_slots: Vec<Vec<ArgSlot>>,
    },
    /// Compact closed form — a projection per source site, and a template per
    /// lifted seed. Rust generates every block's slots from these. `chunk_tables`
    /// are the per-axis output chunk sizes, indexed by `Chunk` seed leaves.
    Projected {
        projections: Vec<Projection>,
        seeds: Vec<Seed>,
        chunk_tables: Vec<Vec<i64>>,
    },
}

#[pyclass]
pub struct FusedBlockwiseLayer {
    name: String,
    func: Py<PyAny>,
    empty_kwargs: Py<PyAny>,
    dep_names: Vec<String>,
    slots: SlotSource,
    numblocks: Vec<usize>,
}

fn eval_int(expr: &IntExpr, bid: &[u32], chunk_tables: &[Vec<i64>]) -> i64 {
    match expr {
        IntExpr::Const(v) => *v,
        IntExpr::Bid(o) => bid[*o] as i64,
        IntExpr::Chunk(a) => chunk_tables[*a][bid[*a] as usize],
    }
}

fn eval_seed(seed: &Seed, bid: &[u32], chunk_tables: &[Vec<i64>]) -> ArgSlot {
    match seed {
        Seed::Scalar(e) => ArgSlot::Scalar(Num::Int(eval_int(e, bid, chunk_tables))),
        Seed::IntTuple(es) => {
            ArgSlot::IntTuple(es.iter().map(|e| eval_int(e, bid, chunk_tables)).collect())
        }
        Seed::List(items) => {
            ArgSlot::List(items.iter().map(|s| eval_seed(s, bid, chunk_tables)).collect())
        }
    }
}

/// Reject a seed template whose leaf references an axis out of range, or a
/// `Chunk` leaf whose axis has no (or too short a) chunk table — a mis-inferred
/// template would index out of bounds at expansion time.
fn check_int_expr(expr: &IntExpr, ndim: usize, chunk_tables: &[Vec<i64>], nb: &[usize]) -> PyResult<()> {
    let bad = pyo3::exceptions::PyNotImplementedError::new_err("seed template axis out of range");
    match expr {
        IntExpr::Const(_) => Ok(()),
        IntExpr::Bid(o) => {
            if *o >= ndim {
                Err(bad)
            } else {
                Ok(())
            }
        }
        IntExpr::Chunk(a) => {
            // `bid[a]` ranges over `0..nb[a]`, so the chunk table must cover it.
            if *a >= ndim || *a >= chunk_tables.len() || chunk_tables[*a].len() < nb[*a] {
                Err(bad)
            } else {
                Ok(())
            }
        }
    }
}

fn check_seed_dims(seed: &Seed, ndim: usize, chunk_tables: &[Vec<i64>], nb: &[usize]) -> PyResult<()> {
    match seed {
        Seed::Scalar(e) => check_int_expr(e, ndim, chunk_tables, nb),
        Seed::IntTuple(es) => {
            for e in es {
                check_int_expr(e, ndim, chunk_tables, nb)?;
            }
            Ok(())
        }
        Seed::List(items) => {
            for it in items {
                check_seed_dims(it, ndim, chunk_tables, nb)?;
            }
            Ok(())
        }
    }
}

/// True if `obj` is a leaf template node `("const", v)` / `("bid", o)` — as
/// opposed to a container node `("tuple"|"list", [children])`. Determines whether
/// a container collapses to an `IntTuple` (all-leaf children) or a `List`,
/// matching `seed_to_slot`'s all-int check.
fn seed_tmpl_is_leaf(obj: &Bound<'_, PyAny>) -> bool {
    if let Ok(t) = obj.cast::<PyTuple>() {
        if let Ok(tag) = t.get_item(0).and_then(|x| x.extract::<String>()) {
            return tag == "const" || tag == "bid" || tag == "chunk";
        }
    }
    false
}

/// Parse a leaf template node into an [`IntExpr`].
fn parse_int_expr(obj: &Bound<'_, PyAny>) -> PyResult<IntExpr> {
    let t = obj.cast::<PyTuple>()?;
    let tag: String = t.get_item(0)?.extract()?;
    match tag.as_str() {
        "const" => Ok(IntExpr::Const(t.get_item(1)?.extract()?)),
        "bid" => Ok(IntExpr::Bid(t.get_item(1)?.extract()?)),
        "chunk" => Ok(IntExpr::Chunk(t.get_item(1)?.extract()?)),
        _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "seed template leaf must be ('const'|'bid'|'chunk', int)",
        )),
    }
}

/// Compile a Python seed template — `("const", v)`, `("bid", o)`, `("chunk", a)`,
/// or a container `("tuple"|"list", [children])` — into a [`Seed`], resolving the
/// tuple/list collapsing exactly as `seed_to_slot` would after `_apply_template`.
fn parse_seed(obj: &Bound<'_, PyAny>) -> PyResult<Seed> {
    let t = obj.cast::<PyTuple>().map_err(|_| {
        pyo3::exceptions::PyNotImplementedError::new_err("seed template node must be a 2-tuple")
    })?;
    let tag: String = t.get_item(0)?.extract()?;
    match tag.as_str() {
        "const" | "bid" | "chunk" => Ok(Seed::Scalar(parse_int_expr(obj)?)),
        "tuple" | "list" => {
            let kids: Vec<Bound<'_, PyAny>> =
                t.get_item(1)?.try_iter()?.collect::<PyResult<_>>()?;
            if kids.iter().all(seed_tmpl_is_leaf) {
                let exprs = kids
                    .iter()
                    .map(parse_int_expr)
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(Seed::IntTuple(exprs))
            } else {
                let seeds = kids.iter().map(parse_seed).collect::<PyResult<Vec<_>>>()?;
                Ok(Seed::List(seeds))
            }
        }
        _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "seed template node must be const/bid/chunk/tuple/list",
        )),
    }
}

/// Convert a per-block seed value (a nested structure of ints, e.g.
/// `(block_id, numblocks)`) into an owned `ArgSlot`. A plain int becomes a
/// `Scalar`, a flat int sequence an `IntTuple` (rendering as a real tuple), and a
/// nested sequence a `List` of the same. Rejects anything that isn't int-shaped
/// so a mis-inferred seed declines to the tuples path rather than emitting a
/// wrong graph.
///
/// Fidelity note: only the innermost flat int runs stay tuples; a *nested*
/// sequence's outer container renders as a Python list, not a tuple (the grammar
/// has no tuple-of-slots element). Harmless for the consumers that reach the seed
/// path — `_trim` and the stencil shapes only index/iterate the seed — but a
/// hypothetical consumer doing `isinstance(seed, tuple)` on an outer container
/// would see the difference. The Python side (`_seed_spec`) only lifts such
/// literals, so it owns that contract.
fn seed_to_slot(obj: &Bound<'_, PyAny>) -> PyResult<ArgSlot> {
    // bool is an int subclass in Python; seeds are coordinates, never bools.
    if obj.is_instance_of::<pyo3::types::PyBool>() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "seed value must be an int or nested int sequence, got bool",
        ));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(ArgSlot::Scalar(Num::Int(i)));
    }
    let items: Vec<Bound<'_, PyAny>> = if let Ok(t) = obj.cast::<PyTuple>() {
        t.iter().collect()
    } else if let Ok(l) = obj.cast::<PyList>() {
        l.iter().collect()
    } else {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "seed value must be an int or nested int sequence",
        ));
    };
    // A flat int tuple renders as a real tuple; a nested one as a List of the
    // same (a List indexes/unpacks like a tuple, which is all the consumers do).
    let all_int = items
        .iter()
        .all(|it| !it.is_instance_of::<pyo3::types::PyBool>() && it.extract::<i64>().is_ok());
    if all_int {
        let ints: Vec<i64> = items
            .iter()
            .map(|it| it.extract::<i64>().unwrap())
            .collect();
        return Ok(ArgSlot::IntTuple(ints));
    }
    let mut nested: Vec<ArgSlot> = Vec::with_capacity(items.len());
    for it in &items {
        nested.push(seed_to_slot(it)?);
    }
    Ok(ArgSlot::List(nested))
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
        seed_slots: Vec<Vec<Py<PyAny>>>,
    ) -> PyResult<Self> {
        let expected: usize = if numblocks.is_empty() {
            1
        } else {
            numblocks.iter().product()
        };
        // `seed_slots` is either empty (the common no-block_id case: no seeds for
        // any block) or one list per output block.
        if dep_slots.len() != expected || (!seed_slots.is_empty() && seed_slots.len() != expected) {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "dep/seed slots length does not match output blocks",
            ));
        }
        for slots in &dep_slots {
            for (dep_idx, _) in slots {
                if *dep_idx >= dep_names.len() {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                        "dep slot index out of range",
                    ));
                }
            }
        }
        // Pre-convert each block's seed values into owned int-shaped ArgSlots up
        // front, so expansion is pure Rust and a non-int seed is rejected here.
        let seed_slots: Vec<Vec<ArgSlot>> = seed_slots
            .into_iter()
            .map(|block| {
                block
                    .into_iter()
                    .map(|obj| seed_to_slot(obj.bind(py)))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            name,
            func,
            empty_kwargs: PyDict::new(py).unbind().into_any(),
            dep_names,
            slots: SlotSource::Materialized {
                dep_slots,
                seed_slots,
            },
            numblocks,
        })
    }

    /// Compact constructor: instead of a materialized per-block slot list, take a
    /// coordinate PROJECTION per source site and an int TEMPLATE per lifted seed,
    /// and generate every block's slots in Rust (see module docs). Used by
    /// Python's `_analytical_site_spec` / `_seed_spec` — the common shapes — so
    /// the O(blocks) per-block materialization never happens in Python.
    ///
    /// `projections`: per site `(dep_idx, [(kind, val)])`, kind `0` = constant
    /// block index `val`, kind `1` = track output block dim `val`. `seed_templates`:
    /// the Python template objects (`("const", v)` / `("bid", o)` / `("chunk", a)`
    /// / a container `("tuple"|"list", [children])`), compiled once here.
    /// `chunk_tables`: per-axis output chunk sizes, indexed by `("chunk", a)` seed
    /// leaves (`chunks[a][bid[a]]`).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_projections(
        py: Python<'_>,
        name: String,
        func: Py<PyAny>,
        numblocks: Vec<usize>,
        dep_names: Vec<String>,
        projections: Vec<(usize, Vec<(u8, u32)>)>,
        seed_templates: Vec<Py<PyAny>>,
        chunk_tables: Vec<Vec<i64>>,
    ) -> PyResult<Self> {
        // Every validation failure here raises `NotImplementedError` (not
        // `ValueError`): a malformed spec should DECLINE to the Python
        // fallback path (`to_task_records` -> `_slow_records`), which the walk
        // reaches only by catching `NotImplementedError` — never crash the whole
        // graph collection. These guards can't fire for a spec the Python side
        // actually produces (indices/dims derive from `range(len(nb))` and the
        // dep-name map, validated against probes); they are defensive only.
        let ndim = numblocks.len();
        let mut projs: Vec<Projection> = Vec::with_capacity(projections.len());
        for (dep_idx, coords) in projections {
            if dep_idx >= dep_names.len() {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "projection dep index out of range",
                ));
            }
            let mut cs: Vec<Coord> = Vec::with_capacity(coords.len());
            for (kind, val) in coords {
                match kind {
                    0 => cs.push(Coord::Const(val)),
                    1 => {
                        if (val as usize) >= ndim {
                            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                                "projection block-id dim out of range",
                            ));
                        }
                        cs.push(Coord::Bid(val as usize));
                    }
                    _ => {
                        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                            "projection coord kind must be 0 (const) or 1 (bid)",
                        ));
                    }
                }
            }
            projs.push(Projection {
                dep_idx,
                coords: cs,
            });
        }

        let seeds = seed_templates
            .into_iter()
            .map(|obj| parse_seed(obj.bind(py)))
            .collect::<PyResult<Vec<_>>>()?;
        // A block-id/chunk reference in a seed must land inside the output grid
        // and (for chunk leaves) have a chunk table covering that axis.
        for seed in &seeds {
            check_seed_dims(seed, ndim, &chunk_tables, &numblocks)?;
        }

        Ok(Self {
            name,
            func,
            empty_kwargs: PyDict::new(py).unbind().into_any(),
            dep_names,
            slots: SlotSource::Projected {
                projections: projs,
                seeds,
                chunk_tables,
            },
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
            // Source dependency refs first, then any lifted-literal seeds — the
            // args order Python assigned when building `inkeys`.
            let slots: Vec<ArgSlot> = match &self.slots {
                SlotSource::Materialized {
                    dep_slots,
                    seed_slots,
                } => {
                    let mut s: Vec<ArgSlot> = dep_slots[task_idx]
                        .iter()
                        .map(|(dep_idx, dep_coord)| ArgSlot::Dep {
                            name_idx: *dep_idx,
                            coord: dep_coord.clone(),
                        })
                        .collect();
                    if let Some(seeds) = seed_slots.get(task_idx) {
                        s.extend(seeds.iter().cloned());
                    }
                    s
                }
                SlotSource::Projected {
                    projections,
                    seeds,
                    chunk_tables,
                } => {
                    let mut s: Vec<ArgSlot> = projections
                        .iter()
                        .map(|p| ArgSlot::Dep {
                            name_idx: p.dep_idx,
                            coord: p
                                .coords
                                .iter()
                                .map(|c| match c {
                                    Coord::Const(v) => *v,
                                    Coord::Bid(o) => coord[*o],
                                })
                                .collect(),
                        })
                        .collect();
                    for seed in seeds {
                        s.push(eval_seed(seed, &coord, chunk_tables));
                    }
                    s
                }
            };
            tasks.push(NeutralTask {
                nbytes: 0,
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
