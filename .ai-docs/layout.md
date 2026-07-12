# Package Layout — Where Things Live

One grain: **an operation family is one module in the subpackage that owns its
domain, holding the expression class(es) and the user-facing function(s)
together.** `stacking/_stack.py` holds `Stack` and `stack()`;
`io/_from_map.py` holds `FromMap` and `from_map()`; `manipulation/_squeeze.py`
holds `Squeeze` and `squeeze()`. Operation N+1 takes this shape.

## Subpackages (operations)

| Subpackage | Owns |
|------------|------|
| `creation/` | arange, linspace, ones/zeros/full, eye, tri, diag(onal), pad, tile, repeat, meshgrid/indices |
| `io/` | from_array/zarr/tiledb/npy_stack/delayed/map/graph, store, to_* |
| `linalg/` | tensordot/matmul/dot, qr, svd, lu, cholesky, solve, norm |
| `manipulation/` | transpose/swapaxes/moveaxis, flip/rot90, roll, reshape/ravel, expand_dims/atleast_*, squeeze |
| `random/` | Generator / RandomState APIs |
| `reductions/` | sum/mean/... tree reductions, arg reductions, cumulative, percentile, trace |
| `routines/` | the long tail of NumPy routines: where, unique, diff, outer, bincount, searchsorted, ... |
| `slicing/` | `x[...]` / `x[...] = v` machinery: basic slices, bool/int-array indexing, vindex, `.blocks` |
| `stacking/` | stack, concatenate, block, vstack/hstack/dstack |
| `fft.py` | the `da.fft.*` namespace (a module, not a package) |

`core/` is not an op domain: it holds the entry points everything builds on —
`from_array`/`asarray`/`array` (`_conversion.py`), `blockwise`/`elemwise`
(`_blockwise_funcs.py`) — and re-exports `Array` and `from_graph`.

Same-basename rule: a module's basename plus its package must identify its
contents. Don't create a second `_foo.py` whose contents could be confused
with an existing one (`_broadcast_to.py` = `BroadcastTo` expression +
`broadcast_to`; `routines/_broadcast.py` = `broadcast_arrays`/`unify_chunks`).
When a function and its expression drift into different packages, merge them
into the op module instead (`from_graph` lives with `FromGraph` in
`io/_from_graph.py`; `dask_array.core.from_graph` is a re-export).

## Top level (machinery)

Top-level `_*.py` modules are shared machinery, not op homes:

- Expression system: `_expr.py` (`ArrayExpr`, `RootAlias`, `ChunksFreeze`),
  `_blockwise.py` (`Blockwise`/`Elemwise`/`FusedBlockwise` + fusion),
  `_materialize.py` (`_lower`/`_materialize`/`_LOWER_CACHE` — expression →
  task graph, the single choke point), `_rechunk.py`
- Collection layer: `_collection.py` (`Array` wrapper + methods),
  `_new_collection.py`
- Chunk-level kernels and helpers: `_chunk.py`, `_core_utils.py`, `_utils.py`,
  `_dispatch.py`, `_chunk_types.py`, `_numpy_compat.py`, `_backends*.py`
- Frameworks that generate operations rather than being one: `_map_blocks.py`,
  `_overlap.py`, `_gufunc.py`, `_ufunc.py` (the elemwise operator table),
  `_einsum.py`, `_shuffle.py`, `_histogram.py`, and `_broadcast_to.py`
  (broadcasting is core semantics; `_collection` needs it at module scope).
  Treat this set as closed — new operations go in a subpackage.
- Display/diagnostics: `_svg.py`, `_visualize.py`, `_expr_flow.py`,
  `_diagnostics.py`, `_templates.py`
- Interop: `_xarray.py`/`xarray.py`, `_frisky/` (one Rust-backed layer wrapper
  per expression kind; see AGENTS.md)

## Import rules — who may import `_collection` at module top

`dask_array/__init__.py` establishes load order, and `_collection.py`'s own
module body is the constraint: everything `_collection` imports at module
scope (transitively) executes *while `_collection` is still half-built*, so
those modules must defer any `_collection` import into function bodies.

- **Must defer** (they are in `_collection`'s module-scope import closure):
  the subpackages `core/`, `io/`, `manipulation/`, `slicing/`, `stacking/`,
  and the machinery modules `_expr.py`, `_materialize.py`, `_blockwise.py`,
  `_broadcast_to.py`, `_rechunk.py`, `_chunk.py`, `_utils.py`,
  `_core_utils.py`, `_dispatch.py`, `_chunk_types.py`, `_new_collection.py`,
  `_templates.py`. The standard pattern is a function-local
  `from dask_array._collection import asarray` inside the user-facing
  function (see `stacking/_stack.py`).
- **May import at top**: everything outside that closure — `creation/`,
  `linalg/`, `random/`, `reductions/`, `routines/`, `fft.py`, `_ufunc.py`,
  `_gufunc.py`, `_map_blocks.py`, `_overlap.py`, `_histogram.py`,
  `_einsum.py`, `_shuffle.py`. Most of these load after `_collection`
  completes; `fft.py` instead loads just before it and is what triggers it
  (see `routines/_topk.py` for the plain top-level import style).
- Expression modules never need `_collection` at module scope anyway: graph
  building goes through `_materialize.py` (which imports only `_expr`), and
  wrapping results uses `_new_collection.new_collection`. `_expr.py` imports
  `_materialize` at call time inside `__dask_graph__`/`_layer` (the reverse
  edge is module-scoped, so this one stays deferred).

When in doubt, measure: wrap `builtins.__import__`, snapshot
`sorted(m for m in sys.modules if m.startswith("dask_array"))` the moment
`dask_array._collection` finishes executing, and import the package —
anything in the snapshot (that isn't merely mid-load above `_collection` on
the import stack, like `fft.py`, which triggers it) is in the closure and
must defer.

## Adding operation N+1

1. Pick the subpackage whose table row matches; create
   `<subpackage>/_<op>.py` with the expression class and the user function
   together (template in [expression-system.md](./expression-system.md)).
2. Export the function from the subpackage `__init__.py`, then from the root
   `dask_array/__init__.py` (which imports subpackage names directly, e.g.
   `from dask_array.routines import ...`) and its `__all__`.
3. If NumPy has a method form, add a thin delegating method on `Array`
   (`_collection.py`) with a function-local import — method bodies are the
   one place `_collection` reaches back into op modules.
4. Optional Frisky fast path: add `_frisky_layer()` on the expression and a
   layer in `_frisky/` + `crates/dask-array-python/src/` (see AGENTS.md);
   without it the generic fallback is used, which is correct, just slower.
