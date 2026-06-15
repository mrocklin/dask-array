# Dask Array Standalone Package

This is a re-implementation of dask arrays with a query optimization system, array expressions.

This provides large scale, chunked, parallel n-dimensional arrays that know how to optimize themselves.

## Drop-in Replacement for dask.array

Normal Dask arrays can be used as follows:

```python
import dask.array as da
```

We're exactly the same

```python
import dask_array as da
```

Except that now computations will rearrange themselves into more
computationally efficient forms.

## Relationship with Original Dask Repository

This repository still depends on the original Dask repository for task
schedulers, task specifications, and so on, but it doesn't borrow from the
legacy dask.array package, or any of the HighLevelGraph machinery.  It also
borrows the original Expression (Expr) system originally designed for Dask
DataFrames.

## Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `ArrayExpr` | `_expr.py` | Base class for all expressions |
| `Array` | `_collection.py` | User-facing wrapper, provides API and mutation|
| `Blockwise` | `_blockwise.py` | Aligned block operations |
| `Elemwise` | `_blockwise.py` | Broadcasting element-wise ops |
| ...  | ... | various other ArrayExpr subclasses |


## Testing

Run tests with:
```bash
uv run python -m pytest dask_array/tests/ -q
```

For specific test files:
```bash
uv run python -m pytest dask_array/tests/test_reductions.py -q
```

### Testing Pattern

We mostly depend on existing tests that we're pulling from the legacy dask.array
project.  They often look like this:

```python
import numpy as np
import dask_array as da
from dask.array.utils import assert_eq

def test_sum():
    x = np.random.random((10, 10))
    d = da.from_array(x, chunks=(5, 5))
    assert_eq(d.sum(), x.sum())
```

Notably:

-  We use pytest
-  We use flat functions (not grouped into classes)
-  We compare behavior against numpy
-  We use `assert_eq`, which tests various things about the Dask object structure, aside from value equality
-  Tests are very fast when possible (milliseconds)

## Frisky task-graph support

`dask_array` can hand its graphs to **Frisky** (a sibling Rust `dask.distributed`
reimplementation) as flat task *records* generated in Rust, instead of
materializing millions of Python `Task` objects on the client. It's transparent
and opt-in: with a Frisky scheduler active, `dask.compute(x)` / `dask.persist(x)`
take the records path; anything unsupported falls back to stock dask. Full design
and status: `plans/frisky-rust-task-gen.md`.

**Structures.** One *layer* per expression kind — a Rust object in
`crates/dask-array-python/src/` with a thin wrapper in `dask_array/_frisky/` —
builds a neutral form (`common.rs`) expanded two ways: `to_dask_graph()` →
`{key: dask Task}` (the legacy/correctness path the pytest suite validates) and
`to_task_records()` → flat `(key, func, args, kwargs, deps)` records (the fast path
Frisky submits). `collect_task_records` (`_frisky/collect.py`) walks the lowered
tree; any node without a native layer (or whose layer defers) is translated by the
generic `GraphRecordsLayer` (`_frisky/graph_records.py`), which reuses the expr's
own legacy `_layer()` — faithful by construction.

**Protocols.**

- Routing: `_layer()` is `try: self._frisky_layer().to_dask_graph() except
  (NotImplementedError, ImportError): <legacy graph>`; `_frisky_layer()` raises for
  any variant it doesn't fully handle.
- Records: `key`/`deps` are strings, `func`/`kwargs` shared Python objects, `args` a
  tuple whose dependency slots hold a `dask._task_spec.TaskRef(dep_key)`.
- Hand-off: `Array.__frisky_task_records__()` (duck-typed — Frisky never imports
  `dask_array`) returns them; `Client.submit_tasks(records, output_keys)` submits.
  The surface is versioned by `PROTOCOL_REVISION`, synced between `_frisky/base.py`
  and Frisky's `lib.rs` (mismatch fails loudly at import).
- `collect_task_records` checks every dep is produced by some record, else raises →
  whole-graph fallback — an unfaithful translation degrades rather than miscomputes.

**Constraints.**

- Frisky stays array-agnostic — ordinary records only, no cross-crate dependency.
- Falling back is always correct (legacy/adapter), just not the native fast path.
- The pytest suite covers only the `_layer`/`to_dask_graph` path; the records path
  is validated by `bench/diff_records.py` (vs `scheduler="synchronous"`).
- Build with `maturin develop` (`--release` for perf) and
  `MATURIN_IMPORT_HOOK_ENABLED=0`; bump `PROTOCOL_REVISION` on both sides when the
  record surface changes.
