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
| `_materialize` | `_materialize.py` | Expression → task graph (optimize + pin output keys) |
| `Blockwise` | `_blockwise.py` | Aligned block operations |
| `Elemwise` | `_blockwise.py` | Broadcasting element-wise ops |
| ...  | ... | various other ArrayExpr subclasses |

Operations live in domain subpackages (`stacking/`, `manipulation/`,
`routines/`, ...), one module per operation family holding the expression
class and the user function together. The placement rule — including which
modules may import `_collection` at module top — is
[.ai-docs/layout.md](.ai-docs/layout.md).


## Building and releasing

See [docs/development.md](docs/development.md) for building (pure-Python by
default; the Rust accelerator is opt-in via `maturin develop`), running the test
suites (dask and Frisky), and cutting a release.

## Testing

[docs/development.md](docs/development.md) is the canonical home for test
invocation (including the Frisky variant). The standard run:

```bash
uv run --extra test --extra complete --extra sparse python -m pytest dask_array/tests/ -q
```

For specific test files:
```bash
uv run --extra test --extra complete --extra sparse python -m pytest dask_array/tests/test_reductions.py -q
```

Keep the extras consistent between runs — flapping them makes `uv` rebuild the
environment each time.

### Testing Pattern

We mostly depend on existing tests that we're pulling from the legacy dask.array
project.  They often look like this:

```python
import numpy as np
import dask_array as da
from dask_array._test_utils import assert_eq

def test_sum():
    x = np.random.random((10, 10))
    d = da.from_array(x, chunks=(5, 5))
    assert_eq(d.sum(), x.sum())
```

Import `assert_eq` from `dask_array._test_utils`, not `dask.array.utils` — the
suite deliberately avoids importing legacy `dask.array`, which registers global
tokenize/dispatch handlers as a side effect.

Notably:

-  We use pytest
-  We use flat functions (not grouped into classes)
-  We compare behavior against numpy
-  We use `assert_eq`, which tests various things about the Dask object structure, aside from value equality
-  Tests are very fast when possible (milliseconds)

## Frisky task-graph support

`dask_array` can hand its graphs to **Frisky** (a sibling Rust `dask.distributed`
reimplementation) as task *records* generated in Rust — for supported layers a
single binary blob per layer, which is where the vast majority of tasks land —
instead of materializing millions of Python `Task` objects on the client. It's
transparent and opt-in: with a Frisky scheduler active, `dask.compute(x)` /
`dask.persist(x)` take the records path; anything unsupported falls back to
stock dask. Design records:
`plans/frisky-rust-task-gen.md` (architecture) and
`plans/frisky-binary-records-gaps.md` (the binary-chunks push).

**Structures.** One *layer* per expression kind — a Rust object in
`crates/dask-array-python/src/` with a thin wrapper in `dask_array/_frisky/`
(import the wrapper's submodule directly) — builds a neutral form (`common.rs`)
with three converters, in descending preference:

- `to_records_chunk()` → ONE binary `bytes` blob for the whole layer (the
  grammar Frisky's `records_proto` decodes) — O(1) Python objects per layer,
  the fast path;
- `to_task_records()` → flat `(key, func, args, kwargs, deps)` Python tuples,
  O(tasks), for layers whose Rust backend declines the binary chunk;
- `to_dask_graph()` → `{key: dask Task}`, for focused parity checks only.

`collect_record_chunks` (`_frisky/collect.py`) walks the lowered tree; each node
contributes a binary chunk (plus per-layer group metadata for Frisky's display,
and any side records the layer needs — e.g. from_array's single source-array
holder) or, failing that, plain records. A node without a native layer — or
whose layer raises `NotImplementedError` — is translated by the generic
`GraphRecordsLayer` (`_frisky/graph_records.py`), which reuses the expr's own
legacy `_layer()`, faithful by construction. `dask_array/_frisky/inventory.py`
and its standalone sibling `bench/tier_probe.py` report the task-weighted tier
split (binary / native_tuples / adapter / fallback).

**Protocols.**

- Routing: ordinary `_layer()` implementations stay on the Python Dask graph path.
  Frisky collection walks call `_frisky_layer()` directly; `_frisky_layer()` raises
  for any variant it doesn't fully handle.
- Hand-off (duck-typed on `Array` in `_collection.py` — Frisky never imports
  `dask_array`): `__frisky_records_chunks__()` returns
  `(chunks, records, chunk_groups)` — binary layer chunks, residual plain
  records, and per-chunk layer metadata; Frisky decodes both task sources and
  unions them under one `dask.order` pass. `__frisky_graph__()` is the
  tuples-only variant, and `__frisky_output_keys__()` gives the flat stringified
  output keys.
- Records: `key`/`deps` are strings, `func`/`kwargs` shared Python objects, `args` a
  tuple whose dependency slots hold a `dask._task_spec.TaskRef(dep_key)`.
- Versioning: `NATIVE_BUILD_GENERATION`, synced between `_frisky/base.py` and the
  Rust `lib.rs`, makes a stale local `.so` fail loudly at import — a LOCAL
  build-freshness check, not a wire protocol (Frisky never reads it). The
  Frisky-facing records grammar is versioned separately by
  `common::RECORDS_PROTOCOL_VERSION` (`common.rs`).
- Unknown (`nan`) chunk *sizes* are fine — records are keyed by block coordinate
  and numblocks is known even when sizes are not; only masked arrays decline the
  whole graph (`_check_frisky_supported`). And completeness is enforced — every
  dep must be produced by some record (`collect_task_records` checks this
  client-side on the tuples path; on the chunks path Frisky checks the combined
  graph) — with any violation falling back to stock dask: an unfaithful
  translation degrades rather than miscomputes.

**Constraints and testing.**

- Frisky stays array-agnostic — ordinary records only, no cross-crate dependency.
- Falling back is always correct (adapter/stock dask), just not the fast path.
- The records path is pytest-covered: `dask_array/tests/test_frisky_protocol.py`
  decodes binary chunks and pins which layers go binary, and CI runs the full
  suite under `--scheduler=frisky` (the `test-frisky-rust` job in
  `.github/workflows/ci.yml`). The `bench/` differentials (`diff_records.py`,
  `diff_layers.py`) remain for ad-hoc deep checks.
- The build is pure-Python by default; the native accelerator is opt-in via
  `maturin develop` (the records path falls back to the pure-Python
  `GraphRecordsLayer` when `_rust` is absent). Bump `NATIVE_BUILD_GENERATION` on
  both sides after any Rust change. See [docs/development.md](docs/development.md).
