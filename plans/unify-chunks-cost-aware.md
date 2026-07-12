# Cost-aware chunk unification

Status: done 2026-07-05, kept as design record. Shipped as the `auto`
unify-chunks policy: `_MERGE_COST_RATIO` + `moved_fraction` cost model in
`dask_array/_expr.py` (`unify_chunks_expr`), config documented at the top of
`dask_array/__init__.py`, behavior summarized in `.ai-docs/optimizations.md`
("Rechunk Insertion"). The "Why" below is the design rationale; the opening
description of the old fixed-direction rule is historical.

When elemwise operands disagree on chunking, `unify_chunks_expr`
(`dask_array/_expr.py`) picks a common layout and rechunks the operands to it.
Today's rule (`coarse_blockdim`) merges nested chunkings up to the coarsest
operand, with a size guard: above `array.unify-chunks-limit` (default 512 MiB)
it falls back to refinement and warns.  `array.unify-chunks-policy: refine`
opts into stock-dask refinement wholesale.

We want to replace the fixed direction with a cost-aware choice.

## Why

`bench/bench_unify_policy.py` measures the two fixed directions (numbers in
its docstring, from a real bug writeup):

- On the **inflation shape** (per-day-chunked 2D × nested 2-chunk 1D), merging
  is a pure loss: 40% slower wall, ~8× the estimated transfer, and it once
  inflated 46 MB chunks into 3 GB chunks across a whole downstream DAG — the
  incident that motivated the guard.
- On the **shatter shape** (coarse 3D − per-element-indexed partner, the
  xarray groupby pattern) and on the **macro synthetic DAG**, refinement is
  the disaster: 4× and 5× slower, ~5× and ~9× the tasks.

The asymmetry has a physical root: refining an operand only *splits* chunks
(slicing, nearly free), while merging an operand *moves* its bytes
(concatenation).  So the cheap direction depends on which operand is heavier —
a 6 MB time vector should follow a 4 GB array's layout, never the reverse, and
a per-element-chunked 100 kB indexer should follow everyone else.  A rule of
the shape "rechunk the lighter operands toward the heavier operand's layout,
subject to the byte cap" picks the winner in all three bench cases by
construction.  The exact cost function, tie-breaking, and how task count
enters (frisky makes tasks cheap, but 9× is still real — see the macro case)
are yours to work out; `ArrayExpr.transfer_bytes` is an existing per-node
(min, max) communication estimate that may serve as the cost model.

## Where to work

- `dask_array/_expr.py`: `coarse_blockdim` (per-dim, no byte context) and
  `unify_chunks_expr` (sees the operand expressions — shapes, dtypes, chunks —
  so per-operand byte math belongs here; the size guard is a worked example).
- `bench/bench_unify_policy.py` is the harness: a new policy should win or tie
  **all three cases**.  Add cases if your rule has other interesting regimes
  (e.g. two heavy operands with misaligned-but-comparable layouts).
- `dask_array/tests/test_binary_op_chunks.py` holds the behavior tests.  Mind
  the NOTE there: unification runs lazily at first `.chunks` access and
  expressions are singletons, so config must wrap the access and tests need
  shapes no other test builds.

## Constraints

- Full suite green (`uv run python -m pytest dask_array/tests/ -q`); watch the
  existing coarse-preference tests — if your rule changes their expectations,
  that may be correct, but say so explicitly rather than silently updating.
- Keep `array.unify-chunks-limit` as a hard backstop regardless of policy, and
  keep the `refine` policy escape hatch working.
- Chunks must stay deterministic per expression identity (singleton caching);
  config read at construction is the established pattern.
- A downstream trading-research pipeline (the statarb repo) validates against
  this code with a full-scale lazy chunk audit and cluster runs; its healthy
  behavior today includes ~180–330 MB merges from rolling-window halos that
  should keep merging.  That side will A/B after your change lands — you don't
  need to reproduce it, just don't assume the bench is the whole world.
