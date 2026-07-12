# Smooth dask_array + xarray Integration

Status: external-dependency design record (last updated 2026-06-04). Phases 1-2
live on branches in the sibling dask and xarray checkouts, not in released
packages; this repo's side (phase 3, the chunk manager) is done. Kept as the
cross-repo design reference until the upstream protocol work lands or is
abandoned.

## Summary

Use Dask as the owner of a generic composite-expression protocol, xarray as an
ordered child-expression provider, and dask_array as the owner of xarray's chunk
manager.

Use a private Dask `CompositeExpr` to preserve that many child expressions
belong to one original collection. A bare `_ExprSequence` represents many
top-level results, not one grouped collection result.

## Key Changes

- Dask adds ordered protocol methods:
  - `__dask_exprs__() -> Sequence[Expr] | None`
  - `__dask_rebuild_from_exprs__(exprs: Sequence[Expr]) -> Any`
- Dask adds private `CompositeExpr` and uses it from `collections_to_expr` after
  `.expr` handling and before `HLGExpr` fallback.
- xarray implements the ordered protocol on `Dataset` and `DataArray`, without
  adding `Dataset.expr`, `DataArray.expr`, `DatasetExpr`, or `DataArrayExpr`.
- dask_array keeps owning `dask_array._xarray:DaskArrayExprManager`.

## Implementation Phases

1. Done in `/Users/mrocklin/workspace/dask` branch
   `codex/composite-expr-protocol`: Dask `CompositeExpr`, protocol detection,
   local `persist`/`optimize` rebuilds, and fake composite-collection tests.
2. Done in `/Users/mrocklin/workspace/xarray-composite-expr` branch
   `codex/composite-expr-protocol`: xarray `Dataset` and `DataArray`
   implement ordered child expression extraction and rebuild through
   `dask._collections.new_collection`.
3. Done in this repo: dask_array keeps the existing chunk manager and has
   regression coverage for chunk-manager uniqueness and Dask
   `new_collection(expr)` round trips.
4. In progress: broadened workflow coverage now includes open_mfdataset,
   shared subexpressions, chunked non-index coordinates, rechunk/reduction
   chains, groupby, map_blocks fallback, and apply_ufunc. Remaining work is to
   decide whether xarray map_blocks should produce dask_array expressions
   directly instead of legacy dask.array output.

## Test Plan

- Dask:
  - `collections_to_expr` returns `CompositeExpr` for protocol collections.
  - Composite compute returns one rebuilt collection.
  - Composite persist returns one lazy rebuilt collection with persisted child
    arrays.
  - Composite optimize returns rebuilt collections sharing optimized child
    graphs.
  - HLG fallback remains unchanged.
- xarray:
  - `collections_to_expr(ds)` is composite when all chunked children expose
    `.expr`.
  - `Dataset.compute`, `DataArray.compute`, `Dataset.persist`, and
    `DataArray.persist` preserve xarray structure and values.
  - Mixed legacy/expression data falls back safely.
- dask_array:
  - `uv run pytest dask_array/tests/test_xarray.py -q`
  - Focused tests for chunk-manager uniqueness and `new_collection` round trips.

## Current Verification

- Dask focused tests:
  `PYTHONPATH=/Users/mrocklin/workspace/dask uv run pytest -c /dev/null ...`
  for the six new composite protocol tests; passed.
- xarray focused tests:
  `PYTHONPATH=/Users/mrocklin/workspace/dask:/Users/mrocklin/workspace/xarray-composite-expr:/Users/mrocklin/workspace/dask-array uv run pytest -o addopts= -W ignore::pytest.PytestUnknownMarkWarning /Users/mrocklin/workspace/xarray-composite-expr/xarray/tests/test_dask_expr_protocol.py -q`;
  13 tests passed.
- dask_array focused tests:
  `PYTHONPATH=/Users/mrocklin/workspace/dask:/Users/mrocklin/workspace/xarray-composite-expr:/Users/mrocklin/workspace/dask-array uv run pytest dask_array/tests/test_xarray.py -q`;
  passed.

## Assumptions

- Ordered child protocol is sufficient for v1.
- `CompositeExpr` is private Dask infrastructure.
- dask_array remains the xarray chunk-manager owner.
