# Plan: Reduce dask.array Imports

Goal: Make dask_array more standalone by reducing imports from dask.array modules.

## Current Import Summary (198 total)

| Module | Count | Priority | Action |
|--------|-------|----------|--------|
| `dask.array.utils` | 72 | Medium | Copy needed utilities |
| `dask.array.core` | 45 | High | Copy needed functions |
| `dask.array.numpy_compat` | 17 | Low | Keep or copy shims |
| `dask.array.slicing` | 9 | Medium | Copy slicing utils |
| `dask.array.chunk` | 9 | Low | Already have `_chunk.py` |
| `dask.array.creation` | 5 | Medium | Review what's needed |
| `dask.array.backends` | 5 | Low | Keep as dependency |
| `dask.array.dispatch` | 4 | Low | Keep as dependency |
| `dask.array.reductions` | 3 | High | Clean up remaining |
| `dask.array.rechunk` | 3 | Medium | Copy if needed |
| `dask.array.reshape` | 2 | Medium | Copy if needed |
| Other | 14 | Low | Review case by case |

## Task Breakdown

### Task 1: Clean up remaining dask.array.reductions imports (High Priority)
- Find and remove the 3 remaining imports
- These should use local `dask_array.reductions` instead

Files to check:
```
grep -r "from dask.array.reductions" dask_array/
```

### Task 2: Copy dask.array.utils functions (Medium Priority)
Functions commonly imported:
- `meta_from_array`
- `array_safe`, `asarray_safe`, `asanyarray_safe`
- `compute_meta`
- `validate_axis`
- `is_arraylike`
- `svd_flip`
- `solve_triangular_safe`

Create: `dask_array/_utils.py`

### Task 3: Copy dask.array.core helpers (High Priority)
Functions commonly imported:
- `normalize_chunks`
- `is_scalar_for_elemwise`
- `getter_inline`
- `concatenate3`
- `tensordot_lookup`
- `_pass_extra_kwargs`
- `apply_and_enforce`
- `apply_infer_dtype`

Create: `dask_array/_core_utils.py`

### Task 4: Copy dask.array.numpy_compat (Low Priority)
Shims for numpy version compatibility:
- `NUMPY_GE_200`
- `normalize_axis_tuple`
- `normalize_axis_index`
- `_Recurser`

Since we target numpy >= 2.0, many of these may be unnecessary.

Create: `dask_array/_numpy_compat.py` (simplified)

### Task 5: Copy dask.array.slicing utilities (Medium Priority)
- `sanitize_index`
- Other slicing helpers

Review what's actually needed vs what's already in our slicing code.

### Task 6: Review dask.array.chunk usage (Low Priority)
We already have `dask_array/_chunk.py`. Check if:
- All needed functions are there
- Imports can be switched to local version

### Task 7: IO modules (Low Priority)
`io/_zarr.py` and `io/_store.py` delegate heavily to dask.array.
Options:
1. Keep as thin wrappers (current approach)
2. Copy full implementation

Recommendation: Keep as-is for now, these are optional features.

## Execution Strategy

Run tasks in parallel where possible:
- Task 1 (reductions cleanup) - independent
- Task 2 + 3 (utils + core) - can run together
- Task 4 (numpy_compat) - independent
- Task 5 (slicing) - after Task 2/3
- Task 6 (chunk) - independent
- Task 7 (IO) - defer

## Success Criteria

1. No imports from `dask.array.reductions`
2. Reduced imports from `dask.array.core` and `dask.array.utils`
3. All existing tests still pass
4. Package can function with minimal dask.array dependency

## Detailed Import Analysis

### dask.array.utils (by frequency)
- `meta_from_array` (25+ uses) - critical
- `assert_eq`, `same_keys` (13 uses) - test utilities, keep importing
- `validate_axis` (7 uses)
- `array_safe`, `asarray_safe`, `asanyarray_safe` (8 uses)
- `is_arraylike` (3 uses)
- `solve_triangular_safe` (2 uses) - linalg specific
- `svd_flip` (2 uses) - linalg specific
- `compute_meta` (2 uses)
- `arange_safe` (1 use)
- `is_cupy_type` (1 use)
- `allclose` (1 use) - test utility

### dask.array.core (by frequency)
- `normalize_chunks` (12 uses) - critical
- `unknown_chunk_message` (6 uses) - error message string
- `_concatenate2` (2 uses) - already using in reductions
- `concatenate3` (2 uses)
- `is_scalar_for_elemwise` (3 uses)
- `broadcast_shapes`, `broadcast_chunks` (3 uses)
- `apply_infer_dtype` (3 uses)
- `handle_out` (1 use) - already using in reductions
- `getter_inline` (1 use)
- `tensordot_lookup` (1 use)
- `_pass_extra_kwargs`, `apply_and_enforce` (1 use)
- IO functions: `from_zarr`, `to_zarr`, `to_hdf5`, `load_chunk` - keep as deps

## Notes

- Some imports (dispatch, backends) are fine to keep - they're infrastructure
- Focus on removing imports that bring in heavy dependencies or circular refs
- Test after each batch of changes
- Test utilities (`assert_eq`, `same_keys`) should stay as imports from dask
