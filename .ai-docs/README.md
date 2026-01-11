# AI Documentation Index

Documentation to help you understand this codebase before starting work.

## Available Documents

| Document | Read when... |
|----------|--------------|
| [expression-system.md](./expression-system.md) | Creating new operations or expressions. Understanding `ArrayExpr`, `Array`, `_parameters`, `_meta`, `chunks`, `_layer()`, `_name`. Debugging with `pprint()`, `__dask_graph__()`. Using `new_collection()`, `substitute_parameters()`. |
| [blockwise.md](./blockwise.md) | Working with `Blockwise`, `Elemwise`, element-wise ops, broadcasting. Understanding `out_ind`, `args`, `adjust_chunks`, `new_axes`, index mapping. Implementing `_task()`. Working on `FusedBlockwise` or fusion. |
| [reductions.md](./reductions.md) | Implementing or debugging reductions (`sum`, `mean`, `var`, `argmax`, etc.). Understanding tree reduction pattern with `chunk`/`combine`/`aggregate`. Working with `PartialReduce`, `split_every`, `keepdims`, `axis`. Adding new aggregation functions. |
| [slicing.md](./slicing.md) | Working with array indexing. Understanding `SliceSlicesIntegers`, `VIndexArray`, `BooleanIndexFlattened`. Handling unknown chunks (`np.nan`). Implementing `_accept_slice()`. Using `fuse_slice()`, `_slice_1d()`, `normalize_index()`. |
| [io.md](./io.md) | Reading data into arrays. Understanding `FromArray`, `_region` for deferred slicing, `inline_array` parameter. Adding new IO sources (zarr, h5py, custom). Working with `normalize_chunks()`, getter functions, locking. |
| [optimizations.md](./optimizations.md) | Adding or debugging optimizations. Implementing `_simplify_down()`, `_lower()`. Working on slice pushdown, rechunk pushdown, `FromArray`/IO optimization. Understanding fusion algorithm, `_is_blockwise_fusable`. Using `substitute_parameters()` in rewrites. |
| [testing.md](./testing.md) | Writing or fixing tests. Using `assert_eq` (not `np.testing.assert_array_equal`). Testing optimizations with `._name` comparison and `.simplify()`. Parametrization with `pytest.mark.parametrize`. Running tests with `uv run python -m pytest`. |

## Quick Reference

- **Expression system**: Lazy computation trees that optimize before generating task graphs
- **Blockwise**: Operations that map functions across aligned array blocks
- **Reductions**: Tree pattern with chunk/combine/aggregate for `sum`, `mean`, etc.
- **Slicing**: Multiple slice types for different indexing patterns
- **IO**: `FromArray` pattern with deferred slicing via `_region`
- **Optimizations**: `simplify()` → `lower()` → `fuse()` pipeline, pushdown patterns
- **Testing**: Use `assert_eq`, test structure with `._name` comparison, TDD approach
