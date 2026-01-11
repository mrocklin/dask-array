# Reductions

Reductions aggregate array values along axes using a parallel tree reduction pattern. This is fundamentally different from [blockwise](./blockwise.md) operations.

## Tree Reduction Pattern

Reductions use a three-phase hierarchical approach:

```
Input chunks:     [c0] [c1] [c2] [c3] [c4] [c5] [c6] [c7]
                    \   /     \   /     \   /     \   /
Chunk phase:       p0       p1        p2        p3
                    \       /          \        /
Combine phase:        x0                  x1
                       \                  /
Aggregate phase:              final
```

**Three functions define a reduction:**
- `chunk`: Applied to each input chunk independently
- `combine`: Merges intermediate results (optional, defaults to `aggregate`)
- `aggregate`: Produces final result

## Core API (`reductions/_reduction.py`)

```python
def reduction(
    x,                    # Input array
    chunk,               # Function applied to each chunk
    aggregate,           # Final aggregation function
    axis=None,           # Axes to reduce (None = all)
    keepdims=False,      # Preserve reduced dimensions
    dtype=None,          # Output dtype
    split_every=None,    # Tree width (default: 16)
    combine=None,        # Intermediate aggregation (optional)
    concatenate=True,    # Auto-concat intermediate results
    output_size=1,       # Output size per reduced axis
    weights=None,        # Optional weights
)
```

## Simple Reductions

For operations where chunk/combine/aggregate are identical:

```python
# reductions/_common.py
def sum(a, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
    if dtype is None:
        dtype = getattr(np.zeros(1, dtype=a.dtype).sum(), "dtype", object)
    return reduction(
        a,
        chunk=chunk.sum,      # np.sum on each chunk
        aggregate=chunk.sum,  # np.sum on aggregates
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        out=out,
    )
```

**Examples:** `sum`, `prod`, `min`, `max`, `any`, `all`

## NaN-Aware Reductions

Handle NaN values in the chunk phase, then use standard aggregation:

```python
def nansum(a, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
    return reduction(
        a,
        chunk=chunk.nansum,    # np.nansum ignores NaNs
        aggregate=chunk.sum,   # np.sum (no NaNs in intermediate results)
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
    )
```

**Examples:** `nansum`, `nanprod`, `nanmin`, `nanmax`, `nanmean`

## Complex Reductions: Mean

Mean requires tracking both sum and count:

```python
def mean_chunk(x, dtype="f8", **kwargs):
    n = numel(x, dtype=dtype, **kwargs)      # Count elements
    total = chunk.sum(x, dtype=dtype, **kwargs)  # Sum values
    return {"n": n, "total": total}  # Return dict, not array!

def mean_combine(pairs, dtype="f8", axis=None, **kwargs):
    # Combine multiple {n, total} dicts
    ns = deepmap(lambda pair: pair["n"], pairs)
    n = _concatenate2(ns, axes=axis).sum(axis=axis, **kwargs)
    totals = deepmap(lambda pair: pair["total"], pairs)
    total = _concatenate2(totals, axes=axis).sum(axis=axis, **kwargs)
    return {"n": n, "total": total}

def mean_agg(pairs, dtype="f8", axis=None, **kwargs):
    # Final division
    n = ...  # Combined count
    total = ...  # Combined sum
    return divide(total, n, dtype=dtype)

def mean(a, axis=None, dtype=None, keepdims=False, split_every=None):
    return reduction(
        a,
        chunk=mean_chunk,
        aggregate=mean_agg,
        combine=mean_combine,
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        concatenate=False,  # Don't auto-concat dicts!
    )
```

**Key point:** When `concatenate=False`, intermediate results are passed as nested lists/dicts rather than concatenated arrays.

## Complex Reductions: Variance

Variance uses Welford's online algorithm to combine partial variances:

```python
def moment_chunk(A, order=2, dtype="f8", **kwargs):
    n = numel(A, **kwargs)
    total = chunk.sum(A, dtype=dtype, **kwargs)
    u = total / n  # Mean
    d = A - u      # Deviations
    # Compute power sums for variance
    xs = [chunk.sum(d**i, dtype=dtype, **kwargs) for i in range(2, order + 1)]
    M = np.stack(xs, axis=-1)
    return {"total": total, "n": n, "M": M}

def moment_combine(pairs, order=2, dtype="f8", axis=None, **kwargs):
    # Merge partial variances using parallel algorithm
    # Uses binomial expansion to combine power sums
    ...
```

## Argument Reductions (argmin, argmax)

Return indices rather than values. Requires tracking global offsets:

```python
def arg_chunk(func, argfunc, x, axis, offset_info):
    vals = func(x, axis=axis, keepdims=True)
    arg = argfunc(x, axis=axis, keepdims=True)

    # Convert local indices to global indices
    offset, total_shape = offset_info
    ind = np.unravel_index(arg.ravel()[0], x.shape)
    total_ind = tuple(o + i for o, i in zip(offset, ind))
    arg[:] = np.ravel_multi_index(total_ind, total_shape)

    # Return structured array with both values and indices
    result = np.empty_like(vals, dtype=[("vals", vals.dtype), ("arg", arg.dtype)])
    result["vals"] = vals
    result["arg"] = arg
    return result
```

## Cumulative Reductions

Sequential pattern for `cumsum`, `cumprod`:

```python
class CumReduction(ArrayExpr):
    def _layer(self):
        # Phase 1: Apply cumulative function to each chunk
        for key in product(*map(range, x.numblocks)):
            dsk[(name + "-chunk",) + key] = (cumfunc, (x.name,) + key)

        # Phase 2: Add previous blocks' totals
        for i in range(1, n_blocks_on_axis):
            # extra[i] = binop(extra[i-1], last_element[i-1])
            # result[i] = binop(extra[i], chunk_result[i])
```

## The `split_every` Parameter

Controls tree width and depth:

| split_every | Tree depth | Memory/combine | Use case |
|-------------|------------|----------------|----------|
| 2 | log₂(n) | Very low | Memory-constrained |
| 8 | log₈(n) | Medium | Balanced |
| 16 (default) | log₁₆(n) | Higher | Performance |
| n+ | 1 | Huge | Single machine |

**Per-axis control:**
```python
# Different tree widths for different axes
x.sum(axis=(0, 1), split_every={0: 4, 1: 8})
```

## PartialReduce Expression

The expression class representing one reduction step:

```python
class PartialReduce(ArrayExpr):
    _parameters = [
        "array",          # Input expression
        "func",           # Reduction function
        "split_every",    # Dict mapping axis → split_every value
        "keepdims",
        "dtype",
        "name",
        "reduced_meta",
    ]
```

## Implementing a New Reduction

### Simple Case (idempotent)

```python
def my_reduction(a, axis=None, keepdims=False, split_every=None):
    return reduction(
        a,
        chunk=np.my_func,
        aggregate=np.my_func,
        axis=axis,
        keepdims=keepdims,
        dtype=a.dtype,
        split_every=split_every,
    )
```

### Complex Case (tracking state)

```python
def my_complex_reduction(a, axis=None, keepdims=False, split_every=None):
    def chunk_func(x, axis=None, keepdims=False):
        # Return dict with intermediate state
        return {"value": ..., "count": ..., "extra": ...}

    def combine_func(pairs, axis=None, keepdims=True):
        # Merge intermediate states
        return {"value": ..., "count": ..., "extra": ...}

    def agg_func(pairs, axis=None, keepdims=False):
        # Produce final result from merged state
        return final_value

    return reduction(
        a,
        chunk=chunk_func,
        aggregate=agg_func,
        combine=combine_func,
        axis=axis,
        keepdims=keepdims,
        dtype=output_dtype,
        split_every=split_every,
        concatenate=False,  # Important for dict returns
    )
```

## Axis and Keepdims Handling

```python
# Axis normalization
if axis is None:
    axis = tuple(range(x.ndim))
if isinstance(axis, int):
    axis = (axis,)
axis = validate_axis(axis, x.ndim)  # Handle negative indices

# During reduction, always keepdims=True internally
# Final keepdims applied in aggregate phase
```

## Optimization: Slice Pushdown

`PartialReduce._accept_slice()` enables:

```python
# Before optimization
x.sum(axis=0)[:5]

# After optimization (fewer elements processed)
x[:, :5].sum(axis=0)
```

## Key Files

| File | Purpose |
|------|---------|
| `reductions/_reduction.py` | Core `reduction()` function, `PartialReduce` class |
| `reductions/_common.py` | `sum`, `mean`, `var`, `std`, `min`, `max`, nan-variants |
| `reductions/_cumulative.py` | `cumsum`, `cumprod` |
| `reductions/_arg_reduction.py` | `argmin`, `argmax` |
| `reductions/_percentile.py` | `percentile`, `quantile`, `median` |
| `_chunk.py` | Chunk-level numpy wrappers |
