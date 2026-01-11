# Dask Array Standalone Package

This is a re-implementation of dask arrays with a query optimization system, array expressions.

This provides large scale, chunked, parallel n-dimensional arrays that know how
to optimize themselves.

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

## Repository Structure

TODO: review and update

```
dask_array/
├── __init__.py           # Public API exports
├── _expr.py              # Base ArrayExpr class
├── _collection.py        # Array wrapper, method definitions
├── _blockwise.py         # Blockwise, Elemwise base classes
├── _chunk.py             # Chunk-level numpy wrappers
├── _slicing.py           # Getitem, setitem, slice expressions
├── _reshape.py           # Reshape operations
├── _concatenate.py       # Concatenation
├── reductions/           # Reduction operations
│   ├── _reduction.py     # reduction(), _tree_reduce()
│   ├── _common.py        # sum, mean, var, median, quantile, etc.
│   ├── _cumulative.py    # cumsum, cumprod
│   └── _percentile.py    # percentile, nanpercentile
├── manipulation/         # transpose, flip, roll, expand_dims
├── routines/             # diff, where, gradient
├── linalg/               # Matrix operations
├── core/                 # asarray, from_array
├── creation/             # ones, zeros, arange, etc.
├── io/                   # zarr, npy_stack
├── fft/                  # FFT operations
├── random/               # Random number generation
└── tests/                # Test suite
```

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
