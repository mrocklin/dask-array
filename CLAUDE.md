# Dask Array Standalone Package

This is a standalone extraction of dask array expressions from the main dask repository.
This project will replace the dask.array project in the future.

## Initial Project Goal

Extract `dask/array/_array_expr/` into a standalone package that:
- Imports as `import dask_array as da`
- Depends on dask for core infrastructure (expressions, task spec, schedulers)
- Has its own implementations of utilities (not importing from `dask.array`)
- Targets modern numpy (>= 2.0)

## Architecture

Array expressions separate **intent** from **execution**:

```python
# User code creates expression tree
result = (x + y).sum(axis=0)

# Expression tree: Sum(Elemwise(+, x.expr, y.expr), axis=0)
# Graph generated only at compute time
```

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `ArrayExpr` | `_expr.py` | Base class for all expressions |
| `Array` | `_collection.py` | User-facing wrapper |
| `Blockwise` | `_blockwise.py` | Aligned block operations |
| `Elemwise` | `_blockwise.py` | Broadcasting element-wise ops |

### Key Dependencies on dask (OK to keep)

- `dask._expr` - Expression base classes
- `dask._task_spec` - Task specification
- `dask.base` - `DaskMethodsMixin`, `compute`, `persist`
- `dask.threaded`, `dask.multiprocessing` - Schedulers

### Dependencies to Remove/Copy

We're working to remove imports from:
- `dask.array.utils` - Copy needed utilities locally
- `dask.array.core` - Copy needed helper functions
- `dask.array.reductions` - Already mostly done
- `dask.array.slicing` - Copy needed utilities
- `dask.array.numpy_compat` - Simplify for numpy >= 2.0
- `dask.highlevelgraph` - Graph representation
- `dask.array.dispatch` - Type dispatch infrastructure
- `dask.array.backends` - Backend system

## Repository Structure

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

We mostly depend on existing tests that we're pulling from the dask.array
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

Sometimes we also have tests for optimizations (this is new to array
expressions).

## Current Status

### Working
- Expression system and optimizations (slice pushdown, fusion, etc.)
- Creation functions (ones, zeros, arange, from_array, etc.)
- Reductions (sum, mean, var, std, min, max, argmin, argmax, median, quantile, etc.)
- Slicing and indexing
- Basic linear algebra (dot, matmul, tensordot)
- Random number generation
- FFT operations

### In Progress
- Removing `dask.array` imports (see `plans/reduce-dask-array-imports.md`)

## Key Principles

1. **Expression-based**: All operations build expression trees, graphs generated at compute time
2. **Standalone**: Minimize imports from `dask.array.*` modules
3. **Modern**: Target numpy >= 2.0, Python >= 3.10
4. **Simple**: Avoid unnecessary abstraction, keep implementations clear

## Plans

See `plans/` directory for implementation plans:
- `reduce-dask-array-imports.md` - Plan to reduce dask.array dependencies

## Anti-patterns

- **Don't** import from `dask.array.foo` - use `dask_array.foo`
- **Don't** import from `dask.array.core` for utilities - copy them locally
- **Don't** create dask.array.Array objects - use dask_array.Array
- **Don't** put function implementations in `_collection.py` - keep in separate modules
