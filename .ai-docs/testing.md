# Testing

This guide covers testing patterns, TDD practices, and the critical `assert_eq` utility.

## Philosophy

1. **Test-Driven Development**: Write tests before implementation
2. **Fast tests**: Each test should complete in milliseconds
3. **Compare against NumPy**: Dask arrays should match NumPy behavior
4. **Structural testing**: Verify optimization structure, not just values

## Running Tests

```bash
# Run all tests
uv run python -m pytest dask_array/tests/ -q

# Run specific test file
uv run python -m pytest dask_array/tests/test_reductions.py -q

# Run in parallel (faster)
uv run python -m pytest dask_array/tests/ -n auto

# Run specific test
uv run python -m pytest dask_array/tests/test_collection.py::test_from_array -q
```

## Test Style

### Flat Functions (No Classes)

```python
import pytest
import numpy as np
import dask_array as da
from dask_array._test_utils import assert_eq

def test_from_array():
    x = np.random.random((10, 10))
    d = da.from_array(x, chunks=(5, 5))
    assert_eq(d, x)
    assert d.chunks == ((5, 5), (5, 5))
```

### Parametrization

```python
@pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__"])
def test_arithmetic_ops(arr, op):
    result = getattr(arr, op)(2)
    expected = getattr(arr.compute(), op)(2)
    assert_eq(result, expected)

@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
@pytest.mark.parametrize("func", ["sum", "mean", "max"])
def test_reductions(arr, func, axis):
    result = getattr(arr, func)(axis=axis)
    expected = getattr(arr.compute(), func)(axis=axis)
    assert_eq(result, expected)
```

## The `assert_eq` Utility

**Always use `assert_eq` instead of `==` for array comparisons.**

Located in `_test_utils.py`, it validates:

| Check | What it verifies |
|-------|------------------|
| `check_shape` | Shapes match |
| `check_dtype` | Data types match |
| `check_type` | Python types match (scalars stay scalars) |
| `check_meta` | `_meta` consistency before/after compute |
| `check_chunks` | Actual chunk shapes match expected |
| `check_ndim` | Dimensions preserved through compute |
| Value equality | `allclose()` with NaN handling |

### Basic Usage

```python
# Compare dask array to numpy array
assert_eq(dask_array, numpy_array)

# Compare two dask arrays
assert_eq(da.sum(x), da.sum(y))

# With options
assert_eq(result, expected, check_dtype=False)
```

### Why It Matters

`assert_eq` catches subtle bugs that `np.testing.assert_array_equal` misses:

```python
# This only checks values:
np.testing.assert_array_equal(result.compute(), expected)

# This catches more issues:
assert_eq(result, expected)  # Also checks dtype, shape, chunks, meta...
```

## Testing Optimizations

**Prefer structural equality over value-only tests.**

### Core Pattern

```python
def test_slice_through_elemwise():
    """(x + y)[:5] should optimize to x[:5] + y[:5]"""
    x = da.ones((100, 100), chunks=(10, 10))
    y = da.ones((100, 100), chunks=(10, 10))

    result = (x + y)[:5]
    expected = x[:5] + y[:5]

    # 1. Structural equality - verifies optimization happened
    assert result.expr.simplify()._name == expected.expr.simplify()._name

    # 2. Value correctness - ensures no breakage
    assert_eq(result, expected)
```

### Why `._name` Comparison?

Names are **deterministic** based on expression structure. Different expression trees cannot produce the same name.

```python
# These have different ._name values:
(x + y)[:5].expr._name  # "slice-abc123"
x[:5] + y[:5].expr._name  # Different structure = different name
```

### Why `.simplify()` Both Sides?

The expected expression often contains optimizable patterns:

```python
# x[:5] itself might simplify (e.g., slice into from_array)
assert result.expr.simplify()._name == expected.expr.simplify()._name
```

Skip `.simplify()` on expected only when it's already a base expression:

```python
x = da.from_array(arr, chunks=(5, 5))
assert x.T.T.expr.simplify()._name == x.expr._name  # x.expr is already simple
```

### Task Count Verification

```python
def test_optimization_reduces_tasks():
    x = da.from_array(np.ones((100, 100)), chunks=(10, 10))

    full_tasks = len(x.sum(axis=0).optimize().__dask_graph__())
    sliced_tasks = len(x.sum(axis=0)[:5].optimize().__dask_graph__())

    assert sliced_tasks < full_tasks
```

### Type Checking

```python
def test_rechunk_pushes_through_elemwise():
    x = da.ones((10, 10), chunks=(2, 2))
    result = (x + 1).rechunk((5, 5))
    opt = result.expr.optimize()

    # Verify rechunk pushed through - result should be Elemwise, not Rechunk
    assert type(opt).__name__ == "Elemwise"
    assert_eq(result, x.compute() + 1)
```

## Test File Organization

```
dask_array/tests/
├── test_collection.py          # Core array ops & optimization
├── test_reductions.py          # sum, mean, var, etc.
├── test_creation.py            # zeros, ones, arange
├── test_slicing.py             # Basic slicing
├── test_slice_pushdown.py      # Slice optimization tests
├── test_rechunk_pushdown.py    # Rechunk optimization tests
├── test_routines.py            # Utility functions
└── ...
```

**Naming convention**: `test_<feature>[_<subfeature>].py`

## Test Markers

```python
@pytest.mark.slow
def test_large_reduction():
    # Large/slow test
    ...

@pytest.mark.xfail(reason="Known issue")
def test_known_failure():
    ...
```

## TDD Workflow

### 1. Write Test First

```python
def test_slice_through_new_operation():
    """x.new_op()[:5] should optimize to x[:5].new_op()"""
    x = da.ones((100, 100), chunks=(10, 10))

    result = x.new_op()[:5]
    expected = x[:5].new_op()

    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, expected)
```

### 2. Run Test (Expect Failure)

```bash
uv run python -m pytest dask_array/tests/test_new.py::test_slice_through_new_operation -v
```

### 3. Implement Feature

Add `_accept_slice()` method to the new operation class.

### 4. Run Test (Expect Pass)

```bash
uv run python -m pytest dask_array/tests/test_new.py::test_slice_through_new_operation -v
```

### 5. Add Edge Cases

```python
@pytest.mark.parametrize("slc", [
    slice(5),
    slice(5, 10),
    slice(None, None, 2),
    (slice(5), slice(10)),
])
def test_slice_through_new_operation_variants(slc):
    x = da.ones((100, 100), chunks=(10, 10))
    result = x.new_op()[slc]
    expected = x[slc].new_op()
    assert result.expr.simplify()._name == expected.expr.simplify()._name
```

## Common Patterns

### Compare Dask vs NumPy

```python
def test_operation():
    x = np.random.random((10, 10))
    d = da.from_array(x, chunks=(5, 5))
    assert_eq(d.operation(), x.operation())
```

### Verify Optimization Structure

```python
def test_optimization_applied():
    result = naive_expression
    expected = optimized_form
    assert result.expr.optimize()._name == expected.expr.optimize()._name
```

### Check Deterministic Naming

```python
def test_deterministic_names():
    x = da.ones((10, 10), chunks=(5, 5))
    assert x.sum()._name == x.sum()._name  # Same operation = same name
```

## Key Files

| File | Purpose |
|------|---------|
| `_test_utils.py` | `assert_eq` and other utilities |
| `tests/test_collection.py` | Core tests and optimization examples |
| `tests/test_slice_pushdown.py` | Slice optimization tests |

## Anti-patterns

- **Don't** use `np.testing.assert_array_equal` for dask array comparisons
- **Don't** test only values without structural checks for optimization tests
- **Don't** write slow tests (keep them fast, milliseconds each)
- **Don't** test implementation details that might change
