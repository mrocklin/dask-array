"""Tests for slice pushdown into IO expressions."""

from __future__ import annotations

import numpy as np
import pytest

import dask_array as da
from dask_array.io import FromArray
from dask_array.slicing import SliceSlicesIntegers
from dask_array._test_utils import assert_eq

# Parametrized correctness tests: (array_shape, chunks, slice_tuple)
SLICE_CASES = [
    # Basic slices
    ((10, 10), (2, 2), (slice(0, 2), slice(0, 2))),  # corner
    ((10, 10), (2, 2), (slice(0, 4), slice(0, 4))),  # 2x2 chunks
    ((10, 10), (5, 5), (slice(0, 5), slice(0, 5))),  # chunk boundary
    ((10, 10), (5, 5), (slice(2, 7), slice(3, 8))),  # mid-chunk
    ((10, 10), (5, 5), (slice(None), slice(None))),  # full slice
    # Edge cases
    ((10, 10), (2, 2), (slice(5, 5), slice(None))),  # empty
    ((10, 10), (2, 2), (slice(3, 4), slice(None))),  # single row
    ((10, 10), (2, 2), (slice(-4, -1), slice(-3, None))),  # negative
    ((10, 10, 10), (3, 3, 3), (slice(1, 4), slice(2, 5), slice(3, 6))),  # 3D
    # Adversarial
    ((10, 10), (10, 10), (slice(2, 5), slice(3, 7))),  # single chunk source
    ((10, 10), (3, 4), (slice(1, 8), slice(2, 9))),  # uneven chunks
    ((10, 10), (3, 3), (slice(9, None), slice(9, None))),  # last chunk
    ((10, 10, 10), (3, 3, 3), (slice(2, 5),)),  # partial dims
]


@pytest.mark.parametrize("shape,chunks,slc", SLICE_CASES)
def test_slice_correctness(shape, chunks, slc):
    """Sliced dask array matches sliced numpy array."""
    arr = np.arange(np.prod(shape)).reshape(shape)
    x = da.from_array(arr, chunks=chunks)
    assert_eq(x[slc], arr[slc])


# Task count tests: (array_shape, chunks, slice_tuple, expected_tasks)
TASK_COUNT_CASES = [
    ((10, 10), (2, 2), (slice(0, 2), slice(0, 2)), 1),  # 1 chunk
    ((10, 10), (2, 2), (slice(0, 4), slice(0, 4)), 4),  # 2x2 chunks
    ((10, 10), (5, 5), (slice(0, 5), slice(0, 5)), 1),  # boundary
    ((10, 10), (5, 5), (slice(2, 7), slice(3, 8)), 4),  # spans 2x2
    ((20, 20), (5, 5), (slice(0, 3), slice(None)), 4),  # 1x4 row
    ((20, 20), (5, 5), (slice(None), slice(0, 3)), 4),  # 4x1 col
    ((20, 20), (5, 5), (slice(0, 12), slice(0, 12)), 9),  # 3x3
    ((100, 100), (10, 10), (slice(0, 5), slice(0, 5)), 1),  # small from large
    ((10, 10, 10), (5, 5, 5), (slice(0, 3), slice(0, 3), slice(0, 3)), 1),  # 3D corner
]


@pytest.mark.parametrize("shape,chunks,slc,expected", TASK_COUNT_CASES)
def test_task_count(shape, chunks, slc, expected):
    """After optimization, task count equals chunks touched."""
    arr = np.arange(np.prod(shape)).reshape(shape)
    x = da.from_array(arr, chunks=chunks)
    y = x[slc].optimize()
    assert len(y.__dask_graph__()) == expected


def test_slice_optimize_slice():
    """Slice, optimize, slice again works correctly."""
    arr = np.arange(100).reshape(10, 10)
    x = da.from_array(arr, chunks=(2, 2))

    y = x[0:6, 0:6].optimize()
    assert len(y.__dask_graph__()) == 9  # 3x3 chunks

    z = y[0:2, 0:2].optimize()
    assert len(z.__dask_graph__()) == 1  # 1 chunk

    assert_eq(z, arr[0:6, 0:6][0:2, 0:2])


def test_slice_through_elemwise():
    """Slice pushes through elemwise into IO."""
    arr = np.arange(100).reshape(10, 10)
    x = da.from_array(arr, chunks=(2, 2))
    y = ((x + 1) * 2)[0:2, 0:2].optimize()
    assert len(y.__dask_graph__()) <= 2
    assert_eq(y, ((arr + 1) * 2)[0:2, 0:2])


def test_nested_slices():
    """Nested slices fuse."""
    arr = np.arange(100).reshape(10, 10)
    x = da.from_array(arr, chunks=(2, 2))
    y = x[1:8, 2:9][1:4, 1:4]
    assert_eq(y, arr[1:8, 2:9][1:4, 1:4])


def test_expression_structure():
    """Verify expression types before/after optimization."""
    x = da.from_array(np.arange(100).reshape(10, 10), chunks=(2, 2))
    y = x[0:2, 0:2]

    assert isinstance(y.expr, SliceSlicesIntegers)
    assert isinstance(y.optimize().expr, FromArray)


def test_steps_and_reverse():
    """Slices with steps still compute correctly."""
    arr = np.arange(100).reshape(10, 10)
    x = da.from_array(arr, chunks=(2, 2))

    assert_eq(x[::2, ::2], arr[::2, ::2])
    assert_eq(x[::-1, ::-1], arr[::-1, ::-1])
    assert_eq(x[::5, ::5], arr[::5, ::5])


def test_non_pushdown_cases():
    """Integer indexing, fancy indexing, newaxis don't break."""
    arr = np.arange(100).reshape(10, 10)
    x = da.from_array(arr, chunks=(2, 2))

    assert_eq(x[5, :], arr[5, :])
    assert_eq(x[[1, 3, 5], :], arr[[1, 3, 5], :])
    assert_eq(x[None, :5, :5], arr[None, :5, :5])


def test_broadcast_to_empty_slice():
    result = da.broadcast_to(da.from_array(np.array([1]), (1,)), (5,))[:0]
    expected = np.array([], dtype=int)

    assert result.chunks == ((0,),)
    assert_eq(result, expected)
    assert_eq(da.Array(result.expr.optimize(fuse=False)), expected)


def test_masked_array():
    """Slice pushdown preserves masks."""
    arr = np.ma.array(np.arange(100).reshape(10, 10), mask=False)
    arr.mask[5, 5] = True
    x = da.from_array(arr, chunks=(3, 3))
    expected = arr[4:7, 4:7].copy()

    arr[5, 5] = 999
    arr.mask[4, 4] = True

    result = x[4:7, 4:7].compute()
    assert_eq(result, expected)
    assert_eq(result.mask, expected.mask)


def test_deterministic_names():
    """Same slice -> same name, different slice -> different name."""
    arr = np.arange(100).reshape(10, 10)
    x1 = da.from_array(arr, chunks=(2, 2))
    x2 = da.from_array(arr, chunks=(2, 2))

    assert x1[0:2, 0:2].optimize().name == x2[0:2, 0:2].optimize().name
    assert x1[0:2, 0:2].optimize().name != x1[0:3, 0:3].optimize().name


def test_slice_then_reduction():
    """Slice followed by reduction."""
    arr = np.arange(100).reshape(10, 10)
    x = da.from_array(arr, chunks=(2, 2))
    assert_eq(x[0:4, 0:4].sum(), arr[0:4, 0:4].sum())


def test_region_numpy_slice():
    """Slice pushdown eagerly slices NumPy sources."""
    arr = np.arange(10000).reshape(100, 100)
    x = da.from_array(arr, chunks=(10, 10))
    # Use a slice that fits within a single chunk
    y = x[12:18, 34:39]

    opt = y.expr.optimize()

    # NumPy arrays are cheap to slice during optimization, unlike zarr/h5py-like
    # sources that need deferred region reads.
    assert opt.operand("_region") is None
    assert opt.array.shape == (6, 5)
    np.testing.assert_array_equal(opt.array, arr[12:18, 34:39])
    # Chunks should be for the sliced region (6x5)
    assert opt.chunks == ((6,), (5,))

    # Verify correctness
    assert_eq(y, arr[12:18, 34:39])


def test_region_numpy_full_slice_does_not_copy():
    arr = np.arange(10000).reshape(100, 100)
    x = da.from_array(arr, chunks=(10, 10))

    opt = x[:, :].expr.optimize()

    assert opt.array is x.expr.array
    assert opt.operand("_region") is None


def test_region_numpy_large_slice_stays_deferred(monkeypatch):
    import dask_array.io._from_array as from_array_mod

    monkeypatch.setattr(from_array_mod, "_NUMPY_SLICE_PUSHDOWN_NBYTES_LIMIT", 16)
    arr = np.arange(100).reshape(10, 10)
    x = da.from_array(arr, chunks=(5, 5))

    opt = x[:5, :5].expr.optimize()

    assert opt.array is x.expr.array
    assert opt.operand("_region") == (slice(None, 5), slice(None, 5))
    assert opt.chunks == ((5,), (5,))


def test_region_single_chunk():
    """Slice within a single chunk produces one task with direct slice."""
    arr = np.arange(10000 * 10000).reshape(10000, 10000)
    x = da.from_array(arr, chunks=(1000, 1000))
    # Small slice within a single chunk
    y = x[1500:1550, 2300:2350]

    opt = y.expr.optimize()
    graph = dict(opt.__dask_graph__())

    # Should be single task (slice fits within one chunk)
    task_keys = [k for k in graph if isinstance(k, tuple) and len(k) == 3]
    assert len(task_keys) == 1

    # The slice should be direct (1500:1550, 2300:2350), not via 1000x1000 chunk
    graph_str = str(graph)
    assert "1000" not in graph_str, "Should slice directly, not via full chunk"

    # Verify correctness
    assert_eq(y, arr[1500:1550, 2300:2350])


def test_region_multiple_chunks():
    """Slice spanning multiple chunks still produces multiple tasks."""
    arr = np.arange(10000).reshape(100, 100)
    x = da.from_array(arr, chunks=(10, 10))
    # Slice spanning 2x2 chunks: 15-25 spans chunks 1,2 in first dim
    # 35-45 spans chunks 3,4 in second dim
    y = x[15:25, 35:45]

    opt = y.expr.optimize()
    graph = dict(opt.__dask_graph__())

    # Should be 2x2=4 tasks (slice spans multiple chunks)
    task_keys = [k for k in graph if isinstance(k, tuple) and len(k) == 3]
    assert len(task_keys) == 4

    # Verify correctness
    assert_eq(y, arr[15:25, 35:45])


def test_region_zarr_deferred(tmp_path):
    """Zarr slicing is deferred - graph contains zarr array, not numpy data."""
    zarr = pytest.importorskip("zarr")
    # Create zarr array
    zarr_path = tmp_path / "test.zarr"
    z = zarr.open(
        str(zarr_path),
        mode="w",
        shape=(10000, 10000),
        dtype="float64",
        chunks=(1000, 1000),
    )
    z[1500:1550, 2300:2350] = np.arange(2500).reshape(50, 50)

    x = da.from_zarr(str(zarr_path))
    y = x[1500:1550, 2300:2350]

    opt = y.expr.optimize()
    graph = dict(opt.__dask_graph__())

    # Should have zarr array in graph, not numpy data
    zarr_arrays = [v for v in graph.values() if isinstance(v, zarr.Array)]
    numpy_arrays = [v for v in graph.values() if isinstance(v, np.ndarray)]

    assert len(zarr_arrays) == 1, "Graph should contain the zarr array"
    assert len(numpy_arrays) == 0, "Graph should not contain numpy arrays (data not loaded)"

    # The zarr array in graph should be the full array, not sliced
    assert zarr_arrays[0].shape == (10000, 10000)

    # Verify correctness
    assert_eq(y, z[1500:1550, 2300:2350])


def test_integer_indexing_pushdown():
    """Integer indexing uses region pushdown to minimize data loading."""
    arr = np.arange(100).reshape(10, 10)
    x = da.from_array(arr, chunks=(5, 5))

    # Pure integer indexing - should be 2 tasks (FromArray + extract)
    y = x[3, 7]
    opt = y.optimize()
    assert len(opt.__dask_graph__()) == 2

    # The inner FromArray should already hold the one-cell NumPy region.
    from_array_expr = opt.expr.array
    assert from_array_expr.operand("_region") is None
    assert from_array_expr.array.shape == (1, 1)
    np.testing.assert_array_equal(from_array_expr.array, arr[3:4, 7:8])

    assert_eq(y, arr[3, 7])

    # Mixed slice + integer
    y = x[:3, 5]
    assert_eq(y, arr[:3, 5])

    y = x[5, 2:8]
    assert_eq(y, arr[5, 2:8])


# ============================================================
# Slice through reduction tests
# ============================================================


def test_slice_through_reduction_optimization():
    """Verify slice pushdown through reduction produces equivalent result.

    x.sum(axis=0)[:5] should simplify to x[:, :5].sum(axis=0)
    """
    x = da.ones((100, 100), chunks=(10, 10))

    # The naive way: full sum then slice
    y = x.sum(axis=0)[:5]

    # The optimized way: slice first, then sum
    expected = x[:, :5].sum(axis=0)

    # After simplification, the names should be equivalent
    # (both sides need simplify since slices also simplify through ones)
    assert y.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_reduction_reduces_tasks():
    """Slice pushdown through reduction should reduce graph size.

    For a from_array with (10, 10) chunks, slicing after reduction
    should result in fewer tasks than computing the full reduction.
    """
    arr = np.arange(10000).reshape(100, 100)
    x = da.from_array(arr, chunks=(10, 10))

    # Full reduction has 10*10 input chunks
    full_sum = x.sum(axis=0)
    full_tasks = len(full_sum.optimize().__dask_graph__())

    # Sliced reduction should have fewer tasks
    sliced_sum = x.sum(axis=0)[:5]
    sliced_tasks = len(sliced_sum.optimize().__dask_graph__())

    # Slicing to first 5 elements (1 chunk column) should have ~10x fewer tasks
    assert sliced_tasks < full_tasks

    # Verify the reduction is correct
    assert_eq(sliced_sum, arr.sum(axis=0)[:5])


def test_slice_through_reduction_axis1():
    """Slice pushdown through sum(axis=1)."""
    x = da.ones((100, 100), chunks=(10, 10))

    # x.sum(axis=1)[:5] should simplify to x[:5, :].sum(axis=1)
    y = x.sum(axis=1)[:5]
    expected = x[:5, :].sum(axis=1)

    assert y.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_reduction_3d():
    """Slice pushdown through reduction on 3D array."""
    x = da.ones((20, 20, 20), chunks=(5, 5, 5))

    # Reduce axis 1, slice result
    # Output axes: [0, 2] become [0, 1] -> slice [:3, :4] maps to input [:3, :, :4]
    y = x.sum(axis=1)[:3, :4]
    expected = x[:3, :, :4].sum(axis=1)

    assert y.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_reduction_multiple_axes():
    """Slice pushdown through reduction on multiple axes."""
    x = da.ones((20, 20, 20), chunks=(5, 5, 5))

    # Reduce axes 0 and 2, only axis 1 remains
    # Output axis 0 -> input axis 1
    y = x.sum(axis=(0, 2))[:5]
    expected = x[:, :5, :].sum(axis=(0, 2))

    assert y.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_reduction_integer_index():
    """Integer indexing through reduction reduces tasks.

    Integer indices are converted to size-1 slices, pushed through,
    then extracted with [0] at the end.
    """
    arr = np.arange(10000).reshape(100, 100)
    x = da.from_array(arr, chunks=(10, 10))

    # Full reduction
    full_tasks = len(x.sum(axis=0).optimize().__dask_graph__())

    # Integer index should have fewer tasks
    result = x.sum(axis=0)[5]
    indexed_tasks = len(result.optimize().__dask_graph__())

    assert indexed_tasks < full_tasks
    assert_eq(result, arr.sum(axis=0)[5])


# =============================================================================
# Slice through creation expressions (ones, zeros, full, empty)
# =============================================================================


def test_slice_ones_returns_smaller_ones():
    """Slicing ones() returns a new ones() with the sliced shape."""
    from dask_array.creation import Ones

    x = da.ones((100, 100), chunks=(10, 10))
    y = x[:15, :25]

    # After simplification, should be Ones with new shape, not Slice(Ones)
    simplified = y.expr.simplify()
    assert isinstance(simplified, Ones)
    assert simplified.shape == (15, 25)


def test_slice_zeros_returns_smaller_zeros():
    """Slicing zeros() returns a new zeros() with the sliced shape."""
    from dask_array.creation import Zeros

    x = da.zeros((100, 100), chunks=(10, 10))
    y = x[:15, :25]

    simplified = y.expr.simplify()
    assert isinstance(simplified, Zeros)
    assert simplified.shape == (15, 25)


def test_slice_full_returns_smaller_full():
    """Slicing full() returns a new full() with the sliced shape."""
    from dask_array.creation import Full

    x = da.full((100, 100), 42, chunks=(10, 10))
    y = x[:15, :25]

    simplified = y.expr.simplify()
    assert isinstance(simplified, Full)
    assert simplified.shape == (15, 25)
    # Verify fill_value is preserved
    assert_eq(y, np.full((15, 25), 42))


def test_slice_creation_correctness():
    """Verify sliced creation expressions produce correct values."""
    assert_eq(da.ones((100, 100), chunks=10)[:15, :25], np.ones((15, 25)))
    assert_eq(da.zeros((100, 100), chunks=10)[:15, :25], np.zeros((15, 25)))
    assert_eq(da.full((100, 100), 7.5, chunks=10)[:15, :25], np.full((15, 25), 7.5))


def test_slice_creation_preserves_dtype():
    """Verify sliced creation preserves dtype."""
    x = da.ones((100, 100), chunks=10, dtype="int32")[:15, :25]
    assert x.dtype == np.dtype("int32")
    assert_eq(x, np.ones((15, 25), dtype="int32"))


# =============================================================================
# Slice through Concatenate
# =============================================================================


def test_slice_through_concat_same_axis_first_array():
    """Slice entirely within first array of concat -> just first array sliced."""
    a = da.ones((10, 5), chunks=5)
    b = da.ones((10, 5), chunks=5)
    result = da.concatenate([a, b], axis=0)[:5]  # Only needs 'a'
    expected = a[:5]

    assert result.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_concat_same_axis_spans_arrays():
    """Slice spans multiple arrays in concat."""
    a = da.ones((10, 5), chunks=5)
    b = da.ones((10, 5), chunks=5)
    c = da.ones((10, 5), chunks=5)
    # slice 5:15 spans a[5:10] and b[0:5]
    result = da.concatenate([a, b, c], axis=0)[5:15]
    expected = da.concatenate([a[5:], b[:5]], axis=0)

    assert result.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_concat_different_axis():
    """Slice on different axis than concat -> push to all inputs."""
    a = da.ones((10, 20), chunks=5)
    b = da.ones((10, 20), chunks=5)
    result = da.concatenate([a, b], axis=0)[:, :5]  # Slice axis 1
    expected = da.concatenate([a[:, :5], b[:, :5]], axis=0)

    assert result.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_concat_correctness():
    """Verify slice through concat produces correct values."""
    a = np.arange(20).reshape(4, 5)
    b = np.arange(20, 40).reshape(4, 5)
    da_a = da.from_array(a, chunks=2)
    da_b = da.from_array(b, chunks=2)

    # Same axis slice
    result = da.concatenate([da_a, da_b], axis=0)[:3]
    assert_eq(result, np.concatenate([a, b], axis=0)[:3])

    # Different axis slice
    result = da.concatenate([da_a, da_b], axis=0)[:, :3]
    assert_eq(result, np.concatenate([a, b], axis=0)[:, :3])

    # Slice spanning both arrays
    result = da.concatenate([da_a, da_b], axis=0)[2:6]
    assert_eq(result, np.concatenate([a, b], axis=0)[2:6])


def test_slice_through_concat_reduces_tasks():
    """Verify slice through concat reduces task count."""
    a = da.ones((100, 100), chunks=10)
    b = da.ones((100, 100), chunks=10)
    concat = da.concatenate([a, b], axis=0)

    full_tasks = len(concat.optimize().__dask_graph__())
    # Slice only first 5 rows - should only need first array
    sliced_tasks = len(concat[:5].optimize().__dask_graph__())

    assert sliced_tasks < full_tasks


# =============================================================================
# Slice through Stack
# =============================================================================


def test_slice_through_stack_selects_subset():
    """Slice on stacked axis selects subset of inputs."""
    a = da.ones((10, 5), chunks=5)
    b = da.ones((10, 5), chunks=5)
    c = da.ones((10, 5), chunks=5)
    # stack gives shape (3, 10, 5), slice [:1] should be stack([a])
    result = da.stack([a, b, c], axis=0)[:1]
    expected = da.stack([a], axis=0)

    assert result.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_stack_other_axis():
    """Slice on non-stacked axis pushes to all inputs."""
    a = da.ones((10, 20), chunks=5)
    b = da.ones((10, 20), chunks=5)
    # stack gives shape (2, 10, 20), slice [:, :5, :10] pushes to each array
    result = da.stack([a, b], axis=0)[:, :5, :10]
    expected = da.stack([a[:5, :10], b[:5, :10]], axis=0)

    assert result.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_stack_mixed():
    """Slice on both stacked and other axes."""
    a = da.ones((10, 20), chunks=5)
    b = da.ones((10, 20), chunks=5)
    c = da.ones((10, 20), chunks=5)
    # stack gives shape (3, 10, 20), slice [:2, :5] keeps a and b, sliced
    result = da.stack([a, b, c], axis=0)[:2, :5]
    expected = da.stack([a[:5], b[:5]], axis=0)

    assert result.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_stack_correctness():
    """Verify slice through stack produces correct values."""
    a = np.arange(20).reshape(4, 5)
    b = np.arange(20, 40).reshape(4, 5)
    c = np.arange(40, 60).reshape(4, 5)
    da_a = da.from_array(a, chunks=2)
    da_b = da.from_array(b, chunks=2)
    da_c = da.from_array(c, chunks=2)

    # Slice on stacked axis
    result = da.stack([da_a, da_b, da_c], axis=0)[:2]
    assert_eq(result, np.stack([a, b, c], axis=0)[:2])

    # Slice on other axis
    result = da.stack([da_a, da_b, da_c], axis=0)[:, :2, :3]
    assert_eq(result, np.stack([a, b, c], axis=0)[:, :2, :3])


def test_slice_through_stack_reduces_tasks():
    """Verify slice through stack reduces task count."""
    a = da.ones((100, 100), chunks=10)
    b = da.ones((100, 100), chunks=10)
    c = da.ones((100, 100), chunks=10)
    stacked = da.stack([a, b, c], axis=0)

    full_tasks = len(stacked.optimize().__dask_graph__())
    # Slice only first array
    sliced_tasks = len(stacked[:1].optimize().__dask_graph__())

    assert sliced_tasks < full_tasks


# =============================================================================
# Slice through BroadcastTo
# =============================================================================


def test_slice_through_broadcast_to_new_dim():
    """Slice on dimension added by broadcast."""
    x = da.ones((10,), chunks=5)
    # broadcast_to adds a new dimension at front: (10,) -> (20, 10)
    result = da.broadcast_to(x, (20, 10))[:5, :]
    expected = da.broadcast_to(x, (5, 10))

    assert result.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_broadcast_to_existing_dim():
    """Slice on dimension that exists in input."""
    x = da.ones((10,), chunks=5)
    # broadcast_to adds new dim: (10,) -> (20, 10)
    result = da.broadcast_to(x, (20, 10))[:, :5]
    expected = da.broadcast_to(x[:5], (20, 5))

    assert result.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_broadcast_to_both_dims():
    """Slice on both new and existing dimensions."""
    x = da.ones((10,), chunks=5)
    result = da.broadcast_to(x, (20, 10))[:5, :3]
    expected = da.broadcast_to(x[:3], (5, 3))

    assert result.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_broadcast_to_broadcasted_dim():
    """Slice on dimension that was size-1 in input."""
    x = da.ones((1, 10), chunks=(1, 5))
    # broadcast_to expands first dim: (1, 10) -> (20, 10)
    result = da.broadcast_to(x, (20, 10))[:5, :3]
    # First dim can't push (was 1), second dim pushes
    expected = da.broadcast_to(x[:, :3], (5, 3))

    assert result.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_broadcast_to_correctness():
    """Verify slice through broadcast_to produces correct values."""
    x = np.arange(10)
    da_x = da.from_array(x, chunks=5)

    # Broadcast to 2D then slice
    result = da.broadcast_to(da_x, (20, 10))[:5, :3]
    expected = np.broadcast_to(x, (20, 10))[:5, :3]
    assert_eq(result, expected)


def test_slice_through_broadcast_to_reduces_tasks():
    """Verify slice through broadcast_to reduces task count."""
    x = da.ones((100,), chunks=10)
    broadcasted = da.broadcast_to(x, (100, 100))

    full_tasks = len(broadcasted.optimize().__dask_graph__())
    # Slice to smaller output
    sliced_tasks = len(broadcasted[:5, :5].optimize().__dask_graph__())

    assert sliced_tasks < full_tasks


# --- Shuffle (take) through Elemwise Tests ---


def test_shuffle_pushes_through_elemwise_add():
    """(x + y)[[1,3,5]] should optimize to x[[1,3,5]] + y[[1,3,5]]."""
    x = da.arange(20, chunks=5)
    y = da.arange(20, chunks=5)

    indices = [1, 3, 5, 7, 9]
    result = (x + y)[indices]
    expected = x[indices] + y[indices]

    # Structure should match
    assert result.expr.simplify()._name == expected.expr.simplify()._name

    # Verify correctness
    x_np = np.arange(20)
    y_np = np.arange(20)
    assert_eq(result, (x_np + y_np)[indices])


def test_shuffle_pushes_through_elemwise_mul():
    """(x * y)[[2,4,6]] should optimize to x[[2,4,6]] * y[[2,4,6]]."""
    x = da.arange(30, chunks=10)
    y = da.arange(30, chunks=10)

    indices = [2, 4, 6, 8]
    result = (x * y)[indices]
    expected = x[indices] * y[indices]

    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, expected)


def test_shuffle_pushes_through_elemwise_2d():
    """Shuffle on 2D array along axis 0."""
    x = da.ones((10, 8), chunks=(5, 4))
    y = da.ones((10, 8), chunks=(5, 4))

    indices = [0, 2, 4, 6]
    result = (x + y)[indices, :]
    expected = x[indices, :] + y[indices, :]

    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, expected)


def test_shuffle_pushes_through_elemwise_scalar():
    """Shuffle through elemwise with scalar."""
    x = da.arange(20, chunks=5)

    indices = [1, 5, 9, 13]
    result = (x + 1)[indices]
    expected = x[indices] + 1

    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, expected)


def test_shuffle_pushes_through_unary_elemwise():
    """Shuffle through unary elemwise (e.g. negative)."""
    x = da.arange(20, chunks=5)

    indices = [2, 4, 6, 8]
    result = (-x)[indices]
    expected = -(x[indices])

    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, expected)


def test_shuffle_through_elemwise_reduces_work():
    """Taking a subset should reduce computation by only computing needed elements."""
    x = da.ones((100,), chunks=10)
    y = da.ones((100,), chunks=10)

    # Take only 10 of 100 elements
    indices = list(range(0, 100, 10))  # [0, 10, 20, ..., 90]
    result = (x + y)[indices]

    # Optimized should have fewer tasks since we only compute what we need
    unopt_tasks = len(result.__dask_graph__())
    opt_tasks = len(result.optimize().__dask_graph__())

    # Optimization should reduce task count
    assert opt_tasks <= unopt_tasks


def test_shuffle_through_elemwise_with_broadcast_2d():
    """Shuffle through elemwise with 2D broadcast operand (size-1 dimension).

    (a * y2d)[[5]] should optimize to a[[5]] * y2d (shuffle only non-broadcast input).
    """
    a = da.from_array(np.arange(200).reshape(10, 20), chunks=(4, 5))
    y2d = da.from_array(np.arange(20).reshape(1, 20), chunks=(1, 20))

    result = (a * y2d)[[5]]
    expected = a[[5]] * y2d

    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, expected)


def test_shuffle_through_elemwise_with_broadcast_1d():
    """Shuffle through elemwise with 1D broadcast operand.

    (a * y1d)[[5]] should optimize to a[[5]] * y1d (shuffle only the 2D input).
    """
    a = da.from_array(np.arange(200).reshape(10, 20), chunks=(4, 5))
    y1d = da.from_array(np.arange(20), chunks=20)

    result = (a * y1d)[[5]]
    expected = a[[5]] * y1d

    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, expected)


# --- Shuffle through Transpose Tests ---


def test_shuffle_pushes_through_transpose():
    """x.T[[1,3,5]] should optimize to x[:, [1,3,5]].T."""
    x = da.arange(20, chunks=5).reshape((4, 5))

    indices = [1, 3]
    result = x.T[indices, :]  # Take rows 1, 3 from transposed (5, 4)
    expected = x[:, indices].T  # Take cols 1, 3 from (4, 5), then transpose

    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, expected)


def test_shuffle_pushes_through_transpose_axis1():
    """x.T[:, [0,2]] should optimize to x[[0,2], :].T."""
    x = da.arange(20, chunks=5).reshape((4, 5))

    indices = [0, 2]
    result = x.T[:, indices]  # Take cols 0, 2 from transposed (5, 4)
    expected = x[indices, :].T  # Take rows 0, 2 from (4, 5), then transpose

    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, expected)


def test_shuffle_pushes_through_transpose_3d():
    """Shuffle through 3D transpose."""
    x = da.ones((2, 3, 4), chunks=2)

    indices = [0, 2]
    # Transpose (2,3,4) -> (4,3,2), then take along axis 0
    result = x.transpose((2, 1, 0))[indices, :, :]
    # Equivalent: take along axis 2 of original, then transpose
    expected = x[:, :, indices].transpose((2, 1, 0))

    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, expected)


# --- Shuffle through Concatenate/Stack Tests ---


def test_shuffle_pushes_through_concatenate():
    """Shuffle on non-concat axis pushes to all inputs."""
    a = da.arange(20, chunks=5).reshape((4, 5))
    b = da.arange(20, 40, chunks=5).reshape((4, 5))

    concat = da.concatenate([a, b], axis=1)  # (4, 10)
    indices = [0, 2]
    result = concat[indices, :]  # Take rows 0, 2

    expected = da.concatenate([a[indices, :], b[indices, :]], axis=1)

    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, expected)


def test_shuffle_pushes_through_stack():
    """Shuffle on non-stack axis pushes to all inputs."""
    a = da.arange(12, chunks=4).reshape((3, 4))
    b = da.arange(12, 24, chunks=4).reshape((3, 4))

    stacked = da.stack([a, b], axis=0)  # (2, 3, 4)
    indices = [0, 2]
    result = stacked[:, indices, :]  # Take along axis 1

    expected = da.stack([a[indices, :], b[indices, :]], axis=0)

    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, expected)


# --- Shuffle through Blockwise Tests ---


def test_shuffle_pushes_through_blockwise():
    """Shuffle through blockwise when adjust_chunks doesn't affect shuffle axis."""
    from dask_array._blockwise import Blockwise

    # map_blocks creates a generic Blockwise with no adjust_chunks
    x = da.ones((4, 6), chunks=(2, 3))
    mapped = x.map_blocks(lambda b: b * 2)

    indices = [0, 2]
    result = mapped[indices, :]

    # Expected: shuffle first, then map_blocks
    expected = x[indices, :].map_blocks(lambda b: b * 2)

    # Verify the optimization happened - Blockwise should be at top
    opt = result.expr.simplify()
    assert isinstance(opt, Blockwise)

    # Verify correctness
    assert_eq(result, expected)


def test_shuffle_does_not_push_through_blockwise_adjust_chunks():
    """Shuffle does NOT push through blockwise when adjust_chunks affects shuffle axis."""
    from dask_array._shuffle import Shuffle

    # map_blocks with explicit chunks sets adjust_chunks
    x = da.ones((8, 6), chunks=(2, 3))
    # Providing chunks means each output block has these chunk sizes (adjust_chunks)
    # This creates output with shape (4, 6) chunks (1, 3)
    mapped = x.map_blocks(lambda b: b * 2, chunks=(1, 3))

    indices = [0, 2]  # Taking along axis 0 - NOT all indices
    result = mapped[indices, :]

    # Shuffle should stay at top (not push through) because axis 0 has adjust_chunks
    opt = result.expr.simplify()
    assert isinstance(opt, Shuffle)

    # Still correct
    assert_eq(result, mapped.compute()[indices, :])


def test_shuffle_not_pushed_into_shared_node():
    """A pushed shuffle re-derives the node in full (same elements,
    reordered), so pushing into a node another parent consumes duplicates
    the node's work. The shuffle stays above the shared chain."""
    from dask_array._blockwise import Elemwise

    x = da.from_array(np.arange(10000.0).reshape(100, 100), chunks=(10, 10))
    y = (x + 1) * 2
    z = y[[5, 3, 1]].sum() + y.sum()

    simplified = z.expr.simplify()
    elemwise_nodes = [n for n in simplified.walk() if isinstance(n, Elemwise)]
    # add + mul of the shared chain, plus the top-level add of the two sums;
    # a duplicated chain would show five
    assert len(elemwise_nodes) == 3

    xn = np.arange(10000.0).reshape(100, 100)
    yn = (xn + 1) * 2
    assert_eq(z, yn[[5, 3, 1]].sum() + yn.sum())


def test_take_not_dropped_when_all_elemwise_inputs_broadcast():
    """(-x)[[0, 0]] on a length-1 axis: every elemwise input broadcasts on
    the shuffle axis, so no input gets shuffled and the pushdown used to
    drop the take entirely, shrinking the result from (2,) back to (1,)."""
    x = da.from_array(np.array([7.0]), chunks=(1,))
    y = (-x)[[0, 0]]

    assert y.expr.optimize().shape == (2,)
    assert_eq(y, np.array([-7.0, -7.0]))


def test_take_not_dropped_on_broadcast_dim():
    """A take on a dimension broadcast from size 1 changes the axis extent,
    so it is not the no-op that a permutation of identical rows would be."""
    b = da.broadcast_to(da.from_array(np.array([5.0]), chunks=(1,)), (100,))
    y = b[[0, 0, 1]]

    assert y.expr.optimize().shape == (3,)
    assert_eq(y, np.array([5.0, 5.0, 5.0]))


# --- ExpandDims (None indexing) Pushdown Tests ---


def test_none_slice_pushes_through_elemwise():
    """Slice with None pushes slicing through elemwise, keeps expand_dims on top."""
    x = da.ones((10, 10), chunks=5)
    y = da.ones((10, 10), chunks=5)

    # (x + y)[None, :5, :] should optimize to (x[:5] + y[:5])[None, :, :]
    result = (x + y)[None, :5, :]
    expected = (x[:5, :] + y[:5, :])[None, :, :]

    # Structure should match after optimization
    assert result.expr.simplify()._name == expected.expr.simplify()._name

    # Verify correctness
    assert_eq(result, expected)


def test_none_slice_multiple_nones():
    """Slice with multiple Nones pushes through correctly."""
    x = da.arange(20, chunks=5).reshape((4, 5))
    y = da.ones((4, 5), chunks=(4, 5))

    # (x + y)[None, :2, None, :3] -> (x[:2, :3] + y[:2, :3])[None, :, None, :]
    result = (x + y)[None, :2, None, :3]
    expected = (x[:2, :3] + y[:2, :3])[None, :, None, :]

    # Structure should match after optimization
    assert result.expr.simplify()._name == expected.expr.simplify()._name

    # Verify correctness
    assert_eq(result, expected)


def test_none_slice_no_slicing():
    """Slice with only None (dimension expansion) uses ExpandDims."""
    from dask_array.manipulation._expand import ExpandDims

    x = da.ones((10, 10), chunks=5)
    y = da.ones((10, 10), chunks=5)

    # (x + y)[None, :, :] - only dimension expansion, no slicing
    result = (x + y)[None, :, :]

    opt = result.expr.simplify()
    # ExpandDims is used for dimension expansion (not Reshape, for fusion compat)
    assert isinstance(opt, ExpandDims)

    # Verify correctness
    x_np = np.ones((10, 10))
    y_np = np.ones((10, 10))
    assert_eq(result, (x_np + y_np)[None, :, :])


def test_none_slice_through_transpose():
    """Slice with None pushes through transpose."""
    x = da.arange(20, chunks=5).reshape((4, 5))

    # x.T[None, :3, :2] -> x[:2, :3].T[None, :, :]
    result = x.T[None, :3, :2]
    expected = x[:2, :3].T[None, :, :]

    # Structure should match after optimization
    assert result.expr.simplify()._name == expected.expr.simplify()._name

    # Verify correctness
    assert_eq(result, expected)


def test_slice_on_keepdims_reduced_axis_values():
    """Indexing the kept reduced axis must not push into the input.

    Found by the optimizer fuzz: the pushdown forwarded the output index to
    the input's reduced axis, so ``sum[0, :]`` returned the first block's
    partial sum instead of the total.
    """
    x = np.arange(6.0).reshape(3, 2)
    d = da.from_array(x, chunks=(1, 1))

    expected = x.sum(axis=0, keepdims=True)
    assert_eq(d.sum(axis=0, keepdims=True)[0, :], expected[0, :])
    assert_eq(d.sum(axis=0, keepdims=True)[0:1, 1], expected[0:1, 1])
    assert_eq(d.mean(axis=1, keepdims=True)[:, 0], x.mean(axis=1, keepdims=True)[:, 0])


def test_empty_slice_on_keepdims_reduced_axis_shape():
    """An empty slice of the kept reduced axis keeps its (empty) shape."""
    x = np.arange(4.0)
    d = da.from_array(x, chunks=2)

    result = d.sum(axis=0, keepdims=True)[0:0]
    expected = x.sum(axis=0, keepdims=True)[0:0]
    assert result.optimize().shape == expected.shape
    assert_eq(result, expected)


def test_slice_on_keepdims_still_pushes_non_reduced_axes():
    """The reduced-axis guard must not disable pushdown on other axes."""
    x = da.from_array(np.arange(10000.0).reshape(100, 100), chunks=(10, 10))
    x_np = np.arange(10000.0).reshape(100, 100)

    sliced = x.sum(axis=0, keepdims=True)[:, 5:20]
    assert_eq(sliced, x_np.sum(axis=0, keepdims=True)[:, 5:20])

    full_tasks = len(x.sum(axis=0, keepdims=True).optimize().__dask_graph__())
    sliced_tasks = len(sliced.optimize().__dask_graph__())
    assert sliced_tasks < full_tasks


def test_empty_source_region_not_dropped():
    """Region pushdown on an already-empty numpy source keeps the region.

    Found by the optimizer fuzz: "region covers the whole source" was tested
    by comparing nbytes, and every region of an empty source has 0 bytes, so
    a region shrinking a different dimension was silently dropped, leaving
    chunks inconsistent with shape.
    """
    x = np.ones((1, 1, 1))
    d = da.from_array(x, chunks=(1, 1, 1))
    leaf = da.from_array(np.ones((1, 0, 1)), chunks=((1,), (0,), (1,)))

    result = (d[:, 0:0, :] + leaf)[:, :, 0:0]
    expected = (x[:, 0:0, :] + np.ones((1, 0, 1)))[:, :, 0:0]

    assert result.optimize().shape == expected.shape
    assert_eq(result, expected)


def test_integer_index_before_expanded_axis():
    """Integer indices on real axes shift expansion axes when pushed through.

    Found by the optimizer fuzz: ExpandDims pushdown only discounted
    integer-removed *expansion* axes when recomputing axis positions, so
    ``expand_dims(x, 1)[0]`` re-expanded at the wrong position (transposed
    shape), and the 1-d variant crashed with AxisError.
    """
    x2 = np.arange(2.0).reshape(1, 2)
    d2 = da.from_array(x2, chunks=(1, 1))
    result = da.expand_dims(d2, 1)[0, :, :]
    expected = np.expand_dims(x2, 1)[0, :, :]
    assert result.optimize().shape == expected.shape
    assert_eq(result, expected)

    x1 = np.arange(1.0)
    d1 = da.from_array(x1, chunks=1)
    result = da.expand_dims(d1, 1)[0, :]
    expected = np.expand_dims(x1, 1)[0, :]
    assert result.optimize().shape == expected.shape
    assert_eq(result, expected)


def test_slice_not_pushed_into_shared_node():
    """Pushing a slice into a node another parent consumes whole duplicates
    the node's work: the full result is materialized anyway, and slicing its
    output is free. y must stay shared here, not split into a sliced copy."""
    from dask_array._blockwise import Elemwise

    x = da.from_array(np.arange(10000.0).reshape(100, 100), chunks=(10, 10))
    y = (x + 1) * 2
    z = y[:99].sum() + y.sum()

    simplified = z.expr.simplify()
    elemwise_nodes = [n for n in simplified.walk() if isinstance(n, Elemwise)]
    # add + mul of the shared chain, plus the top-level add of the two sums;
    # a duplicated chain would show five
    assert len(elemwise_nodes) == 3

    xn = np.arange(10000.0).reshape(100, 100)
    yn = (xn + 1) * 2
    assert_eq(z, yn[:99].sum() + yn.sum())


def test_slice_not_pushed_into_shared_leaf():
    """Same rule at the IO leaf: a region-read duplicates I/O the full
    consumer performs anyway."""
    x = da.from_array(np.arange(10000.0).reshape(100, 100), chunks=(10, 10))
    z = x[:5].sum() + x.sum()

    simplified = z.expr.simplify()
    from_arrays = {n._name for n in simplified.walk() if isinstance(n, FromArray)}
    assert len(from_arrays) == 1

    xn = np.arange(10000.0).reshape(100, 100)
    assert_eq(z, xn[:5].sum() + xn.sum())


def test_multi_window_slices_still_push():
    """When every dependent is a slice, pushing them all means the shared
    node is never computed in full anywhere - keep that win."""
    x = da.from_array(np.arange(10000.0).reshape(100, 100), chunks=(10, 10))
    y = (x + 1) * 2
    z = y[:5] + y[10:15]
    expected = ((x[:5] + 1) * 2) + ((x[10:15] + 1) * 2)

    assert z.expr.simplify()._name == expected.expr.simplify()._name

    xn = np.arange(10000.0).reshape(100, 100)
    yn = (xn + 1) * 2
    assert_eq(z, yn[:5] + yn[10:15])


def test_multi_window_slices_with_grid_sensitive_consumer():
    """A node fanned out to two different-window slices where one slice feeds a
    grid-sensitive op (``map_overlap``).  The multi-window unlink drops the
    shared node's input links when the first sibling pushes, before the
    grid-sensitive sibling's push is declined by ``_preserve_grid_contract`` --
    so the shared node survives under that sibling with a stale gate.  That is a
    lost-sharing pessimization at most; the computed values must still match
    numpy.  (Regression guard for the one non-output-neutral path of the
    multi-window slice-unlink.)"""
    arr = np.arange(400.0).reshape(20, 20)
    n = da.from_array(arr, chunks=(5, 20)) + 1.0  # shared node, two slice consumers

    # window 1 -> grid-sensitive map_overlap (its halo'd input resists slice pushdown)
    w1 = n[2:18, :].map_overlap(lambda b: b * 2.0, depth={0: 1, 1: 0}, boundary="none")
    # window 2 -> plain elemwise (this slice fires + unlinks first)
    w2 = n[5:15, :] * 3.0

    base = arr + 1.0
    assert_eq(w1, base[2:18, :] * 2.0)
    assert_eq(w2, base[5:15, :] * 3.0)

    # both windows in one graph over the shared node
    combined = da.concatenate([w1[:6], w2[:6]], axis=0)
    assert_eq(
        combined,
        np.concatenate([base[2:18, :][:6] * 2.0, base[5:15, :][:6] * 3.0], axis=0),
    )
