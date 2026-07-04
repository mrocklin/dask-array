"""Tests for slice pushdown through MapOverlap.

These tests verify that slicing operations can be pushed through map_overlap
operations, reducing computation by slicing input arrays before applying
overlap boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest

import dask_array as da
from dask_array._test_utils import assert_eq


def add_neighbors(x):
    """Add neighboring values along axis 0. Uses overlap data."""
    result = x.copy()
    if x.shape[0] > 2:
        result[1:-1] = x[:-2] + x[1:-1] + x[2:]
    return result


# =============================================================================
# Case 1: Slice on non-overlap axis (should push through)
# =============================================================================


def test_slice_through_overlap_non_overlap_axis():
    """Slice on axis without overlap pushes through."""
    arr = np.arange(10000).reshape((100, 100)).astype(float)
    x = da.from_array(arr, chunks=(10, 10))

    # Overlap only on axis 0
    result = x.map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")

    # Slice on axis 1 (no overlap) - should be equivalent to slicing input first
    sliced = result[:, :20]
    expected = x[:, :20].map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")

    # Verify expression structure matches
    assert sliced.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_overlap_middle_slice():
    """Slice in the middle of non-overlap axis."""
    arr = np.arange(10000).reshape((100, 100)).astype(float)
    x = da.from_array(arr, chunks=(10, 10))

    result = x.map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")

    # Middle slice on axis 1 (no overlap)
    sliced = result[:, 30:70]
    expected = x[:, 30:70].map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")

    assert sliced.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_overlap_correctness():
    """Verify slice through overlap produces correct values."""
    arr = np.arange(64).reshape((8, 8)).astype(float)
    x = da.from_array(arr, chunks=(4, 4))

    result = x.map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")

    # Slice on axis 1
    sliced = result[:, 2:6]
    expected = x[:, 2:6].map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")

    assert sliced.expr.simplify()._name == expected.expr.simplify()._name


# =============================================================================
# Case 2: Slice on overlap axis (pushes through with padding)
# =============================================================================


def test_slice_on_overlap_axis_pushes_with_padding():
    """Slice on axis with overlap pushes through with padded input."""
    arr = np.arange(10000).reshape((100, 100)).astype(float)
    x = da.from_array(arr, chunks=(10, 10))

    result = x.map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")

    # Slice on axis 0 (has overlap) - should push through with padded input
    # [:50] with depth=2 needs input [:52], then trim to [:50]
    sliced = result[:50, :]
    expected = x[:52, :].map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")[:50, :]

    assert sliced.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_on_both_axes_one_has_overlap():
    """Slice on both axes when one has overlap."""
    arr = np.arange(10000).reshape((100, 100)).astype(float)
    x = da.from_array(arr, chunks=(10, 10))

    result = x.map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")
    sliced = result[:50, :50]

    # Axis 1 has no overlap: slice pushes directly
    # Axis 0 has depth=2: need padded input [:52], then trim to [:50]
    expected = x[:52, :50].map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")[:50, :]

    assert sliced.expr.simplify()._name == expected.expr.simplify()._name


# =============================================================================
# Case 3: Multi-dimensional overlap
# =============================================================================


def add_neighbors_2d(x):
    """Add neighboring values along both axes. Uses overlap data."""
    result = x.copy()
    if x.shape[0] > 2:
        result[1:-1, :] += x[:-2, :] + x[2:, :]
    if x.shape[1] > 2:
        result[:, 1:-1] += x[:, :-2] + x[:, 2:]
    return result


def lag1(x):
    result = np.full_like(x, np.nan)
    result[1:] = x[:-1]
    return result


def test_slice_through_2d_overlap():
    """Slice through 2D overlap - pushes when beneficial."""
    arr = np.arange(10000).reshape((100, 100)).astype(float)
    x = da.from_array(arr, chunks=(10, 10))

    result = x.map_overlap(add_neighbors_2d, depth={0: 1, 1: 1}, boundary="none")

    # Slice on axis 1 with depth=1 needs input [:, :41], then trim to [:, :40]
    sliced = result[:, :40]
    expected = x[:, :41].map_overlap(add_neighbors_2d, depth={0: 1, 1: 1}, boundary="none")[:, :40]

    assert sliced.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_2d_overlap_middle():
    """Middle slice through 2D overlap on non-overlap dimension."""
    arr = np.arange(10000).reshape((100, 100)).astype(float)
    x = da.from_array(arr, chunks=(10, 10))

    # Overlap only on axis 0
    result = x.map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")

    # Middle slice on axis 1 (no overlap)
    sliced = result[:, 25:75]
    expected = x[:, 25:75].map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")

    assert sliced.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_through_1d_overlap_on_3d_array():
    """Slice on multiple non-overlap axes."""
    arr = np.arange(1000).reshape((10, 10, 10)).astype(float)
    x = da.from_array(arr, chunks=(5, 5, 5))

    # Overlap only on axis 0
    result = x.map_overlap(add_neighbors, depth={0: 1, 1: 0, 2: 0}, boundary="none")

    # Slice on axes 1 and 2 (neither has overlap)
    sliced = result[:, :3, :3]
    expected = x[:, :3, :3].map_overlap(add_neighbors, depth={0: 1, 1: 0, 2: 0}, boundary="none")

    assert sliced.expr.simplify()._name == expected.expr.simplify()._name


# =============================================================================
# Case 4: Asymmetric overlap
# =============================================================================


def test_slice_through_asymmetric_overlap():
    """Slice through asymmetric overlap (different left/right depth)."""
    arr = np.arange(64).reshape((8, 8)).astype(float)
    x = da.from_array(arr, chunks=(4, 4))

    # Asymmetric overlap on axis 0
    result = x.map_overlap(add_neighbors, depth={0: (2, 1), 1: 0}, boundary="none")

    # Slice on axis 1 (no overlap)
    sliced = result[:, 2:6]
    expected = x[:, 2:6].map_overlap(add_neighbors, depth={0: (2, 1), 1: 0}, boundary="none")

    assert sliced.expr.simplify()._name == expected.expr.simplify()._name


def test_slice_on_asymmetric_overlap_axis_pushes():
    """Slice on axis with asymmetric overlap pushes through with padding."""
    arr = np.arange(10000).reshape((100, 100)).astype(float)
    x = da.from_array(arr, chunks=(10, 10))

    result = x.map_overlap(add_neighbors, depth={0: (2, 1), 1: 0}, boundary="none")

    # Slice axis 0 with asymmetric depth (2, 1) - needs extra 1 on right
    # [:50] needs input [:51], then trim to [:50]
    sliced = result[:50, :]
    expected = x[:51, :].map_overlap(add_neighbors, depth={0: (2, 1), 1: 0}, boundary="none")[:50, :]

    assert sliced.expr.simplify()._name == expected.expr.simplify()._name


# =============================================================================
# Case 5: Zero overlap (edge case)
# =============================================================================


def test_slice_through_zero_overlap():
    """Slice through axis with zero overlap pushes through."""
    arr = np.arange(10000).reshape((100, 100)).astype(float)
    x = da.from_array(arr, chunks=(10, 10))

    # Zero overlap - no actual overlap computation needed
    result = x.map_overlap(add_neighbors, depth=0, boundary="none")

    # Slice on axis 0 - with zero overlap, slice should push through
    sliced = result[:50, :]
    expected = x[:50, :].map_overlap(add_neighbors, depth=0, boundary="none")

    assert sliced.expr.simplify()._name == expected.expr.simplify()._name


# =============================================================================
# Case 6: Task reduction verification
# =============================================================================


def test_slice_through_overlap_reduces_tasks():
    """Verify slice pushdown reduces task count."""
    arr = np.arange(10000).reshape((100, 100)).astype(float)
    x = da.from_array(arr, chunks=(10, 10))

    result = x.map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")

    full = result
    sliced = result[:, :10]  # Take only first 10 columns

    full_tasks = len(full.optimize().__dask_graph__())
    sliced_tasks = len(sliced.optimize().__dask_graph__())

    # Sliced should have fewer tasks (processes 1 column of chunks vs 10)
    assert sliced_tasks < full_tasks


def test_slice_through_overlap_reduces_numblocks():
    """Verify slice pushdown reduces number of output blocks."""
    arr = np.arange(10000).reshape((100, 100)).astype(float)
    x = da.from_array(arr, chunks=(10, 10))

    result = x.map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")
    sliced = result[:, :10]

    # Full result: 10x10 chunks
    assert result.numblocks == (10, 10)

    # Sliced result: 10x1 chunks (only 1 column of blocks)
    assert sliced.numblocks == (10, 1)


# =============================================================================
# Case 7: Correctness with computed values
# =============================================================================


@pytest.mark.parametrize(
    "shape,chunks,depth,slice_",
    [
        # Start slices (:n form) on non-overlap axes
        ((80, 80), (20, 20), {0: 2, 1: 0}, (slice(None), slice(20))),
        ((80, 80), (20, 20), {0: 0, 1: 2}, (slice(20), slice(None))),
        # Middle slices (k:n form) on non-overlap axes
        ((80, 80), (20, 20), {0: 2, 1: 0}, (slice(None), slice(20, 60))),
        ((80, 80), (20, 20), {0: 0, 1: 2}, (slice(20, 60), slice(None))),
        # End slices (k: form) on non-overlap axes
        ((80, 80), (20, 20), {0: 2, 1: 0}, (slice(None), slice(40, None))),
        ((80, 80), (20, 20), {0: 0, 1: 2}, (slice(40, None), slice(None))),
    ],
)
def test_slice_through_overlap_parametrized(shape, chunks, depth, slice_):
    """Parametrized correctness tests for slice through overlap."""
    arr = np.arange(np.prod(shape)).reshape(shape).astype(float)
    x = da.from_array(arr, chunks=chunks)

    result = x.map_overlap(add_neighbors, depth=depth, boundary="none")
    sliced = result[slice_]

    # Build expected: slice input first, then overlap
    input_sliced = x[slice_]
    expected = input_sliced.map_overlap(add_neighbors, depth=depth, boundary="none")

    # Verify expression structure matches
    assert sliced.expr.simplify()._name == expected.expr.simplify()._name


# =============================================================================
# Case 8: Special cases (trim=False, uniform depth)
# =============================================================================


def test_map_overlap_no_trim_slice_pushes():
    """With trim=False, slice should push through to input."""
    arr = np.arange(10000).reshape((100, 100)).astype(float)
    x = da.from_array(arr, chunks=(10, 10))

    # With trim=False, there's no Trim wrapper, so slice can push through
    result = x.map_overlap(add_neighbors, depth={0: 2}, boundary="none", trim=False)

    # Slice on axis 1 (no overlap on axis 1) - pushes directly through
    sliced = result[:, :30]
    expected = x[:, :30].map_overlap(add_neighbors, depth={0: 2}, boundary="none", trim=False)

    assert sliced.expr.simplify()._name == expected.expr.simplify()._name


def test_map_overlap_uniform_depth_correctness():
    """Test with uniform depth (int instead of dict).

    When slicing on an axis with overlap, the optimization pads the input
    slice to include data needed for overlap, then trims the output.
    """
    arr = np.arange(10000).reshape((100, 100)).astype(float)
    x = da.from_array(arr, chunks=(10, 10))

    result = x.map_overlap(add_neighbors_2d, depth=2, boundary="none")
    sliced = result[:, :30]

    # Expected: pad input by depth on sliced axis, apply overlap, then trim
    # [:, :30] with depth=2 needs input [:, :32] to preserve overlap semantics
    expected = x[:, :32].map_overlap(add_neighbors_2d, depth=2, boundary="none")[:, :30]

    assert sliced.expr.simplify()._name == expected.expr.simplify()._name


# =============================================================================
# Case 9: Value correctness verification
# =============================================================================


def test_slice_through_overlap_value_correctness():
    """Verify optimized slice produces correct values."""
    arr = np.arange(10000).reshape((100, 100)).astype(float)
    x = da.from_array(arr, chunks=(10, 10))

    result = x.map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")

    # Slice on non-overlap axis
    sliced = result[:, :50]

    # Compare against unoptimized computation
    full_result = result.compute()
    assert_eq(sliced, full_result[:, :50])


def test_slice_on_overlap_axis_value_correctness():
    """Verify slice on overlap axis produces correct values."""
    arr = np.arange(10000).reshape((100, 100)).astype(float)
    x = da.from_array(arr, chunks=(10, 10))

    result = x.map_overlap(add_neighbors_2d, depth=2, boundary="none")

    # Slice on axis with overlap
    sliced = result[:50, :50]

    # Compare against unoptimized computation
    full_result = result.compute()
    assert_eq(sliced, full_result[:50, :50])


def test_nested_overlap_tail_slice_after_rechunk():
    arr = np.arange(30, dtype="float64").reshape(15, 2)
    x = da.from_array(arr, chunks=(3, 2))

    inner = da.map_overlap(lambda block: block, x, depth={0: 4, 1: 0}, boundary="none", trim=True)
    assert inner.chunks[0] == (6, 9)

    outer = da.map_overlap(lag1, inner, depth={0: 1, 1: 0}, boundary=np.nan, trim=True)
    result = outer[8:10]

    expected = np.full_like(arr, np.nan)
    expected[1:] = arr[:-1]
    assert_eq(result, expected[8:10])


def test_nested_overlap_lowers_in_linear_work():
    """A depth-D chain of map_overlap must lower with O(D) work, not O(2**D).

    Regression guard: map_blocks once read its input collection's ``.name``
    while building the trim layer, which forced a full nested re-lowering of the
    intermediate.  Inside overlap's own ``_lower`` that recursed once per nesting
    level, so each added overlap doubled the lowering work (a 16-deep chain took
    ~40s).  ``MapOverlap._lower`` should fire exactly once per overlap.
    """
    from dask_array._overlap import MapOverlap

    # Deep enough that a 2**D regression is unmistakable (2**13 = 8192 >> 13) but
    # shallow enough that a regressed run fails in seconds rather than hanging.
    depth = 12
    x = da.ones((70, 5), chunks=(10, 5))
    for _ in range(depth):
        x = x.map_overlap(lambda b: b, depth={0: 1}, boundary="none")

    calls = []
    original = MapOverlap._lower
    try:
        MapOverlap._lower = lambda self: (calls.append(1), original(self))[1]
        graph = x.__dask_graph__()
    finally:
        MapOverlap._lower = original

    # One lowering per overlap (allow one extra for a benign future pass); the
    # exponential blowup would be 2**(D+1) - 1.
    assert len(calls) <= depth + 1
    assert len(graph) > 0


def test_slice_pushdown_into_nested_overlap_is_correct():
    """Slicing through a nested overlap+rechunk chain must give correct values.

    Regression guard for a silently-wrong result: overlap's ``_lower`` used to
    ``simplify()`` its trim input and rechunk it back to the *un-sliced* grid.
    When a slice pushed into the outer map_overlap (changing the block grid),
    that pin forced the correctly-sliced grid back to the stale one, truncating
    the output (here to shape (3, 2) instead of (5, 2)).
    """
    n = 16
    arr = np.arange(n * 2, dtype="float64").reshape(n, 2)
    x = da.from_array(arr, chunks=(3, 2))
    x = da.map_overlap(lambda b: b * 2.0, x, depth={0: 2, 1: 0}, boundary="none", trim=True)
    x = da.map_overlap(lambda b: b, x, depth={0: 3, 1: 0}, boundary="periodic", trim=True)
    x = x.rechunk((7, 2))
    x = da.map_overlap(lambda b: b * 2.0, x, depth={0: 1, 1: 0}, boundary="periodic", trim=True)
    x = da.map_overlap(lambda b: b * 2.0, x, depth={0: 3, 1: 0}, boundary="periodic", trim=True)

    # Full (no slice pushdown) result is the oracle for the sliced one.
    full = x.compute()
    assert_eq(x[6:11], full[6:11])
