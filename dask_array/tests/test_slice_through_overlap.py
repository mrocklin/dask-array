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
# Case 1: Slice on non-overlap axis
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
    """A no-cull slice stays above overlap but still computes correct values."""
    from dask_array._overlap import MapOverlap
    from dask_array.slicing._basic import SliceSlicesIntegers

    arr = np.arange(64).reshape((8, 8)).astype(float)
    x = da.from_array(arr, chunks=(4, 4))

    result = x.map_overlap(add_neighbors, depth={0: 2, 1: 0}, boundary="none")

    sliced = result[:, 2:6]
    opt = sliced.expr.simplify()

    assert isinstance(opt, SliceSlicesIntegers)
    assert isinstance(opt.array, MapOverlap)
    assert_eq(sliced, result.compute()[:, 2:6])


def test_no_cull_slice_stays_above_overlap_over_computed_input():
    from dask_array._blockwise import Elemwise
    from dask_array._overlap import MapOverlap
    from dask_array.slicing._basic import SliceSlicesIntegers

    arr = np.arange(40.0)
    x = da.from_array(arr, chunks=(8,), asarray=False)

    result = (x + 1).map_overlap(lambda block: block, depth={0: 1}, boundary="none")
    opt = result[7:39].expr.simplify()

    assert isinstance(opt, SliceSlicesIntegers)
    assert isinstance(opt.array, MapOverlap)
    assert isinstance(opt.array.array, Elemwise)
    assert opt.array.array.chunks == ((8, 8, 8, 8, 8),)
    assert_eq(result[7:39], result.compute()[7:39])


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
    """A no-cull slice stays above asymmetric overlap."""
    from dask_array._overlap import MapOverlap
    from dask_array.slicing._basic import SliceSlicesIntegers

    arr = np.arange(64).reshape((8, 8)).astype(float)
    x = da.from_array(arr, chunks=(4, 4))

    # Asymmetric overlap on axis 0
    result = x.map_overlap(add_neighbors, depth={0: (2, 1), 1: 0}, boundary="none")

    sliced = result[:, 2:6]
    opt = sliced.expr.simplify()

    assert isinstance(opt, SliceSlicesIntegers)
    assert isinstance(opt.array, MapOverlap)
    assert_eq(sliced, result.compute()[:, 2:6])


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


def test_tail_slice_with_asymmetric_positive_overlap_lowers():
    day = 8_640
    depth = 25_919
    arr = np.arange(13 * day, dtype="float64")
    x = da.from_array(arr, chunks=(day,))

    def lead1(block):
        result = np.full_like(block, np.nan)
        result[:-1] = block[1:]
        return result

    full = da.map_overlap(lead1, x, depth={0: (0, depth)}, boundary="none", trim=True)
    result = full[-day:]

    assert result.__dask_graph__()
    assert_eq(result, full.compute()[-day:])


def test_tail_slice_with_second_input_asymmetric_overlap_lowers():
    arr = np.arange(50, dtype="float64")
    x = da.from_array(arr, chunks=(10,))
    y = da.from_array(arr * 2, chunks=(10,))

    full = da.map_overlap(
        lambda a, b: a + b,
        x,
        y,
        depth=[{0: 0}, {0: (0, 25)}],
        boundary=["none", "none"],
        trim=True,
    )
    result = full[-10:]

    assert result.__dask_graph__()


def test_periodic_edge_slice_keeps_global_boundary_context():
    arr = np.arange(10, dtype="float64")
    x = da.from_array(arr, chunks=(5,))

    full = da.map_overlap(lag1, x, depth={0: 1}, boundary="periodic", trim=True)

    assert_eq(full[:3], full.compute()[:3])


def test_overlap_axis_slice_with_no_rechunk_lowers():
    arr = np.arange(40, dtype="float64")
    x = da.from_array(arr, chunks=(20,))

    full = da.map_overlap(lambda block: block, x, depth={0: 10}, boundary="none", trim=True, allow_rechunk=False)
    result = full[5:15]

    assert result.__dask_graph__()


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


# =============================================================================
# Boundary chunks smaller than a one-sided overlap depth
# =============================================================================
# With boundary "none" the first block gets no left halo (and the last block
# no right halo), so an edge chunk of size <= depth produces a block smaller
# than a full kernel window (before + after + 1). Moving-window kernels such
# as bottleneck's (xarray's dask rolling path uses depth=(window - 1, 0))
# reject such blocks. overlap() must merge the short edge chunk into its
# neighbor. Integer-valued data keeps the window sums float-exact.


def trailing_window_sum(x):
    """Rolling 10-sum with partial head windows; needs a full window per block."""
    if x.shape[0] < 10:
        raise ValueError(f"block of {x.shape[0]} rows is smaller than the window (10)")
    c = np.cumsum(x, axis=0)
    out = c.astype("float64")
    out[10:] = c[10:] - c[:-10]
    return out


def leading_window_sum(x):
    """Forward-looking 10-sum with partial tail windows; needs a full window per block."""
    if x.shape[0] < 10:
        raise ValueError(f"block of {x.shape[0]} rows is smaller than the window (10)")
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)], axis=0)
    end = np.minimum(np.arange(x.shape[0]) + 10, x.shape[0])
    return c[end] - c[:-1]


@pytest.mark.parametrize(
    "chunks",
    [
        (9, 10, 10, 11),  # short first chunk: exactly window - 1, no left halo
        (10, 9, 10, 11),  # short middle chunk: healed by the neighbor halo
        (11, 10, 10, 9),  # short last chunk: fine for a trailing window
        (10, 10, 10, 10),  # first chunk exactly == window: no merge needed
        (2,) * 20,  # every chunk below depth
        (8, 1, 10, 21),  # depth-merge lands exactly on depth (== window - 1)
    ],
)
def test_map_overlap_short_boundary_chunk_trailing_window(chunks):
    arr = np.arange(160, dtype="float64").reshape(40, 4)
    x = da.from_array(arr, chunks=(chunks, (4,)))
    result = da.map_overlap(
        trailing_window_sum, x, depth={0: (9, 0), 1: 0}, boundary="none", trim=True, dtype="float64"
    )
    assert_eq(result, trailing_window_sum(arr))


def test_map_overlap_short_boundary_chunk_leading_window():
    arr = np.arange(160, dtype="float64").reshape(40, 4)
    x = da.from_array(arr, chunks=((11, 10, 10, 9), (4,)))
    result = da.map_overlap(leading_window_sum, x, depth={0: (0, 9), 1: 0}, boundary="none", trim=True, dtype="float64")
    assert_eq(result, leading_window_sum(arr))


def test_nested_overlap_tail_slice_with_short_first_chunk():
    """Shift stacked on a rolling window, tail-sliced, with a merged first chunk."""
    arr = np.arange(160, dtype="float64").reshape(40, 4)
    x = da.from_array(arr, chunks=((9, 10, 10, 11), (4,)))
    inner = da.map_overlap(trailing_window_sum, x, depth={0: (9, 0), 1: 0}, boundary="none", trim=True, dtype="float64")
    outer = da.map_overlap(lag1, inner, depth={0: (1, 0), 1: 0}, boundary="none", trim=True, dtype="float64")
    result = outer[35:]
    result._lowered_expr  # trim_internal's adjust_chunks grid must stay consistent
    expected = np.full_like(arr, np.nan)
    expected[1:] = trailing_window_sum(arr)[:-1]
    assert_eq(result, expected[35:])


def test_slice_ending_inside_first_window_declines_pushdown():
    """A pushed slice whose expanded extent is shorter than one kernel window
    (before + after + 1) must decline: with boundary "none" the missing rows
    are simply not in the slice, so no rechunk can heal the block and a
    moving-window kernel rejects it. Production signature: xarray rolling
    (depth (window - 1, 0)) sliced with stop == window - 1 crashed worker-side
    with "Moving window (=8640) must between 1 and 8639"."""
    arr = np.arange(160, dtype="float64").reshape(40, 4)
    x = da.from_array(arr, chunks=((10,) * 4, (4,)))
    r = da.map_overlap(trailing_window_sum, x, depth={0: (9, 0), 1: 0}, boundary="none", trim=True, dtype="float64")
    expected = trailing_window_sum(arr)
    assert_eq(r[:9], expected[:9])  # stop == window - 1: expanded extent == depth
    assert_eq(r[3:9], expected[3:9])  # interior start, same clipped extent
    assert_eq(r[:10], expected[:10])  # control: a full window still pushes down


def test_tail_slice_inside_last_window_declines_pushdown():
    arr = np.arange(160, dtype="float64").reshape(40, 4)
    x = da.from_array(arr, chunks=((10,) * 4, (4,)))
    r = da.map_overlap(leading_window_sum, x, depth={0: (0, 9), 1: 0}, boundary="none", trim=True, dtype="float64")
    expected = leading_window_sum(arr)
    assert_eq(r[-9:], expected[-9:])  # expanded extent == after == window - 1
    assert_eq(r[-10:], expected[-10:])  # control
