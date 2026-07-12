from __future__ import annotations

import warnings

import numpy as np
import pytest

import dask_array as da
from dask_array._rechunk import TasksRechunk
from dask_array._test_utils import assert_eq


def _contains_sliding_window_view(expr):
    func = getattr(expr, "func", None)
    if func is np.lib.stride_tricks.sliding_window_view:
        return True
    return any(_contains_sliding_window_view(dep) for dep in expr.dependencies())


def _contains_tasks_rechunk(expr):
    if isinstance(expr, TasksRechunk):
        return True
    return any(_contains_tasks_rechunk(dep) for dep in expr.dependencies())


def _contains_overlap(expr):
    from dask_array._overlap import OverlapInternal

    if isinstance(expr, OverlapInternal):
        return True
    return any(_contains_overlap(dep) for dep in expr.dependencies())


@pytest.mark.parametrize(
    "reduction", ["sum", "mean", "min", "max", "prod", "nansum", "nanmean", "nanmin", "nanmax", "nanprod"]
)
def test_sliding_window_reduction_window_spanning_many_chunks_keeps_native_chunks(reduction):
    # The statarb shape: a rolling window several times larger than the time
    # chunks.  The fused reduction must keep the input's native chunking and
    # never rechunk or overlap up to the window size.
    rng = np.random.default_rng(42)
    data = rng.normal(size=(13 * 96, 3))
    if reduction == "prod" or reduction == "nanprod":
        data = 1 + data / 100
    if reduction.startswith("nan"):
        data[rng.random(data.shape) < 0.2] = np.nan
        data[100:600, 1] = np.nan  # includes all-NaN windows
    x = da.from_array(data, chunks=(96, 2))
    window = 480  # spans five 96-element chunks

    view = da.sliding_window_view(x, window_shape=window, axis=0)
    result = getattr(da, reduction)(view, axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = getattr(np, reduction)(np.lib.stride_tricks.sliding_window_view(data, window, axis=0), axis=-1)

    optimized = result.expr.optimize()
    assert optimized.chunks == ((96,) * 8 + (1,), (2, 1))
    assert not _contains_tasks_rechunk(optimized)
    assert not _contains_overlap(optimized)
    assert_eq(result, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("window", [13, 20])
@pytest.mark.parametrize("reduction", ["sum", "min", "nanmean"])
def test_sliding_window_reduction_irregular_chunks(reduction, window):
    # Right-edge bands crossing multiple small chunks, and (window=13) a
    # chunk larger than the window depth, which falls back to the overlap
    # path; values must be right either way.
    rng = np.random.default_rng(7)
    data = rng.normal(size=80)
    if reduction == "nanmean":
        data[rng.random(80) < 0.3] = np.nan
    x = da.from_array(data, chunks=((7, 12, 9, 14, 8, 12, 6, 12),))

    view = da.sliding_window_view(x, window_shape=window, axis=0)
    result = getattr(da, reduction)(view, axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = getattr(np, reduction)(np.lib.stride_tricks.sliding_window_view(data, window, axis=0), axis=-1)

    if window == 20:  # every chunk fits under the window depth: native path
        assert result.expr.simplify().chunks == ((7, 12, 9, 14, 8, 11),)
    assert_eq(result, expected, rtol=1e-12)


def test_sliding_window_reduction_window_one_past_chunk():
    # depth == chunk size exactly: bands start exactly at block boundaries
    data = np.arange(80, dtype=np.float64)
    x = da.from_array(data, chunks=8)

    result = da.sliding_window_view(x, window_shape=9, axis=0).sum(axis=-1)
    expected = np.lib.stride_tricks.sliding_window_view(data, 9, axis=0).sum(axis=-1)

    assert result.expr.simplify().chunks == ((8,) * 9,)
    assert_eq(result, expected, rtol=1e-13)


@pytest.mark.parametrize("func_name", ["move_sum", "move_mean", "move_min", "move_max"])
@pytest.mark.parametrize("min_count", [1, None, 300])
def test_map_overlap_bottleneck_moving_window_keeps_native_chunks(func_name, min_count):
    # xarray's dask rolling path: map_overlap(bottleneck.move_*, depth=(window-1, 0)).
    # With a window several times the chunk size the reduction must keep the
    # input's native chunks and match bottleneck exactly, NaNs included.
    bn = pytest.importorskip("bottleneck")
    from dask_array.reductions._sliding_window import MovingWindowReduction

    func = getattr(bn, func_name)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(13 * 96, 4))
    data[rng.random(data.shape) < 0.2] = np.nan
    data[100:600, 2] = np.nan  # all-NaN windows
    x = da.from_array(data, chunks=(96, 2))
    window = 480  # spans five 96-element chunks

    result = x.map_overlap(func, depth={0: (window - 1, 0)}, dtype="f8", window=window, min_count=min_count, axis=0)
    expected = func(data, window, min_count=min_count, axis=0)

    optimized = result.expr.optimize()
    assert isinstance(optimized, MovingWindowReduction)
    assert optimized.chunks == x.chunks
    assert not _contains_tasks_rechunk(optimized)
    assert_eq(result, expected, rtol=1e-13, atol=1e-13)


def test_map_overlap_bottleneck_moving_window_irregular_chunks():
    bn = pytest.importorskip("bottleneck")
    rng = np.random.default_rng(1)
    data = rng.normal(size=1248)
    data[rng.random(1248) < 0.2] = np.nan
    x = da.from_array(data, chunks=((100, 51, 96, 96, 200, 96, 313, 200, 96),))
    window = 400

    result = x.map_overlap(bn.move_sum, depth={0: (window - 1, 0)}, dtype="f8", window=window, min_count=1, axis=0)
    expected = bn.move_sum(data, window, min_count=1, axis=0)

    assert result.expr.optimize().chunks == x.chunks
    assert_eq(result, expected, rtol=1e-13, atol=1e-13)


def test_map_overlap_bottleneck_moving_window_large_chunk_falls_back():
    # A chunk bigger than the window can't use the banded plan; the overlap
    # path must still produce the right answer.
    bn = pytest.importorskip("bottleneck")
    from dask_array.reductions._sliding_window import MovingWindowReduction

    rng = np.random.default_rng(2)
    data = rng.normal(size=200)
    x = da.from_array(data, chunks=((30, 110, 30, 30),))
    window = 40

    result = x.map_overlap(bn.move_sum, depth={0: (window - 1, 0)}, dtype="f8", window=window, min_count=1, axis=0)
    expected = bn.move_sum(data, window, min_count=1, axis=0)

    assert not isinstance(result.expr.optimize(), MovingWindowReduction)
    assert_eq(result, expected, rtol=1e-13, atol=1e-13)


def test_sliding_window_sum_large_offset_stays_accurate():
    # A prefix-sum-difference scheme would lose precision here; the banded
    # combine must not.
    rng = np.random.default_rng(3)
    noise = rng.normal(size=12 * 64)
    data = 1e9 + noise
    x = da.from_array(data, chunks=64)
    window = 256

    result = da.sliding_window_view(x, window_shape=window, axis=0).sum(axis=-1)
    windows = np.lib.stride_tricks.sliding_window_view(data, window, axis=0)
    exact = window * 1e9 + np.lib.stride_tricks.sliding_window_view(noise, window, axis=0).sum(axis=-1)

    assert result.expr.simplify().chunks == ((64,) * 8 + (1,),)
    assert_eq(result, exact, rtol=1e-13)
    assert_eq(result, windows.sum(axis=-1), rtol=1e-13)


@pytest.mark.parametrize("reduction", ["min", "max", "sum", "prod", "mean"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sliding_window_reduction_over_window_axis_avoids_window_block(reduction, keepdims):
    data = (1 + (np.arange(80 * 4 * 5, dtype=np.float32) % 5) / 100).reshape(80, 4, 5)
    x = da.from_array(data, chunks=(16, 4, 5))
    y = da.sliding_window_view(x, window_shape=24, axis=0, automatic_rechunk=False)

    result = getattr(y, reduction)(axis=-1, keepdims=keepdims)
    expected = getattr(np.lib.stride_tricks.sliding_window_view(data, 24, axis=0), reduction)(
        axis=-1, keepdims=keepdims
    )

    assert y.chunks == ((32, 25), (4,), (5,), (24,))
    # The fused reduction runs on the input's native 16-element chunks
    # instead of the view's window-sized ones.
    native_chunks = ((16, 16, 16, 9), (4,), (5,)) + (((1,),) if keepdims else ())
    assert result.expr.simplify().chunks == native_chunks
    assert_eq(result, expected, rtol=1e-5)
    assert _contains_sliding_window_view(result.expr)
    assert not _contains_sliding_window_view(result.expr.simplify())


@pytest.mark.parametrize("reduction", ["min", "max", "sum", "prod", "mean"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sliding_window_reduction_keeps_non_window_chunks(reduction, keepdims):
    data = (1 + (np.arange(96 * 32 * 48, dtype=np.float32) % 5) / 100).reshape(96, 32, 48)
    x = da.from_array(data, chunks=(24, 24, 24))

    windowed = da.sliding_window_view(x, window_shape=72, axis=0)
    result = getattr(windowed, reduction)(axis=-1, keepdims=keepdims)
    expected = getattr(np.lib.stride_tricks.sliding_window_view(data, 72, axis=0), reduction)(
        axis=-1, keepdims=keepdims
    )

    expected_chunks = ((24, 1), (24, 8), (24, 24), (1,)) if keepdims else ((24, 1), (24, 8), (24, 24))
    assert result.expr.simplify().chunks == expected_chunks
    assert_eq(result, expected, rtol=1e-5)


@pytest.mark.parametrize("reduction", ["any", "all"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sliding_window_boolean_reduction_keeps_non_window_chunks(reduction, keepdims):
    data = (np.arange(96 * 32 * 48).reshape(96, 32, 48) % 5) == 0
    x = da.from_array(data, chunks=(24, 24, 24))

    windowed = da.sliding_window_view(x, window_shape=72, axis=0)
    result = getattr(windowed, reduction)(axis=-1, keepdims=keepdims)
    expected = getattr(np.lib.stride_tricks.sliding_window_view(data, 72, axis=0), reduction)(
        axis=-1, keepdims=keepdims
    )

    expected_chunks = ((24, 1), (24, 8), (24, 24), (1,)) if keepdims else ((24, 1), (24, 8), (24, 24))
    assert result.expr.simplify().chunks == expected_chunks
    assert_eq(result, expected)


@pytest.mark.parametrize("reduction", ["nansum", "nanprod", "nanmin", "nanmax", "nanmean"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sliding_window_nan_reduction_keeps_non_window_chunks(reduction, keepdims):
    data = (1 + (np.arange(96 * 32 * 48, dtype=np.float64) % 5) / 10).reshape(96, 32, 48)
    data[::7, :, :] = np.nan
    data[:80, 0, 0] = np.nan
    x = da.from_array(data, chunks=(24, 24, 24))

    windowed = da.sliding_window_view(x, window_shape=72, axis=0)
    result = getattr(da, reduction)(windowed, axis=-1, keepdims=keepdims)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = getattr(np, reduction)(
            np.lib.stride_tricks.sliding_window_view(data, 72, axis=0), axis=-1, keepdims=keepdims
        )

    expected_chunks = ((24, 1), (24, 8), (24, 24), (1,)) if keepdims else ((24, 1), (24, 8), (24, 24))
    assert result.expr.simplify().chunks == expected_chunks
    assert not _contains_sliding_window_view(result.expr.simplify())
    assert_eq(result, expected)


@pytest.mark.parametrize("reduction", ["var", "std"])
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sliding_window_moment_reduction_keeps_non_window_chunks(reduction, ddof, keepdims):
    data = (1 + (np.arange(96 * 32 * 48, dtype=np.float64) % 13) / 10).reshape(96, 32, 48)
    x = da.from_array(data, chunks=(24, 24, 24))

    windowed = da.sliding_window_view(x, window_shape=72, axis=0)
    result = getattr(windowed, reduction)(axis=-1, ddof=ddof, keepdims=keepdims)
    expected = getattr(np.lib.stride_tricks.sliding_window_view(data, 72, axis=0), reduction)(
        axis=-1, ddof=ddof, keepdims=keepdims
    )

    expected_chunks = ((25,), (24, 8), (24, 24), (1,)) if keepdims else ((25,), (24, 8), (24, 24))
    assert result.expr.simplify().chunks == expected_chunks
    assert not _contains_sliding_window_view(result.expr.simplify())
    assert_eq(result, expected)


@pytest.mark.parametrize("reduction", ["nanvar", "nanstd"])
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sliding_window_nan_moment_reduction_keeps_non_window_chunks(reduction, ddof, keepdims):
    data = (1 + (np.arange(96 * 32 * 48, dtype=np.float64) % 13) / 10).reshape(96, 32, 48)
    data[::7, :, :] = np.nan
    data[:80, 0, 0] = np.nan
    x = da.from_array(data, chunks=(24, 24, 24))

    windowed = da.sliding_window_view(x, window_shape=72, axis=0)
    result = getattr(da, reduction)(windowed, axis=-1, ddof=ddof, keepdims=keepdims)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = getattr(np, reduction)(
            np.lib.stride_tricks.sliding_window_view(data, 72, axis=0), axis=-1, ddof=ddof, keepdims=keepdims
        )

    expected_chunks = ((25,), (24, 8), (24, 24), (1,)) if keepdims else ((25,), (24, 8), (24, 24))
    assert result.expr.simplify().chunks == expected_chunks
    assert not _contains_sliding_window_view(result.expr.simplify())
    assert_eq(result, expected, rtol=1e-7, atol=1e-8)


def test_sliding_window_var_uses_stable_block_algorithm():
    data = (1e9 + (np.arange(96 * 8, dtype=np.float64) % 13) / 10).reshape(96, 8)
    x = da.from_array(data, chunks=(24, 4))

    windowed = da.sliding_window_view(x, window_shape=72, axis=0)
    result = windowed.var(axis=-1)
    expected = np.lib.stride_tricks.sliding_window_view(data, 72, axis=0).var(axis=-1)

    assert result.expr.simplify().chunks == ((25,), (4, 4))
    assert_eq(result, expected, rtol=1e-7, atol=1e-8)


def test_sliding_window_nanvar_uses_stable_block_algorithm():
    data = (1e9 + (np.arange(96 * 8, dtype=np.float64) % 13) / 10).reshape(96, 8)
    data[::7, :] = np.nan
    x = da.from_array(data, chunks=(24, 4))

    windowed = da.sliding_window_view(x, window_shape=72, axis=0)
    result = da.nanvar(windowed, axis=-1)
    expected = np.nanvar(np.lib.stride_tricks.sliding_window_view(data, 72, axis=0), axis=-1)

    assert result.expr.simplify().chunks == ((25,), (4, 4))
    assert not _contains_sliding_window_view(result.expr.simplify())
    assert_eq(result, expected, rtol=5e-7, atol=1e-8)


def test_sliding_window_reduction_avoids_rechunking_left_padding_chunk():
    window = 4
    data = np.arange(10 * 2, dtype=np.int64).reshape(10, 2)
    padding = np.full((window - 1, 2), -1, dtype=data.dtype)
    x = da.concatenate(
        [
            da.from_array(padding, chunks=(window - 1, 2)),
            da.from_array(data, chunks=(10, 2)),
        ],
        axis=0,
    )
    full_data = np.concatenate([padding, data])

    result = da.sliding_window_view(x, window_shape=window, axis=0).sum(axis=-1)
    expected = np.lib.stride_tricks.sliding_window_view(full_data, window, axis=0).sum(axis=-1)

    optimized = result.expr.optimize()
    assert optimized.chunks == ((window - 1, data.shape[0] - (window - 1)), (2,))
    assert not _contains_tasks_rechunk(optimized)
    assert_eq(result, expected)


@pytest.mark.parametrize("reduction", ["var", "nanvar", "std", "nanstd"])
def test_sliding_window_var_explicit_integer_dtype(reduction):
    data = np.arange(24, dtype=np.int64) * 3
    x = da.from_array(data, chunks=8)

    windowed = da.sliding_window_view(x, window_shape=3, axis=0)
    result = getattr(da, reduction)(windowed, axis=-1, dtype="i8")
    variance_func = np.nanvar if reduction.startswith("nan") else np.var
    variance = variance_func(np.lib.stride_tricks.sliding_window_view(data, 3, axis=0), axis=-1, dtype="i8")
    expected = np.sqrt(variance).astype("i8") if reduction.endswith("std") else variance

    assert result.dtype == expected.dtype
    assert not _contains_sliding_window_view(result.expr.simplify())
    assert_eq(result, expected)


def test_sliding_window_reduction_slice_pushdown_preserves_reducer_kind():
    data = (1 + (np.arange(96 * 8, dtype=np.float64) % 13) / 10).reshape(96, 8)
    data[::7, :] = np.nan
    x = da.from_array(data, chunks=(24, 4))

    windowed = da.sliding_window_view(x, window_shape=72, axis=0)
    result = da.nanvar(windowed, axis=-1)[:10]
    expected = np.nanvar(np.lib.stride_tricks.sliding_window_view(data, 72, axis=0), axis=-1)[:10]

    assert result.expr.simplify().sliding_window_reducer == "nanvar"
    assert_eq(result, expected)


@pytest.mark.parametrize("reduction", ["nansum", "nanprod", "nanmin", "nanmax", "nanmean"])
def test_sliding_window_nan_reduction_complex_values(reduction):
    data = np.array(
        [
            1 + 1j,
            np.nan + 2j,
            3 + 3j,
            4 + np.nan * 1j,
            5 + 5j,
            6 + 6j,
            np.nan + np.nan * 1j,
            8 + 8j,
        ],
        dtype="complex128",
    )
    x = da.from_array(data, chunks=4)

    windowed = da.sliding_window_view(x, window_shape=3, axis=0)
    result = getattr(da, reduction)(windowed, axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = getattr(np, reduction)(np.lib.stride_tricks.sliding_window_view(data, 3, axis=0), axis=-1)

    assert not _contains_sliding_window_view(result.expr.simplify())
    assert_eq(result, expected)


def test_sliding_window_nansum_object_dtype_stays_on_general_path():
    data = np.array([1, np.nan, 2, 3, np.nan, 5], dtype=object)
    x = da.from_array(data, chunks=3)

    windowed = da.sliding_window_view(x, window_shape=3, axis=0)
    result = da.nansum(windowed, axis=-1)
    expected = np.nansum(np.lib.stride_tricks.sliding_window_view(data, 3, axis=0), axis=-1)

    assert _contains_sliding_window_view(result.expr.simplify())
    assert_eq(result, expected)


@pytest.mark.parametrize("reduction", ["var", "std"])
@pytest.mark.parametrize("dtype", ["f4", "c8"])
def test_sliding_window_var_complex_explicit_dtype(reduction, dtype):
    data = (np.arange(24, dtype=np.float32) + 1j * np.arange(24, dtype=np.float32)).astype("complex64")
    x = da.from_array(data, chunks=8)

    windowed = da.sliding_window_view(x, window_shape=3, axis=0)
    result = getattr(windowed, reduction)(axis=-1, dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        expected = getattr(np.lib.stride_tricks.sliding_window_view(data, 3, axis=0), reduction)(axis=-1, dtype=dtype)

    assert result.dtype == expected.dtype
    if dtype == "f4":
        assert result.expr.simplify().chunks == ((8, 8, 6),)
        assert not _contains_sliding_window_view(result.expr.simplify())
    else:
        assert _contains_sliding_window_view(result.expr.simplify())
    assert_eq(result, expected)


@pytest.mark.parametrize("reduction", ["nanvar", "nanstd"])
@pytest.mark.parametrize("dtype", ["f4", "c8"])
def test_sliding_window_nanvar_complex_explicit_dtype(reduction, dtype):
    data = (np.arange(24, dtype=np.float32) + 1j * np.arange(24, dtype=np.float32)).astype("complex64")
    data[::5] = np.nan + 0j
    x = da.from_array(data, chunks=8)

    windowed = da.sliding_window_view(x, window_shape=3, axis=0)
    result = getattr(da, reduction)(windowed, axis=-1, dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        expected = getattr(np, reduction)(
            np.lib.stride_tricks.sliding_window_view(data, 3, axis=0), axis=-1, dtype=dtype
        )

    assert result.dtype == expected.dtype
    if dtype == "f4":
        assert result.expr.simplify().chunks == ((8, 8, 6),)
        assert not _contains_sliding_window_view(result.expr.simplify())
    else:
        assert _contains_sliding_window_view(result.expr.simplify())
    assert_eq(result, expected)


def test_sliding_window_var_complex_explicit_complex_dtype_stays_on_moment_path():
    real = 10_000 + (np.arange(24, dtype=np.float32) % 7) / 10
    imag = (np.arange(24, dtype=np.float32) % 5) / 3
    data = (real + 1j * imag).astype("complex64")
    x = da.from_array(data, chunks=8)

    windowed = da.sliding_window_view(x, window_shape=3, axis=0)
    result = windowed.var(axis=-1, dtype="c8")
    expected = np.lib.stride_tricks.sliding_window_view(data, 3, axis=0).var(axis=-1, dtype="c8")

    assert _contains_sliding_window_view(result.expr.simplify())
    assert_eq(result, expected, rtol=1e-5, atol=1e-7)


def test_sliding_window_nanvar_complex_explicit_complex_dtype_stays_on_moment_path():
    real = 10_000 + (np.arange(24, dtype=np.float32) % 7) / 10
    imag = (np.arange(24, dtype=np.float32) % 5) / 3
    data = (real + 1j * imag).astype("complex64")
    data[::5] = np.nan + 0j
    x = da.from_array(data, chunks=8)

    windowed = da.sliding_window_view(x, window_shape=3, axis=0)
    result = da.nanvar(windowed, axis=-1, dtype="c8")
    expected = np.nanvar(np.lib.stride_tricks.sliding_window_view(data, 3, axis=0), axis=-1, dtype="c8")

    assert _contains_sliding_window_view(result.expr.simplify())
    assert_eq(result, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("data", [np.arange(8, dtype=np.float64), np.ones(8, dtype=np.float64)])
def test_sliding_window_var_ddof_equal_window(data):
    x = da.from_array(data, chunks=4)

    windowed = da.sliding_window_view(x, window_shape=3, axis=0)
    result = windowed.var(axis=-1, ddof=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.lib.stride_tricks.sliding_window_view(data, 3, axis=0).var(axis=-1, ddof=3)

    assert_eq(result, expected)


@pytest.mark.parametrize(
    "data, expected",
    [
        # Existing dask moment reductions only replace negative denominators.
        # A zero denominator follows divide-by-zero behavior.
        (
            np.array([np.nan, 1, 2, np.nan, 3, 4, np.nan, 5], dtype=np.float64),
            np.full(6, np.inf),
        ),
        (
            np.array([np.nan, 1, 1, np.nan, 1, 1, np.nan, 1], dtype=np.float64),
            np.full(6, np.nan),
        ),
    ],
)
def test_sliding_window_nanvar_ddof_equal_count(data, expected):
    x = da.from_array(data, chunks=4)

    windowed = da.sliding_window_view(x, window_shape=3, axis=0)
    result = da.nanvar(windowed, axis=-1, ddof=2)

    assert not _contains_sliding_window_view(result.expr.simplify())
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "reduction, axis, expected_chunks",
    [
        ("min", 1, ((20, 20), (9,), (24, 24))),
        ("prod", 2, ((20, 20), (24, 8), (24, 1))),
    ],
)
def test_sliding_window_reduction_keeps_non_leading_non_window_chunks(reduction, axis, expected_chunks):
    data = (1 + (np.arange(40 * 32 * 48, dtype=np.float32) % 5) / 100).reshape(40, 32, 48)
    x = da.from_array(data, chunks=(20, 24, 24))

    windowed = da.sliding_window_view(x, window_shape=24, axis=axis)
    result = getattr(windowed, reduction)(axis=-1)
    expected = getattr(np.lib.stride_tricks.sliding_window_view(data, 24, axis=axis), reduction)(axis=-1)

    assert result.expr.simplify().chunks == expected_chunks
    assert_eq(result, expected)
