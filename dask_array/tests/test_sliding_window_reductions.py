from __future__ import annotations

import warnings

import numpy as np
import pytest

import dask_array as da
from dask_array._rechunk import TasksRechunk


def _contains_sliding_window_view(expr):
    func = getattr(expr, "func", None)
    if func is np.lib.stride_tricks.sliding_window_view:
        return True
    return any(_contains_sliding_window_view(dep) for dep in expr.dependencies())


def _contains_tasks_rechunk(expr):
    if isinstance(expr, TasksRechunk):
        return True
    return any(_contains_tasks_rechunk(dep) for dep in expr.dependencies())


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
    expected_chunks = ((32, 25), (4,), (5,), (1,)) if keepdims else ((32, 25), (4,), (5,))
    assert result.chunks == expected_chunks
    np.testing.assert_allclose(result.compute(), expected)
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

    expected_chunks = ((25,), (24, 8), (24, 24), (1,)) if keepdims else ((25,), (24, 8), (24, 24))
    assert result.expr.simplify().chunks == expected_chunks
    np.testing.assert_allclose(result.compute(), expected)


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

    expected_chunks = ((25,), (24, 8), (24, 24), (1,)) if keepdims else ((25,), (24, 8), (24, 24))
    assert result.expr.simplify().chunks == expected_chunks
    np.testing.assert_array_equal(result.compute(), expected)


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

    expected_chunks = ((25,), (24, 8), (24, 24), (1,)) if keepdims else ((25,), (24, 8), (24, 24))
    assert result.expr.simplify().chunks == expected_chunks
    assert not _contains_sliding_window_view(result.expr.simplify())
    np.testing.assert_allclose(result.compute(), expected, equal_nan=True)


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
    np.testing.assert_allclose(result.compute(), expected)


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
    np.testing.assert_allclose(result.compute(), expected, rtol=1e-7, atol=1e-8, equal_nan=True)


def test_sliding_window_var_uses_stable_block_algorithm():
    data = (1e9 + (np.arange(96 * 8, dtype=np.float64) % 13) / 10).reshape(96, 8)
    x = da.from_array(data, chunks=(24, 4))

    windowed = da.sliding_window_view(x, window_shape=72, axis=0)
    result = windowed.var(axis=-1)
    expected = np.lib.stride_tricks.sliding_window_view(data, 72, axis=0).var(axis=-1)

    assert result.expr.simplify().chunks == ((25,), (4, 4))
    np.testing.assert_allclose(result.compute(), expected, rtol=1e-7, atol=1e-8)


def test_sliding_window_nanvar_uses_stable_block_algorithm():
    data = (1e9 + (np.arange(96 * 8, dtype=np.float64) % 13) / 10).reshape(96, 8)
    data[::7, :] = np.nan
    x = da.from_array(data, chunks=(24, 4))

    windowed = da.sliding_window_view(x, window_shape=72, axis=0)
    result = da.nanvar(windowed, axis=-1)
    expected = np.nanvar(np.lib.stride_tricks.sliding_window_view(data, 72, axis=0), axis=-1)

    assert result.expr.simplify().chunks == ((25,), (4, 4))
    assert not _contains_sliding_window_view(result.expr.simplify())
    np.testing.assert_allclose(result.compute(), expected, rtol=5e-7, atol=1e-8)


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
    np.testing.assert_array_equal(result.compute(), expected)


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
    np.testing.assert_array_equal(result.compute(), expected)


def test_sliding_window_reduction_slice_pushdown_preserves_reducer_kind():
    data = (1 + (np.arange(96 * 8, dtype=np.float64) % 13) / 10).reshape(96, 8)
    data[::7, :] = np.nan
    x = da.from_array(data, chunks=(24, 4))

    windowed = da.sliding_window_view(x, window_shape=72, axis=0)
    result = da.nanvar(windowed, axis=-1)[:10]
    expected = np.nanvar(np.lib.stride_tricks.sliding_window_view(data, 72, axis=0), axis=-1)[:10]

    assert result.expr.simplify().sliding_window_reducer == "nanvar"
    np.testing.assert_allclose(result.compute(), expected, equal_nan=True)


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
    np.testing.assert_allclose(result.compute(), expected, equal_nan=True)


def test_sliding_window_nansum_object_dtype_stays_on_general_path():
    data = np.array([1, np.nan, 2, 3, np.nan, 5], dtype=object)
    x = da.from_array(data, chunks=3)

    windowed = da.sliding_window_view(x, window_shape=3, axis=0)
    result = da.nansum(windowed, axis=-1)
    expected = np.nansum(np.lib.stride_tricks.sliding_window_view(data, 3, axis=0), axis=-1)

    assert _contains_sliding_window_view(result.expr.simplify())
    np.testing.assert_array_equal(result.compute(), expected)


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
    np.testing.assert_allclose(result.compute(), expected)


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
    np.testing.assert_allclose(result.compute(), expected, equal_nan=True)


def test_sliding_window_var_complex_explicit_complex_dtype_stays_on_moment_path():
    real = 10_000 + (np.arange(24, dtype=np.float32) % 7) / 10
    imag = (np.arange(24, dtype=np.float32) % 5) / 3
    data = (real + 1j * imag).astype("complex64")
    x = da.from_array(data, chunks=8)

    windowed = da.sliding_window_view(x, window_shape=3, axis=0)
    result = windowed.var(axis=-1, dtype="c8")
    expected = np.lib.stride_tricks.sliding_window_view(data, 3, axis=0).var(axis=-1, dtype="c8")

    assert _contains_sliding_window_view(result.expr.simplify())
    np.testing.assert_allclose(result.compute(), expected, rtol=1e-5, atol=1e-7)


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
    np.testing.assert_allclose(result.compute(), expected, rtol=1e-5, atol=1e-7, equal_nan=True)


@pytest.mark.parametrize("data", [np.arange(8, dtype=np.float64), np.ones(8, dtype=np.float64)])
def test_sliding_window_var_ddof_equal_window(data):
    x = da.from_array(data, chunks=4)

    windowed = da.sliding_window_view(x, window_shape=3, axis=0)
    result = windowed.var(axis=-1, ddof=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.lib.stride_tricks.sliding_window_view(data, 3, axis=0).var(axis=-1, ddof=3)

    np.testing.assert_allclose(result.compute(), expected, equal_nan=True)


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
    np.testing.assert_allclose(result.compute(), expected, equal_nan=True)


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
    np.testing.assert_allclose(result.compute(), expected)
