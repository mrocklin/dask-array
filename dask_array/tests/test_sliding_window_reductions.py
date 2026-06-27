from __future__ import annotations

import numpy as np
import pytest

import dask_array as da


def _contains_sliding_window_view(expr):
    func = getattr(expr, "func", None)
    if func is np.lib.stride_tricks.sliding_window_view:
        return True
    return any(_contains_sliding_window_view(dep) for dep in expr.dependencies())


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
