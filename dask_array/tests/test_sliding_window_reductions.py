from __future__ import annotations

import numpy as np
import pytest

import dask_array as da


def _contains_sliding_window_view(expr):
    func = getattr(expr, "func", None)
    if func is np.lib.stride_tricks.sliding_window_view:
        return True
    return any(_contains_sliding_window_view(dep) for dep in expr.dependencies())


@pytest.mark.parametrize("reduction", ["max", "sum", "mean"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sliding_window_reduction_over_window_axis_avoids_window_block(reduction, keepdims):
    data = np.arange(80 * 4 * 5, dtype=np.float32).reshape(80, 4, 5)
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
