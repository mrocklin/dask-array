"""
Dispatch registries for dask_array.

This module provides Dispatch objects for array operations that need to be
dispatched based on array type (numpy, cupy, sparse, etc.).

concatenate_lookup and tensordot_lookup are defined in _core_utils.py but
re-exported here for convenience.
"""

from __future__ import annotations

import numpy as np

from dask.utils import Dispatch

# Re-export from _core_utils for convenience
from dask_array._core_utils import concatenate_lookup, tensordot_lookup

# Dispatch registries for array operations
take_lookup = Dispatch("take")
einsum_lookup = Dispatch("einsum")
empty_lookup = Dispatch("empty")
divide_lookup = Dispatch("divide")
percentile_lookup = Dispatch("percentile")
numel_lookup = Dispatch("numel")
nannumel_lookup = Dispatch("nannumel")


# --- numpy implementations ---


def _divide(x1, x2, out=None, dtype=None):
    """Implementation of numpy.divide that works with dtype kwarg."""
    x = np.divide(x1, x2, out)
    if dtype is not None:
        x = x.astype(dtype)
    return x


def _percentile(a, q, method="linear"):
    """
    Chunk-level percentile calculation.

    Returns (percentile_values, n) tuple where n is the number of elements.
    Used for combining percentiles from multiple chunks.
    """
    from collections.abc import Iterator

    n = len(a)
    if not len(a):
        return None, n
    if isinstance(q, Iterator):
        q = list(q)
    if a.dtype.name == "category":
        result = np.percentile(a.cat.codes, q, method=method)
        import pandas as pd

        return (
            pd.Categorical.from_codes(result, a.dtype.categories, a.dtype.ordered),
            n,
        )
    if type(a.dtype).__name__ == "DatetimeTZDtype":
        import pandas as pd

        if isinstance(a, (pd.Series, pd.Index)):
            a = a.values

    if np.issubdtype(a.dtype, np.datetime64):
        values = a
        if type(a).__name__ in ("Series", "Index"):
            a2 = values.astype("i8")
        else:
            a2 = values.view("i8")
        result = np.percentile(a2, q, method=method).astype(values.dtype)
        if q[0] == 0:
            # https://github.com/dask/dask/issues/6864
            result[0] = min(result[0], values.min())
        return result, n
    if not np.issubdtype(a.dtype, np.number):
        method = "nearest"
    return np.percentile(a, q, method=method), n


def _numel(x, **kwargs):
    """
    A reduction to count the number of elements.

    Returns ndarray result (coerces to numpy).
    """
    import math

    shape = x.shape
    keepdims = kwargs.get("keepdims", False)
    axis = kwargs.get("axis")
    dtype = kwargs.get("dtype", np.float64)

    if axis is None:
        prod = np.prod(shape, dtype=dtype)
        if keepdims is False:
            return prod

        return np.full(shape=(1,) * len(shape), fill_value=prod, dtype=dtype)

    if not isinstance(axis, (tuple, list)):
        axis = [axis]

    prod = math.prod(shape[dim] for dim in axis)
    if keepdims is True:
        new_shape = tuple(
            shape[dim] if dim not in axis else 1 for dim in range(len(shape))
        )
    else:
        new_shape = tuple(shape[dim] for dim in range(len(shape)) if dim not in axis)

    return np.broadcast_to(np.array(prod, dtype=dtype), new_shape)


def _nannumel(x, **kwargs):
    """A reduction to count the number of elements, excluding nans"""
    return np.sum(~(np.isnan(x)), **kwargs)


# --- Register numpy implementations ---

take_lookup.register((object, np.ndarray, np.ma.masked_array), np.take)
einsum_lookup.register((object, np.ndarray), np.einsum)
empty_lookup.register((object, np.ndarray), np.empty)
empty_lookup.register(np.ma.masked_array, np.ma.empty)
divide_lookup.register((object, np.ndarray), _divide)
divide_lookup.register(np.ma.masked_array, np.ma.divide)
percentile_lookup.register(np.ndarray, _percentile)
numel_lookup.register((object, np.ndarray), _numel)
nannumel_lookup.register((object, np.ndarray), _nannumel)


# --- Register masked array numel ---


@numel_lookup.register(np.ma.masked_array)
def _numel_masked(x, **kwargs):
    """Numel implementation for masked arrays."""
    return np.sum(np.ones_like(x), **kwargs)
