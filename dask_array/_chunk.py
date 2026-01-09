"""A set of NumPy functions to apply per chunk"""
from __future__ import annotations

from collections.abc import Container, Iterable, Sequence
from functools import wraps

import numpy as np


def keepdims_wrapper(a_callable):
    """
    A wrapper for functions that don't provide keepdims to ensure that they do.
    """

    @wraps(a_callable)
    def keepdims_wrapped_callable(x, axis=None, keepdims=None, *args, **kwargs):
        r = a_callable(x, *args, axis=axis, **kwargs)

        if not keepdims:
            return r

        axes = axis

        if axes is None:
            axes = range(x.ndim)

        if not isinstance(axes, (Container, Iterable, Sequence)):
            axes = [axes]

        r_slice = tuple()
        for each_axis in range(x.ndim):
            if each_axis in axes:
                r_slice += (None,)
            else:
                r_slice += (slice(None),)

        r = r[r_slice]

        return r

    return keepdims_wrapped_callable


# Wrap NumPy functions to ensure they provide keepdims.
sum = np.sum
prod = np.prod
min = np.min
max = np.max
argmin = keepdims_wrapper(np.argmin)
nanargmin = keepdims_wrapper(np.nanargmin)
argmax = keepdims_wrapper(np.argmax)
nanargmax = keepdims_wrapper(np.nanargmax)
any = np.any
all = np.all
nansum = np.nansum
nanprod = np.nanprod

nancumprod = np.nancumprod
nancumsum = np.nancumsum

nanmin = np.nanmin
nanmax = np.nanmax
mean = np.mean
nanmean = np.nanmean

var = np.var
nanvar = np.nanvar

std = np.std
nanstd = np.nanstd


def topk(a, k, axis, keepdims):
    """Chunk and combine function of topk

    Extract the k largest elements from a on the given axis.
    If k is negative, extract the -k smallest elements instead.
    Note that, unlike in the parent function, the returned elements
    are not sorted internally.
    """
    assert keepdims is True
    axis = axis[0]
    if abs(k) >= a.shape[axis]:
        return a

    a = np.partition(a, -k, axis=axis)
    k_slice = slice(-k, None) if k > 0 else slice(-k)
    return a[tuple(k_slice if i == axis else slice(None) for i in range(a.ndim))]


def topk_aggregate(a, k, axis, keepdims):
    """Final aggregation function of topk

    Invoke topk one final time and then sort the results internally.
    """
    assert keepdims is True
    a = topk(a, k, axis, keepdims)
    axis = axis[0]
    a = np.sort(a, axis=axis)
    if k < 0:
        return a
    return a[
        tuple(
            slice(None, None, -1) if i == axis else slice(None) for i in range(a.ndim)
        )
    ]


def argtopk_preprocess(a, idx):
    """Preparatory step for argtopk

    Put data together with its original indices in a tuple.
    """
    return a, idx


def argtopk(a_plus_idx, k, axis, keepdims):
    """Chunk and combine function of argtopk

    Extract the indices of the k largest elements from a on the given axis.
    If k is negative, extract the indices of the -k smallest elements instead.
    Note that, unlike in the parent function, the returned elements
    are not sorted internally.
    """
    assert keepdims is True
    axis = axis[0]

    a, idx = a_plus_idx

    if abs(k) >= a.shape[axis]:
        return a, idx

    idx2 = np.argpartition(a, -k, axis=axis)
    k_slice = slice(-k, None) if k > 0 else slice(-k)
    idx2 = idx2[tuple(k_slice if i == axis else slice(None) for i in range(a.ndim))]

    return np.take_along_axis(a, idx2, axis), np.take_along_axis(idx, idx2, axis)


def argtopk_aggregate(a_plus_idx, k, axis, keepdims):
    """Final aggregation function of argtopk

    Invoke argtopk one final time, sort the results internally, and drop the data.
    """
    assert keepdims is True
    axis = axis[0]

    a, idx = argtopk(a_plus_idx, k, axis, (axis,), keepdims)
    idx2 = np.argsort(a, axis=axis)

    idx = np.take_along_axis(idx, idx2, axis)
    if k < 0:
        return idx
    return idx[
        tuple(
            slice(None, None, -1) if i == axis else slice(None) for i in range(idx.ndim)
        )
    ]
