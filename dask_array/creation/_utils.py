"""Helper functions for array creation operations.

Copied/adapted from dask.array.creation and dask.array.wrap to reduce
imports from dask.array.* modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from numbers import Number

import numpy as np
from tlz import curry

from dask_array._core_utils import normalize_chunks
from dask.base import tokenize
from dask.utils import funcname


def _parse_wrap_args(func, args, kwargs, shape):
    """Parse arguments for wrap functions (ones, zeros, full, empty).

    Parameters
    ----------
    func : callable
        The numpy function (e.g., np.ones_like)
    args : tuple
        Positional arguments after shape
    kwargs : dict
        Keyword arguments (may include name, chunks, dtype)
    shape : tuple or int
        The desired shape

    Returns
    -------
    dict with keys: shape, dtype, kwargs, chunks, name
    """
    if isinstance(shape, np.ndarray):
        shape = shape.tolist()

    if not isinstance(shape, (tuple, list)):
        shape = (shape,)

    name = kwargs.pop("name", None)
    chunks = kwargs.pop("chunks", "auto")

    dtype = kwargs.pop("dtype", None)
    if dtype is None:
        dtype = func(shape, *args, **kwargs).dtype
    dtype = np.dtype(dtype)

    chunks = normalize_chunks(chunks, shape, dtype=dtype)

    name = name or funcname(func) + "-" + tokenize(
        func, shape, chunks, dtype, args, kwargs
    )

    return {
        "shape": shape,
        "dtype": dtype,
        "kwargs": kwargs,
        "chunks": chunks,
        "name": name,
    }


@curry
def _broadcast_trick_inner(func, shape, meta=(), *args, **kwargs):
    """Inner function for broadcast_trick.

    cupy-specific hack. numpy is happy with hardcoded shape=().
    """
    null_shape = () if shape == () else 1
    return np.broadcast_to(func(meta, *args, shape=null_shape, **kwargs), shape)


def broadcast_trick(func):
    """Provide a decorator to wrap common numpy function with a broadcast trick.

    Dask arrays are currently immutable; thus when we know an array is uniform,
    we can replace the actual data by a single value and have all elements point
    to it, thus reducing the size.

    >>> x = np.broadcast_to(1, (100,100,100))
    >>> x.base.nbytes
    8

    Those array are not only more efficient locally, but dask serialisation is
    aware of the _real_ size of those array and thus can send them around
    efficiently and schedule accordingly.

    Note that those array are read-only and numpy will refuse to assign to them,
    so should be safe.
    """
    inner = _broadcast_trick_inner(func)
    inner.__doc__ = func.__doc__
    inner.__name__ = func.__name__
    return inner


def _get_like_function_shapes_chunks(a, chunks, shape):
    """Helper function for finding shapes and chunks for *_like() array creation functions.

    Parameters
    ----------
    a : dask array
        The input array to get shape/chunks from
    chunks : tuple or None
        Desired chunks (None means use a's chunks)
    shape : tuple or None
        Desired shape (None means use a's shape)

    Returns
    -------
    shape, chunks : tuple, tuple
    """
    if shape is None:
        shape = a.shape
        if chunks is None:
            chunks = a.chunks
    elif chunks is None:
        chunks = "auto"
    return shape, chunks


def expand_pad_value(array, pad_value):
    """Expand pad_value to a per-dimension format.

    Parameters
    ----------
    array : dask array
        The array to be padded (used to get ndim)
    pad_value : various
        The pad value in various formats

    Returns
    -------
    tuple of tuples
        Normalized pad_value as ((before, after), ...) for each dimension
    """
    if isinstance(pad_value, Number) or getattr(pad_value, "ndim", None) == 0:
        pad_value = array.ndim * ((pad_value, pad_value),)
    elif (
        isinstance(pad_value, Sequence)
        and all(isinstance(pw, Number) for pw in pad_value)
        and len(pad_value) == 1
    ):
        pad_value = array.ndim * ((pad_value[0], pad_value[0]),)
    elif (
        isinstance(pad_value, Sequence)
        and len(pad_value) == 2
        and all(isinstance(pw, Number) for pw in pad_value)
    ):
        pad_value = array.ndim * (tuple(pad_value),)
    elif (
        isinstance(pad_value, Sequence)
        and len(pad_value) == array.ndim
        and all(isinstance(pw, Sequence) for pw in pad_value)
        and all((len(pw) == 2) for pw in pad_value)
        and all(all(isinstance(w, Number) for w in pw) for pw in pad_value)
    ):
        pad_value = tuple(tuple(pw) for pw in pad_value)
    elif (
        isinstance(pad_value, Sequence)
        and len(pad_value) == 1
        and isinstance(pad_value[0], Sequence)
        and len(pad_value[0]) == 2
        and all(isinstance(pw, Number) for pw in pad_value[0])
    ):
        pad_value = array.ndim * (tuple(pad_value[0]),)
    else:
        raise TypeError("`pad_value` must be composed of integral typed values.")

    return pad_value


def get_pad_shapes_chunks(array, pad_width, axes, mode):
    """Helper function for finding shapes and chunks of end pads.

    Parameters
    ----------
    array : dask array
        The array to be padded
    pad_width : tuple of tuples
        The pad widths as ((before, after), ...) for each dimension
    axes : tuple of ints
        Which axes to compute pad info for
    mode : str
        The padding mode

    Returns
    -------
    pad_shapes : list of tuples
        Shape for [before, after] pads
    pad_chunks : list of tuples
        Chunks for [before, after] pads
    """
    pad_shapes = [list(array.shape), list(array.shape)]
    pad_chunks = [list(array.chunks), list(array.chunks)]

    for d in axes:
        for i in range(2):
            pad_shapes[i][d] = pad_width[d][i]
            if mode != "constant" or pad_width[d][i] == 0:
                pad_chunks[i][d] = (pad_width[d][i],)
            else:
                pad_chunks[i][d] = normalize_chunks(
                    (max(pad_chunks[i][d]),), (pad_width[d][i],)
                )[0]

    pad_shapes = [tuple(s) for s in pad_shapes]
    pad_chunks = [tuple(c) for c in pad_chunks]

    return pad_shapes, pad_chunks


def linear_ramp_chunk(start, stop, num, dim, step):
    """Helper function to find the linear ramp for a chunk.

    Parameters
    ----------
    start : array
        Starting values (shape has size 1 in dim)
    stop : scalar
        End value for ramp
    num : int
        Number of points in ramp
    dim : int
        Dimension along which to ramp
    step : int
        Direction (1 or -1)

    Returns
    -------
    array with linear ramp values
    """
    num1 = num + 1

    shape = list(start.shape)
    shape[dim] = num
    shape = tuple(shape)

    dtype = np.dtype(start.dtype)

    result = np.empty_like(start, shape=shape, dtype=dtype)
    for i in np.ndindex(start.shape):
        j = list(i)
        j[dim] = slice(None)
        j = tuple(j)

        result[j] = np.linspace(start[i], stop, num1, dtype=dtype)[1:][::step]

    return result


def wrapped_pad_func(array, pad_func, iaxis_pad_width, iaxis, pad_func_kwargs):
    """Wrapper to apply a user-defined pad function along an axis.

    Parameters
    ----------
    array : ndarray
        The input array chunk
    pad_func : callable
        User-defined padding function
    iaxis_pad_width : tuple
        (before, after) pad widths for this axis
    iaxis : int
        The axis index
    pad_func_kwargs : dict
        Keyword arguments to pass to pad_func

    Returns
    -------
    array with padding applied
    """
    result = np.empty_like(array)
    for i in np.ndindex(array.shape[:iaxis] + array.shape[iaxis + 1 :]):
        i = i[:iaxis] + (slice(None),) + i[iaxis:]
        result[i] = pad_func(array[i], iaxis_pad_width, iaxis, pad_func_kwargs)

    return result


def to_backend(x, backend: str | None = None, **kwargs):
    """Move an Array collection to a new backend.

    Parameters
    ----------
    x : Array
        The input Array collection.
    backend : str, Optional
        The name of the new backend to move to. The default
        is the current "array.backend" configuration.

    Returns
    -------
    Array
        A new Array collection with the backend specified
        by ``backend``.
    """
    from dask_array._backends_array import array_creation_dispatch

    # Get desired backend
    backend = backend or array_creation_dispatch.backend
    # Check that "backend" has a registered entrypoint
    backend_entrypoint = array_creation_dispatch.dispatch(backend)
    # Call `ArrayBackendEntrypoint.to_backend`
    return backend_entrypoint.to_backend(x, **kwargs)
