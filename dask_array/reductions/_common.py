"""Common reduction functions using the expression-based reduction framework."""
from __future__ import annotations

import builtins
import math
import warnings
from functools import partial
from numbers import Integral, Number

import numpy as np
from dask.utils import deepmap, derived_from

import builtins

from dask.array.core import _concatenate2
from dask.array.dispatch import divide_lookup, numel_lookup, nannumel_lookup
from dask.array.utils import array_safe, asarray_safe, meta_from_array
from dask_array import _chunk as chunk
from dask_array.reductions._reduction import reduction
from dask_array.reductions._arg_reduction import arg_reduction


def divide(a, b, dtype=None):
    """Safe divide handling different array types."""
    key = lambda x: getattr(x, "__array_priority__", float("-inf"))
    f = divide_lookup.dispatch(type(builtins.max(a, b, key=key)))
    return f(a, b, dtype=dtype)


def numel(x, **kwargs):
    """Count number of elements."""
    return numel_lookup(x, **kwargs)


def nannumel(x, **kwargs):
    """Count number of non-NaN elements."""
    return nannumel_lookup(x, **kwargs)


# Simple reductions
@derived_from(np)
def sum(a, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
    if dtype is None:
        dtype = getattr(np.zeros(1, dtype=a.dtype).sum(), "dtype", object)
    return reduction(
        a,
        chunk.sum,
        chunk.sum,
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        out=out,
    )


@derived_from(np)
def prod(a, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
    if dtype is not None:
        dt = dtype
    else:
        dt = getattr(np.ones((1,), dtype=a.dtype).prod(), "dtype", object)
    return reduction(
        a,
        chunk.prod,
        chunk.prod,
        axis=axis,
        keepdims=keepdims,
        dtype=dt,
        split_every=split_every,
        out=out,
    )


def chunk_min(x, axis=None, keepdims=None):
    """Version of np.min which ignores size 0 arrays"""
    if x.size == 0:
        return array_safe([], x, ndmin=x.ndim, dtype=x.dtype)
    else:
        return np.min(x, axis=axis, keepdims=keepdims)


def chunk_max(x, axis=None, keepdims=None):
    """Version of np.max which ignores size 0 arrays"""
    if x.size == 0:
        return array_safe([], x, ndmin=x.ndim, dtype=x.dtype)
    else:
        return np.max(x, axis=axis, keepdims=keepdims)


@derived_from(np)
def min(a, axis=None, keepdims=False, split_every=None, out=None):
    return reduction(
        a,
        chunk_min,
        chunk.min,
        combine=chunk_min,
        axis=axis,
        keepdims=keepdims,
        dtype=a.dtype,
        split_every=split_every,
        out=out,
    )


@derived_from(np)
def max(a, axis=None, keepdims=False, split_every=None, out=None):
    return reduction(
        a,
        chunk_max,
        chunk.max,
        combine=chunk_max,
        axis=axis,
        keepdims=keepdims,
        dtype=a.dtype,
        split_every=split_every,
        out=out,
    )


@derived_from(np)
def any(a, axis=None, keepdims=False, split_every=None, out=None):
    return reduction(
        a,
        chunk.any,
        chunk.any,
        axis=axis,
        keepdims=keepdims,
        dtype="bool",
        split_every=split_every,
        out=out,
    )


@derived_from(np)
def all(a, axis=None, keepdims=False, split_every=None, out=None):
    return reduction(
        a,
        chunk.all,
        chunk.all,
        axis=axis,
        keepdims=keepdims,
        dtype="bool",
        split_every=split_every,
        out=out,
    )


# Nan-aware simple reductions
@derived_from(np)
def nansum(a, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
    if dtype is not None:
        dt = dtype
    else:
        dt = getattr(chunk.nansum(np.ones((1,), dtype=a.dtype)), "dtype", object)
    return reduction(
        a,
        chunk.nansum,
        chunk.sum,
        axis=axis,
        keepdims=keepdims,
        dtype=dt,
        split_every=split_every,
        out=out,
    )


@derived_from(np)
def nanprod(a, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
    if dtype is not None:
        dt = dtype
    else:
        dt = getattr(chunk.nansum(np.ones((1,), dtype=a.dtype)), "dtype", object)
    return reduction(
        a,
        chunk.nanprod,
        chunk.prod,
        axis=axis,
        keepdims=keepdims,
        dtype=dt,
        split_every=split_every,
        out=out,
    )


def _nanmin_skip(x_chunk, axis, keepdims):
    if x_chunk.size > 0:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "All-NaN slice encountered", RuntimeWarning
            )
            return np.nanmin(x_chunk, axis=axis, keepdims=keepdims)
    else:
        return asarray_safe(
            np.array([], dtype=x_chunk.dtype), like=meta_from_array(x_chunk)
        )


def _nanmax_skip(x_chunk, axis, keepdims):
    if x_chunk.size > 0:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "All-NaN slice encountered", RuntimeWarning
            )
            return np.nanmax(x_chunk, axis=axis, keepdims=keepdims)
    else:
        return asarray_safe(
            np.array([], dtype=x_chunk.dtype), like=meta_from_array(x_chunk)
        )


@derived_from(np)
def nanmin(a, axis=None, keepdims=False, split_every=None, out=None):
    if np.isnan(a.size):
        from dask.array.core import unknown_chunk_message
        raise ValueError(f"Arrays chunk sizes are unknown. {unknown_chunk_message}")
    if a.size == 0:
        raise ValueError(
            "zero-size array to reduction operation fmin which has no identity"
        )
    return reduction(
        a,
        _nanmin_skip,
        _nanmin_skip,
        axis=axis,
        keepdims=keepdims,
        dtype=a.dtype,
        split_every=split_every,
        out=out,
    )


@derived_from(np)
def nanmax(a, axis=None, keepdims=False, split_every=None, out=None):
    if np.isnan(a.size):
        from dask.array.core import unknown_chunk_message
        raise ValueError(f"Arrays chunk sizes are unknown. {unknown_chunk_message}")
    if a.size == 0:
        raise ValueError(
            "zero-size array to reduction operation fmax which has no identity"
        )
    return reduction(
        a,
        _nanmax_skip,
        _nanmax_skip,
        axis=axis,
        keepdims=keepdims,
        dtype=a.dtype,
        split_every=split_every,
        out=out,
    )


# Mean implementation
def mean_chunk(
    x, sum=chunk.sum, numel=numel, dtype="f8", computing_meta=False, **kwargs
):
    if computing_meta:
        return x
    n = numel(x, dtype=dtype, **kwargs)
    total = sum(x, dtype=dtype, **kwargs)
    return {"n": n, "total": total}


def mean_combine(
    pairs,
    sum=chunk.sum,
    numel=numel,
    dtype="f8",
    axis=None,
    computing_meta=False,
    **kwargs,
):
    if not isinstance(pairs, list):
        pairs = [pairs]

    ns = deepmap(lambda pair: pair["n"], pairs) if not computing_meta else pairs
    n = _concatenate2(ns, axes=axis).sum(axis=axis, **kwargs)

    if computing_meta:
        return n

    totals = deepmap(lambda pair: pair["total"], pairs)
    total = _concatenate2(totals, axes=axis).sum(axis=axis, **kwargs)

    return {"n": n, "total": total}


def mean_agg(pairs, dtype="f8", axis=None, computing_meta=False, **kwargs):
    ns = deepmap(lambda pair: pair["n"], pairs) if not computing_meta else pairs
    n = _concatenate2(ns, axes=axis)
    n = np.sum(n, axis=axis, dtype=dtype, **kwargs)

    if computing_meta:
        return n

    totals = deepmap(lambda pair: pair["total"], pairs)
    total = _concatenate2(totals, axes=axis).sum(axis=axis, dtype=dtype, **kwargs)

    with np.errstate(divide="ignore", invalid="ignore"):
        return divide(total, n, dtype=dtype)


@derived_from(np)
def mean(a, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
    if dtype is not None:
        dt = dtype
    elif a.dtype == object:
        dt = object
    else:
        dt = getattr(np.mean(np.zeros(shape=(1,), dtype=a.dtype)), "dtype", object)
    return reduction(
        a,
        mean_chunk,
        mean_agg,
        axis=axis,
        keepdims=keepdims,
        dtype=dt,
        split_every=split_every,
        combine=mean_combine,
        out=out,
        concatenate=False,
    )


@derived_from(np)
def nanmean(a, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
    if dtype is not None:
        dt = dtype
    else:
        dt = getattr(np.mean(np.ones(shape=(1,), dtype=a.dtype)), "dtype", object)
    return reduction(
        a,
        partial(mean_chunk, sum=chunk.nansum, numel=nannumel),
        mean_agg,
        axis=axis,
        keepdims=keepdims,
        dtype=dt,
        split_every=split_every,
        out=out,
        concatenate=False,
        combine=partial(mean_combine, sum=chunk.nansum, numel=nannumel),
    )


# Moment/variance/std implementation
def moment_chunk(
    A,
    order=2,
    sum=chunk.sum,
    numel=numel,
    dtype="f8",
    computing_meta=False,
    implicit_complex_dtype=False,
    **kwargs,
):
    if computing_meta:
        return A
    n = numel(A, **kwargs)

    n = n.astype(np.int64)
    if implicit_complex_dtype:
        total = sum(A, **kwargs)
    else:
        total = sum(A, dtype=dtype, **kwargs)

    with np.errstate(divide="ignore", invalid="ignore"):
        u = total / n
    d = A - u
    if np.issubdtype(A.dtype, np.complexfloating):
        d = np.abs(d)
    xs = [sum(d**i, dtype=dtype, **kwargs) for i in range(2, order + 1)]
    M = np.stack(xs, axis=-1)
    return {"total": total, "n": n, "M": M}


def _moment_helper(Ms, ns, inner_term, order, sum, axis, kwargs):
    M = Ms[..., order - 2].sum(axis=axis, **kwargs) + sum(
        ns * inner_term**order, axis=axis, **kwargs
    )
    for k in range(1, order - 1):
        coeff = math.factorial(order) / (math.factorial(k) * math.factorial(order - k))
        M += coeff * sum(Ms[..., order - k - 2] * inner_term**k, axis=axis, **kwargs)
    return M


def moment_combine(
    pairs,
    order=2,
    ddof=0,
    dtype="f8",
    sum=np.sum,
    axis=None,
    computing_meta=False,
    **kwargs,
):
    if not isinstance(pairs, list):
        pairs = [pairs]

    kwargs["dtype"] = None
    kwargs["keepdims"] = True

    ns = deepmap(lambda pair: pair["n"], pairs) if not computing_meta else pairs
    ns = _concatenate2(ns, axes=axis)
    n = ns.sum(axis=axis, **kwargs)

    if computing_meta:
        return n

    totals = _concatenate2(deepmap(lambda pair: pair["total"], pairs), axes=axis)
    Ms = _concatenate2(deepmap(lambda pair: pair["M"], pairs), axes=axis)

    total = totals.sum(axis=axis, **kwargs)

    with np.errstate(divide="ignore", invalid="ignore"):
        if np.issubdtype(total.dtype, np.complexfloating):
            mu = divide(total, n)
            inner_term = np.abs(divide(totals, ns) - mu)
        else:
            mu = divide(total, n, dtype=dtype)
            inner_term = divide(totals, ns, dtype=dtype) - mu

    xs = [
        _moment_helper(Ms, ns, inner_term, o, sum, axis, kwargs)
        for o in range(2, order + 1)
    ]
    M = np.stack(xs, axis=-1)
    return {"total": total, "n": n, "M": M}


def moment_agg(
    pairs,
    order=2,
    ddof=0,
    dtype="f8",
    sum=np.sum,
    axis=None,
    computing_meta=False,
    **kwargs,
):
    if not isinstance(pairs, list):
        pairs = [pairs]

    kwargs["dtype"] = dtype
    # To properly handle ndarrays, the original dimensions need to be kept for
    # part of the calculation.
    keepdim_kw = kwargs.copy()
    keepdim_kw["keepdims"] = True
    keepdim_kw["dtype"] = None

    ns = deepmap(lambda pair: pair["n"], pairs) if not computing_meta else pairs
    ns = _concatenate2(ns, axes=axis)
    n = ns.sum(axis=axis, **keepdim_kw)

    if computing_meta:
        return n

    totals = _concatenate2(deepmap(lambda pair: pair["total"], pairs), axes=axis)
    Ms = _concatenate2(deepmap(lambda pair: pair["M"], pairs), axes=axis)

    mu = divide(totals.sum(axis=axis, **keepdim_kw), n)

    with np.errstate(divide="ignore", invalid="ignore"):
        if np.issubdtype(totals.dtype, np.complexfloating):
            inner_term = np.abs(divide(totals, ns) - mu)
        else:
            inner_term = divide(totals, ns, dtype=dtype) - mu
    inner_term = np.where(ns == 0, 0, inner_term)
    M = _moment_helper(Ms, ns, inner_term, order, sum, axis, kwargs)

    denominator = n.sum(axis=axis, **kwargs) - ddof

    # taking care of the edge case with empty or all-nans array with ddof > 0
    if isinstance(denominator, Number):
        if denominator < 0:
            denominator = np.nan
    elif denominator is not np.ma.masked:
        denominator[denominator < 0] = np.nan

    return divide(M, denominator, dtype=dtype)


def moment(
    a, order, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None
):
    """Calculate the nth centralized moment.

    Parameters
    ----------
    a : Array
        Data over which to compute moment
    order : int
        Order of the moment that is returned, must be >= 2.
    axis : int, optional
        Axis along which the central moment is computed. The default is to
        compute the moment of the flattened array.
    dtype : data-type, optional
        Type to use in computing the moment. For arrays of integer type the
        default is float64; for arrays of float types it is the same as the
        array type.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        N - ddof, where N represents the number of elements. By default
        ddof is zero.

    Returns
    -------
    moment : Array
    """
    if not isinstance(order, Integral) or order < 0:
        raise ValueError("Order must be an integer >= 0")

    if order < 2:
        from dask_array.creation import ones, zeros
        reduced = a.sum(axis=axis)  # get reduced shape and chunks
        if order == 0:
            # When order equals 0, the result is 1, by definition.
            return ones(
                reduced.shape, chunks=reduced.chunks, dtype="f8", meta=reduced._meta
            )
        # By definition the first order about the mean is 0.
        return zeros(
            reduced.shape, chunks=reduced.chunks, dtype="f8", meta=reduced._meta
        )

    if dtype is not None:
        dt = dtype
    else:
        dt = getattr(np.var(np.ones(shape=(1,), dtype=a.dtype)), "dtype", object)

    implicit_complex_dtype = dtype is None and np.iscomplexobj(a)

    return reduction(
        a,
        partial(
            moment_chunk, order=order, implicit_complex_dtype=implicit_complex_dtype
        ),
        partial(moment_agg, order=order, ddof=ddof),
        axis=axis,
        keepdims=keepdims,
        dtype=dt,
        split_every=split_every,
        out=out,
        concatenate=False,
        combine=partial(moment_combine, order=order),
    )


@derived_from(np)
def var(a, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None):
    if dtype is not None:
        dt = dtype
    else:
        dt = getattr(np.var(np.ones(shape=(1,), dtype=a.dtype)), "dtype", object)

    implicit_complex_dtype = dtype is None and np.iscomplexobj(a)

    return reduction(
        a,
        partial(moment_chunk, implicit_complex_dtype=implicit_complex_dtype),
        partial(moment_agg, ddof=ddof),
        axis=axis,
        keepdims=keepdims,
        dtype=dt,
        split_every=split_every,
        combine=moment_combine,
        name="var",
        out=out,
        concatenate=False,
    )


@derived_from(np)
def nanvar(
    a, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None
):
    if dtype is not None:
        dt = dtype
    else:
        dt = getattr(np.var(np.ones(shape=(1,), dtype=a.dtype)), "dtype", object)

    implicit_complex_dtype = dtype is None and np.iscomplexobj(a)

    return reduction(
        a,
        partial(
            moment_chunk,
            sum=chunk.nansum,
            numel=nannumel,
            implicit_complex_dtype=implicit_complex_dtype,
        ),
        partial(moment_agg, sum=np.sum, ddof=ddof),
        axis=axis,
        keepdims=keepdims,
        dtype=dt,
        split_every=split_every,
        combine=partial(moment_combine, sum=np.nansum),
        out=out,
        concatenate=False,
    )


def _sqrt(a):
    if isinstance(a, np.ma.masked_array) and not a.shape and a.mask.all():
        return np.ma.masked
    return np.sqrt(a)


def safe_sqrt(a):
    """A version of sqrt that properly handles scalar masked arrays."""
    if hasattr(a, "_elemwise"):
        return a._elemwise(_sqrt, a)
    return _sqrt(a)


@derived_from(np)
def std(a, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None):
    result = safe_sqrt(
        var(
            a,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            ddof=ddof,
            split_every=split_every,
            out=out,
        )
    )
    if dtype and dtype != result.dtype:
        result = result.astype(dtype)
    return result


@derived_from(np)
def nanstd(
    a, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None
):
    result = safe_sqrt(
        nanvar(
            a,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            ddof=ddof,
            split_every=split_every,
            out=out,
        )
    )
    if dtype and dtype != result.dtype:
        result = result.astype(dtype)
    return result


# Arg reductions helpers
def _arg_combine(data, axis, argfunc, keepdims=False):
    """Merge intermediate results from ``arg_*`` functions"""
    if isinstance(data, dict):
        # Array type doesn't support structured arrays (e.g., CuPy),
        # therefore `data` is stored in a `dict`.
        assert data["vals"].ndim == data["arg"].ndim
        axis = (
            None
            if len(axis) == data["vals"].ndim or data["vals"].ndim == 1
            else axis[0]
        )
    else:
        axis = None if len(axis) == data.ndim or data.ndim == 1 else axis[0]

    vals = data["vals"]
    arg = data["arg"]
    if axis is None:
        local_args = argfunc(vals, axis=axis, keepdims=keepdims)
        vals = vals.ravel()[local_args]
        arg = arg.ravel()[local_args]
    else:
        local_args = argfunc(vals, axis=axis)
        inds = list(np.ogrid[tuple(map(slice, local_args.shape))])
        inds.insert(axis, local_args)
        inds = tuple(inds)
        vals = vals[inds]
        arg = arg[inds]
        if keepdims:
            vals = np.expand_dims(vals, axis)
            arg = np.expand_dims(arg, axis)
    return arg, vals


def arg_chunk(func, argfunc, x, axis, offset_info):
    arg_axis = None if len(axis) == x.ndim or x.ndim == 1 else axis[0]
    vals = func(x, axis=arg_axis, keepdims=True)
    arg = argfunc(x, axis=arg_axis, keepdims=True)
    if x.ndim > 0:
        if arg_axis is None:
            offset, total_shape = offset_info
            ind = np.unravel_index(arg.ravel()[0], x.shape)
            total_ind = tuple(o + i for (o, i) in zip(offset, ind))
            arg[:] = np.ravel_multi_index(total_ind, total_shape)
        else:
            arg += offset_info

    if isinstance(vals, np.ma.masked_array):
        if "min" in argfunc.__name__:
            fill_value = np.ma.minimum_fill_value(vals)
        else:
            fill_value = np.ma.maximum_fill_value(vals)
        vals = np.ma.filled(vals, fill_value)

    try:
        result = np.empty_like(
            vals, shape=vals.shape, dtype=[("vals", vals.dtype), ("arg", arg.dtype)]
        )
    except TypeError:
        # Array type doesn't support structured arrays (e.g., CuPy)
        result = dict()

    result["vals"] = vals
    result["arg"] = arg
    return result


def arg_combine(argfunc, data, axis=None, **kwargs):
    arg, vals = _arg_combine(data, axis, argfunc, keepdims=True)

    try:
        result = np.empty_like(
            vals, shape=vals.shape, dtype=[("vals", vals.dtype), ("arg", arg.dtype)]
        )
    except TypeError:
        # Array type doesn't support structured arrays (e.g., CuPy).
        result = dict()

    result["vals"] = vals
    result["arg"] = arg
    return result


def arg_agg(argfunc, data, axis=None, keepdims=False, **kwargs):
    return _arg_combine(data, axis, argfunc, keepdims=keepdims)[0]


def nanarg_agg(argfunc, data, axis=None, keepdims=False, **kwargs):
    arg, vals = _arg_combine(data, axis, argfunc, keepdims=keepdims)
    if np.any(np.isnan(vals)):
        raise ValueError("All NaN slice encountered")
    return arg


def _nanargmin(x, axis, **kwargs):
    try:
        return chunk.nanargmin(x, axis, **kwargs)
    except ValueError:
        return chunk.nanargmin(np.where(np.isnan(x), np.inf, x), axis, **kwargs)


def _nanargmax(x, axis, **kwargs):
    try:
        return chunk.nanargmax(x, axis, **kwargs)
    except ValueError:
        return chunk.nanargmax(np.where(np.isnan(x), -np.inf, x), axis, **kwargs)


@derived_from(np)
def argmax(a, axis=None, keepdims=False, split_every=None, out=None):
    return arg_reduction(
        a,
        partial(arg_chunk, chunk.max, chunk.argmax),
        partial(arg_combine, chunk.argmax),
        partial(arg_agg, chunk.argmax),
        axis=axis,
        keepdims=keepdims,
        split_every=split_every,
        out=out,
    )


@derived_from(np)
def argmin(a, axis=None, keepdims=False, split_every=None, out=None):
    return arg_reduction(
        a,
        partial(arg_chunk, chunk.min, chunk.argmin),
        partial(arg_combine, chunk.argmin),
        partial(arg_agg, chunk.argmin),
        axis=axis,
        keepdims=keepdims,
        split_every=split_every,
        out=out,
    )


@derived_from(np)
def nanargmax(a, axis=None, keepdims=False, split_every=None, out=None):
    return arg_reduction(
        a,
        partial(arg_chunk, chunk.nanmax, _nanargmax),
        partial(arg_combine, _nanargmax),
        partial(nanarg_agg, _nanargmax),
        axis=axis,
        keepdims=keepdims,
        split_every=split_every,
        out=out,
    )


@derived_from(np)
def nanargmin(a, axis=None, keepdims=False, split_every=None, out=None):
    return arg_reduction(
        a,
        partial(arg_chunk, chunk.nanmin, _nanargmin),
        partial(arg_combine, _nanargmin),
        partial(nanarg_agg, _nanargmin),
        axis=axis,
        keepdims=keepdims,
        split_every=split_every,
        out=out,
    )


# Median and quantile functions
from collections.abc import Iterable
from functools import reduce
from operator import mul

from dask.array.core import handle_out

try:
    import numbagg
except ImportError:
    numbagg = None


@derived_from(np)
def median(a, axis=None, keepdims=False, out=None):
    """
    This works by automatically chunking the reduced axes to a single chunk if necessary
    and then calling ``numpy.median`` function across the remaining dimensions
    """
    if axis is None:
        raise NotImplementedError(
            "The da.median function only works along an axis.  "
            "The full algorithm is difficult to do in parallel"
        )

    if not isinstance(axis, Iterable):
        axis = (axis,)

    axis = [ax + a.ndim if ax < 0 else ax for ax in axis]

    # rechunk if reduced axes are not contained in a single chunk
    if builtins.any(a.numblocks[ax] > 1 for ax in axis):
        a = a.rechunk({ax: -1 if ax in axis else "auto" for ax in range(a.ndim)})

    result = a.map_blocks(
        np.median,
        axis=axis,
        keepdims=keepdims,
        drop_axis=axis if not keepdims else None,
        chunks=(
            [1 if ax in axis else c for ax, c in enumerate(a.chunks)]
            if keepdims
            else None
        ),
    )

    result = handle_out(out, result)
    return result


@derived_from(np)
def nanmedian(a, axis=None, keepdims=False, out=None):
    """
    This works by automatically chunking the reduced axes to a single chunk
    and then calling ``numpy.nanmedian`` function across the remaining dimensions
    """
    from packaging.version import Version

    if axis is None:
        raise NotImplementedError(
            "The da.nanmedian function only works along an axis or a subset of axes.  "
            "The full algorithm is difficult to do in parallel"
        )

    if not isinstance(axis, Iterable):
        axis = (axis,)

    axis = [ax + a.ndim if ax < 0 else ax for ax in axis]

    # rechunk if reduced axes are not contained in a single chunk
    if builtins.any(a.numblocks[ax] > 1 for ax in axis):
        a = a.rechunk({ax: -1 if ax in axis else "auto" for ax in range(a.ndim)})

    if (
        numbagg is not None
        and Version(numbagg.__version__).release >= (0, 7, 0)
        and a.dtype.kind in "uif"
        and not keepdims
    ):
        func = numbagg.nanmedian
        kwargs = {}
    else:
        func = np.nanmedian
        kwargs = {"keepdims": keepdims}

    result = a.map_blocks(
        func,
        axis=axis,
        drop_axis=axis if not keepdims else None,
        chunks=(
            [1 if ax in axis else c for ax, c in enumerate(a.chunks)]
            if keepdims
            else None
        ),
        **kwargs,
    )

    result = handle_out(out, result)
    return result


def _get_quantile_chunks(a, q, axis, keepdims):
    quantile_chunk = [len(q)] if isinstance(q, Iterable) else []
    if keepdims:
        return quantile_chunk + [
            1 if ax in axis else c for ax, c in enumerate(a.chunks)
        ]
    else:
        return quantile_chunk + [c for ax, c in enumerate(a.chunks) if ax not in axis]


def _span_indexers(a):
    shapes = 1 if len(a.shape) <= 2 else reduce(mul, list(a.shape)[1:-1])
    original_shapes = shapes * a.shape[0]
    indexers = [tuple(np.repeat(np.arange(a.shape[0]), shapes))]

    for i in range(1, len(a.shape) - 1):
        indexer = np.repeat(np.arange(a.shape[i]), shapes // a.shape[i])
        indexers.append(tuple(np.tile(indexer, original_shapes // shapes)))
        shapes //= a.shape[i]
    return indexers


def _custom_quantile(a, q, axis=None, method="linear", keepdims=False, **kwargs):
    if (
        method != "linear"
        or len(axis) != 1
        or axis[0] != len(a.shape) - 1
        or len(a.shape) == 1
        or a.shape[-1] > 1000
    ):
        return np.nanquantile(a, q, axis=axis, method=method, keepdims=keepdims, **kwargs)

    sorted_arr = np.sort(a, axis=-1)
    indexers = _span_indexers(a)
    nr_quantiles = len(indexers[0])

    is_scalar = False
    if not isinstance(q, Iterable):
        is_scalar = True
        q = [q]

    quantiles = []
    reshape_shapes = (1,) + tuple(sorted_arr.shape[:-1]) + ((1,) if keepdims else ())
    for single_q in list(q):
        i = (
            np.ones(nr_quantiles) * (a.shape[-1] - 1)
            - np.isnan(sorted_arr).sum(axis=-1).reshape(-1)
        ) * single_q
        lower_value, higher_value = np.floor(i).astype(int), np.ceil(i).astype(int)

        lower = sorted_arr[tuple(indexers) + (tuple(lower_value),)]
        higher = sorted_arr[tuple(indexers) + (tuple(higher_value),)]

        factor_higher = i - lower_value
        factor_higher = np.where(factor_higher == 0.0, 1.0, factor_higher)
        factor_lower = higher_value - i

        quantiles.append(
            (higher * factor_higher + lower * factor_lower).reshape(*reshape_shapes)
        )

    if is_scalar:
        return quantiles[0].squeeze(axis=0)
    else:
        return np.concatenate(quantiles, axis=0)


@derived_from(np)
def quantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    weights=None,
    interpolation=None,
):
    """
    This works by automatically chunking the reduced axes to a single chunk if necessary
    and then calling ``numpy.quantile`` function across the remaining dimensions
    """
    if interpolation is not None:
        warnings.warn(
            "The `interpolation` argument to quantile was renamed to `method`.",
            FutureWarning,
            stacklevel=2,
        )
        if method != "linear":
            raise TypeError("Cannot pass interpolation and method keywords!")
        method = interpolation

    if axis is None:
        if builtins.any(n_blocks > 1 for n_blocks in a.numblocks):
            raise NotImplementedError(
                "The da.quantile function only works along an axis.  "
                "The full algorithm is difficult to do in parallel"
            )
        else:
            axis = tuple(range(len(a.shape)))

    if not isinstance(axis, Iterable):
        axis = (axis,)

    axis = [ax + a.ndim if ax < 0 else ax for ax in axis]

    # rechunk if reduced axes are not contained in a single chunk
    if builtins.any(a.numblocks[ax] > 1 for ax in axis):
        a = a.rechunk({ax: -1 if ax in axis else "auto" for ax in range(a.ndim)})

    # NumPy >= 2.0 supports weights
    kwargs = {}
    try:
        # Check if weights parameter is supported
        import numpy as np
        if hasattr(np.quantile, '__wrapped__') or weights is not None:
            kwargs["weights"] = weights
    except Exception:
        pass

    result = a.map_blocks(
        np.quantile,
        q=q,
        method=method,
        axis=axis,
        keepdims=keepdims,
        drop_axis=axis if not keepdims else None,
        new_axis=0 if isinstance(q, Iterable) else None,
        chunks=_get_quantile_chunks(a, q, axis, keepdims),
        **kwargs,
    )

    result = handle_out(out, result)
    return result


@derived_from(np)
def nanquantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    weights=None,
    interpolation=None,
):
    """
    This works by automatically chunking the reduced axes to a single chunk
    and then calling ``numpy.nanquantile`` function across the remaining dimensions
    """
    from packaging.version import Version

    if interpolation is not None:
        warnings.warn(
            "The `interpolation` argument to nanquantile was renamed to `method`.",
            FutureWarning,
            stacklevel=2,
        )
        if method != "linear":
            raise TypeError("Cannot pass interpolation and method keywords!")
        method = interpolation

    if axis is None:
        if builtins.any(n_blocks > 1 for n_blocks in a.numblocks):
            raise NotImplementedError(
                "The da.nanquantile function only works along an axis.  "
                "The full algorithm is difficult to do in parallel"
            )
        else:
            axis = tuple(range(len(a.shape)))

    if not isinstance(axis, Iterable):
        axis = (axis,)

    axis = [ax + a.ndim if ax < 0 else ax for ax in axis]

    # rechunk if reduced axes are not contained in a single chunk
    if builtins.any(a.numblocks[ax] > 1 for ax in axis):
        a = a.rechunk({ax: -1 if ax in axis else "auto" for ax in range(a.ndim)})

    if (
        numbagg is not None
        and Version(numbagg.__version__).release >= (0, 8, 0)
        and a.dtype.kind in "uif"
        and weights is None
        and method == "linear"
        and not keepdims
    ):
        func = numbagg.nanquantile
        kwargs = {"quantiles": q}
    else:
        func = _custom_quantile
        kwargs = {
            "q": q,
            "method": method,
            "keepdims": keepdims,
        }
        # NumPy >= 2.0 supports weights
        if weights is not None:
            kwargs["weights"] = weights

    result = a.map_blocks(
        func,
        axis=axis,
        drop_axis=axis if not keepdims else None,
        new_axis=0 if isinstance(q, Iterable) else None,
        chunks=_get_quantile_chunks(a, q, axis, keepdims),
        **kwargs,
    )

    result = handle_out(out, result)
    return result
