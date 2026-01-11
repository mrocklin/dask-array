from __future__ import annotations

import contextlib
import numbers
import warnings
from typing import Any

import numpy as np
from numpy.exceptions import AxisError

from dask.base import is_dask_collection
from dask.utils import has_keyword


def typename(typ: Any, short: bool = False) -> str:
    """Return the name of a type.

    Examples
    --------
    >>> typename(int)
    'int'
    """
    if not isinstance(typ, type):
        return typename(type(typ))
    try:
        if not typ.__module__ or typ.__module__ == "builtins":
            return typ.__name__
        else:
            if short:
                module, *_ = typ.__module__.split(".")
            else:
                module = typ.__module__
            return f"{module}.{typ.__name__}"
    except AttributeError:
        return str(typ)


def is_cupy_type(x) -> bool:
    """Check if x is a CuPy array type."""
    return "cupy" in str(type(x))


def is_arraylike(x) -> bool:
    """Is this object a numpy array or something similar?

    This function tests specifically for an object that already has
    array attributes (e.g. np.ndarray, dask.array.Array, cupy.ndarray,
    sparse.COO), **NOT** for something that can be coerced into an
    array object (e.g. Python lists and tuples).

    Examples
    --------
    >>> import numpy as np
    >>> is_arraylike(np.ones(5))
    True
    >>> is_arraylike(np.ones(()))
    True
    >>> is_arraylike(5)
    False
    >>> is_arraylike('cat')
    False
    """
    is_duck_array = hasattr(x, "__array_function__") or hasattr(x, "__array_ufunc__")

    return bool(
        hasattr(x, "shape")
        and isinstance(x.shape, tuple)
        and hasattr(x, "dtype")
        and not any(is_dask_collection(n) for n in x.shape)
        # We special case scipy.sparse and cupyx.scipy.sparse arrays as having partial
        # support for them is useful in scenarios where we mostly call `map_partitions`
        # or `map_blocks` with scikit-learn functions on dask arrays and dask dataframes.
        and (is_duck_array or "scipy.sparse" in typename(type(x)))
    )


def meta_from_array(x, ndim=None, dtype=None):
    """Normalize an array to appropriate meta object.

    Parameters
    ----------
    x: array-like, callable
        Either an object that looks sufficiently like a Numpy array,
        or a callable that accepts shape and dtype keywords
    ndim: int
        Number of dimensions of the array
    dtype: Numpy dtype
        A valid input for ``np.dtype``

    Returns
    -------
    array-like with zero elements of the correct dtype
    """
    # If using x._meta, x must be a Dask Array, some libraries (e.g. zarr)
    # implement a _meta attribute that are incompatible with Dask Array._meta
    if hasattr(x, "_meta") and is_dask_collection(x) and is_arraylike(x):
        x = x._meta

    if dtype is None and x is None:
        raise ValueError("You must specify the meta or dtype of the array")

    if np.isscalar(x):
        x = np.array(x)

    if x is None:
        x = np.ndarray
    elif dtype is None and hasattr(x, "dtype"):
        dtype = x.dtype

    if isinstance(x, type):
        x = x(shape=(0,) * (ndim or 0), dtype=dtype)

    if isinstance(x, (list, tuple)):
        ndims = [(0 if isinstance(a, numbers.Number) else a.ndim if hasattr(a, "ndim") else len(a)) for a in x]
        a = [a if nd == 0 else meta_from_array(a, nd) for a, nd in zip(x, ndims)]
        return a if isinstance(x, list) else tuple(x)

    if not hasattr(x, "shape") or not hasattr(x, "dtype") or not isinstance(x.shape, tuple):
        return x

    if ndim is None:
        ndim = x.ndim

    try:
        meta = x[tuple(slice(0, 0, None) for _ in range(x.ndim))]
        if meta.ndim != ndim:
            if ndim > x.ndim:
                meta = meta[(Ellipsis,) + tuple(None for _ in range(ndim - meta.ndim))]
                meta = meta[tuple(slice(0, 0, None) for _ in range(meta.ndim))]
            elif ndim == 0:
                meta = meta.sum()
            else:
                meta = meta.reshape((0,) * ndim)
        if meta is np.ma.masked:
            meta = np.ma.array(np.empty((0,) * ndim, dtype=dtype or x.dtype), mask=True)
    except Exception:
        meta = np.empty((0,) * ndim, dtype=dtype or x.dtype)

    if np.isscalar(meta):
        meta = np.array(meta)

    if dtype and meta.dtype != dtype:
        try:
            meta = meta.astype(dtype)
        except ValueError as e:
            if (
                any(
                    s in str(e)
                    for s in [
                        "invalid literal",
                        "could not convert string to float",
                    ]
                )
                and meta.dtype.kind in "SU"
            ):
                meta = np.array([]).astype(dtype)
            else:
                raise e

    return meta


def validate_axis(axis, ndim):
    """Validate an input to axis= keywords."""
    if isinstance(axis, (tuple, list)):
        return tuple(validate_axis(ax, ndim) for ax in axis)
    if not isinstance(axis, numbers.Integral):
        raise TypeError(f"Axis value must be an integer, got {axis}")
    if axis < -ndim or axis >= ndim:
        raise AxisError(f"Axis {axis} is out of bounds for array of dimension {ndim}")
    if axis < 0:
        axis += ndim
    return axis


def arange_safe(*args, like, **kwargs):
    """Use the `like=` from `np.arange` to create a new array dispatching
    to the downstream library. If that fails, falls back to the
    default NumPy behavior, resulting in a `numpy.ndarray`.
    """
    if like is None:
        return np.arange(*args, **kwargs)
    else:
        try:
            return np.arange(*args, like=meta_from_array(like), **kwargs)
        except TypeError:
            return np.arange(*args, **kwargs)


def _array_like_safe(np_func, da_func, a, like, **kwargs):
    """Helper for array_safe, asarray_safe, asanyarray_safe."""
    from dask_array._collection import Array

    if like is a and hasattr(a, "__array_function__"):
        return a

    if isinstance(like, Array):
        return da_func(a, **kwargs)

    if isinstance(a, Array) and is_cupy_type(a._meta):
        a = a.compute(scheduler="sync")

    if hasattr(like, "__array_function__"):
        return np_func(a, like=like, **kwargs)

    if type(like).__module__.startswith("scipy.sparse"):
        # e.g. scipy.sparse.csr_matrix
        kwargs.pop("order", None)
        if np.isscalar(a):
            a = np.array([[a]])
        return type(like)(a, **kwargs)

    # Unknown namespace with no __array_function__ support.
    # Quietly disregard like= parameter.
    return np_func(a, **kwargs)


def array_safe(a, like, **kwargs):
    """If `a` is `dask_array.Array`, return `dask_array.asarray(a, **kwargs)`,
    otherwise return `np.asarray(a, like=like, **kwargs)`, dispatching
    the call to the library that implements the like array.
    """
    from dask_array.core import array

    return _array_like_safe(np.array, array, a, like, **kwargs)


def asarray_safe(a, like, **kwargs):
    """If a is dask_array.Array, return dask_array.asarray(a, **kwargs),
    otherwise return np.asarray(a, like=like, **kwargs), dispatching
    the call to the library that implements the like array.
    """
    from dask_array.core import asarray

    return _array_like_safe(np.asarray, asarray, a, like, **kwargs)


def asanyarray_safe(a, like, **kwargs):
    """If a is dask_array.Array, return dask_array.asanyarray(a, **kwargs),
    otherwise return np.asanyarray(a, like=like, **kwargs), dispatching
    the call to the library that implements the like array.
    """
    from dask_array.core import asanyarray

    return _array_like_safe(np.asanyarray, asanyarray, a, like, **kwargs)


def svd_flip(u, v, u_based_decision=False):
    """Sign correction to ensure deterministic output from SVD.

    This function is useful for orienting eigenvectors such that
    they all lie in a shared but arbitrary half-space. This makes
    it possible to ensure that results are equivalent across SVD
    implementations and random number generator states.

    Parameters
    ----------
    u : (M, K) array_like
        Left singular vectors (in columns)
    v : (K, N) array_like
        Right singular vectors (in rows)
    u_based_decision: bool
        Whether or not to choose signs based
        on `u` rather than `v`, by default False

    Returns
    -------
    u : (M, K) array_like
        Left singular vectors with corrected sign
    v:  (K, N) array_like
        Right singular vectors with corrected sign
    """
    if u_based_decision:
        dtype = u.dtype
        signs = np.sum(u, axis=0, keepdims=True)
    else:
        dtype = v.dtype
        signs = np.sum(v, axis=1, keepdims=True).T
    signs = 2.0 * ((signs >= 0) - 0.5).astype(dtype)
    u, v = u * signs, v * signs.T
    return u, v


def solve_triangular_safe(a, b, lower=False):
    """Solve triangular system using scipy.linalg (or cupyx for GPU)."""
    if is_cupy_type(a):
        import cupyx.scipy.linalg

        return cupyx.scipy.linalg.solve_triangular(a, b, lower=lower)
    else:
        import scipy.linalg

        return scipy.linalg.solve_triangular(a, b, lower=lower)


def compute_meta(func, _dtype, *args, **kwargs):
    """Compute metadata for an operation."""
    from dask_array._expr import ArrayExpr

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        args_meta = [
            (x._meta if isinstance(x, ArrayExpr) else meta_from_array(x) if is_arraylike(x) else x) for x in args
        ]
        kwargs_meta = {
            k: (v._meta if isinstance(v, ArrayExpr) else meta_from_array(v) if is_arraylike(v) else v)
            for k, v in kwargs.items()
        }

        # todo: look for alternative to this, causes issues when using map_blocks()
        # with np.vectorize, such as dask.array.routines._isnonzero_vec().
        if isinstance(func, np.vectorize):
            meta = func(*args_meta)
        else:
            try:
                # some reduction functions need to know they are computing meta
                if has_keyword(func, "computing_meta"):
                    kwargs_meta["computing_meta"] = True
                meta = func(*args_meta, **kwargs_meta)
            except TypeError as e:
                if any(
                    s in str(e)
                    for s in [
                        "unexpected keyword argument",
                        "is an invalid keyword for",
                        "Did not understand the following kwargs",
                    ]
                ):
                    raise
                else:
                    return None
            except ValueError as e:
                # min/max functions have no identity, just use the same input type when there's only one
                if len(args_meta) == 1 and "zero-size array to reduction operation" in str(e):
                    meta = args_meta[0]
                else:
                    return None
            except Exception:
                return None

        if _dtype and getattr(meta, "dtype", None) != _dtype:
            with contextlib.suppress(AttributeError):
                meta = meta.astype(_dtype)

        if np.isscalar(meta):
            meta = np.array(meta)

        return meta
