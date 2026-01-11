"""Test utilities for dask_array.

These functions are used in tests to compare dask arrays with numpy arrays.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from dask.base import is_dask_collection

from dask_array._utils import is_arraylike, is_cupy_type


def normalize_to_array(x):
    """Convert CuPy arrays to numpy arrays."""
    if is_cupy_type(x):
        return x.get()
    else:
        return x


def allclose(a, b, equal_nan=False, **kwargs):
    """Check if two arrays are element-wise equal within a tolerance."""
    a = normalize_to_array(a)
    b = normalize_to_array(b)
    if getattr(a, "dtype", None) != "O":
        if hasattr(a, "mask") or hasattr(b, "mask"):
            return np.ma.allclose(a, b, masked_equal=True, **kwargs)
        else:
            return np.allclose(a, b, equal_nan=equal_nan, **kwargs)
    if equal_nan:
        return a.shape == b.shape and all(np.isnan(b) if np.isnan(a) else a == b for (a, b) in zip(a.flat, b.flat))
    return (a == b).all()


def same_keys(a, b):
    """Check if two dask collections have the same keys in their graphs."""

    def key(k):
        if isinstance(k, str):
            return (k, -1, -1, -1)
        else:
            return k

    return sorted(a.dask, key=key) == sorted(b.dask, key=key)


def _not_empty(x):
    return x.shape and 0 not in x.shape


def assert_eq_shape(a, b, check_ndim=True, check_nan=True):
    """Assert that two shapes are equal, handling NaN values."""
    if check_ndim:
        assert len(a) == len(b)

    for aa, bb in zip(a, b):
        if math.isnan(aa) or math.isnan(bb):
            if check_nan:
                assert math.isnan(aa) == math.isnan(bb)
        else:
            assert aa == bb


def _check_chunks(x, check_ndim=True, scheduler=None):
    """Check that chunk shapes match expected shapes."""
    x = x.persist(scheduler=scheduler)
    dsk = x.dask  # Cache to avoid repeated graph materialization
    for idx in itertools.product(*(range(len(c)) for c in x.chunks)):
        chunk = dsk[(x.name,) + idx]
        if hasattr(chunk, "result"):  # it's a future
            chunk = chunk.result()
        if not hasattr(chunk, "dtype"):
            chunk = np.array(chunk, dtype="O")
        expected_shape = tuple(c[i] for c, i in zip(x.chunks, idx))
        assert_eq_shape(expected_shape, chunk.shape, check_ndim=check_ndim, check_nan=False)
        assert chunk.dtype == x.dtype, "maybe you forgot to pass the scheduler to `assert_eq`?"
    return x


def _get_dt_meta_computed(
    x,
    check_shape=True,
    check_graph=True,
    check_chunks=True,
    check_ndim=True,
    scheduler=None,
):
    """Get dtype, meta, and computed value from an array-like."""
    x_original = x
    x_meta = None
    x_computed = None

    if is_dask_collection(x) and is_arraylike(x):
        assert x.dtype is not None
        adt = x.dtype
        # Note: check_graph is ignored in array-expr mode as it triggers
        # expensive graph regeneration. The HLG validation can be enabled
        # when needed for debugging.
        x_meta = getattr(x, "_meta", None)
        if check_chunks:
            # Replace x with persisted version to avoid computing it twice.
            x = _check_chunks(x, check_ndim=check_ndim, scheduler=scheduler)
        x = x.compute(scheduler=scheduler)
        x_computed = x
        if hasattr(x, "todense"):
            x = x.todense()
        if not hasattr(x, "dtype"):
            x = np.array(x, dtype="O")
        if _not_empty(x):
            assert x.dtype == x_original.dtype
        if check_shape:
            assert_eq_shape(x_original.shape, x.shape, check_nan=False)
    else:
        if not hasattr(x, "dtype"):
            x = np.array(x, dtype="O")
        adt = getattr(x, "dtype", None)

    return x, adt, x_meta, x_computed


def assert_eq(
    a,
    b,
    check_shape=True,
    check_graph=True,
    check_meta=True,
    check_chunks=True,
    check_ndim=True,
    check_type=True,
    check_dtype=True,
    equal_nan=True,
    scheduler="sync",
    **kwargs,
):
    """Assert that two arrays are equal.

    This function handles dask arrays, numpy arrays, and other array-likes.
    It computes dask arrays before comparison and performs various checks.

    Parameters
    ----------
    a, b : array-like
        Arrays to compare
    check_shape : bool
        Whether to check that shapes match
    check_graph : bool
        Whether to validate the dask graph (currently not implemented locally)
    check_meta : bool
        Whether to check metadata consistency
    check_chunks : bool
        Whether to check chunk shapes
    check_ndim : bool
        Whether to check that ndims match
    check_type : bool
        Whether to check that types match
    check_dtype : bool
        Whether to check that dtypes match
    equal_nan : bool
        Whether to treat NaN values as equal
    scheduler : str
        Scheduler to use for computing dask arrays

    Returns
    -------
    bool
        True if arrays are equal
    """
    a_original = a
    b_original = b

    if isinstance(a, (list, int, float)):
        a = np.array(a)
    if isinstance(b, (list, int, float)):
        b = np.array(b)

    a, adt, a_meta, a_computed = _get_dt_meta_computed(
        a,
        check_shape=check_shape,
        check_graph=check_graph,
        check_chunks=check_chunks,
        check_ndim=check_ndim,
        scheduler=scheduler,
    )
    b, bdt, b_meta, b_computed = _get_dt_meta_computed(
        b,
        check_shape=check_shape,
        check_graph=check_graph,
        check_chunks=check_chunks,
        check_ndim=check_ndim,
        scheduler=scheduler,
    )

    if check_dtype and str(adt) != str(bdt):
        raise AssertionError(f"a and b have different dtypes: (a: {adt}, b: {bdt})")

    try:
        assert a.shape == b.shape, f"a and b have different shapes (a: {a.shape}, b: {b.shape})"
        if check_type:
            _a = a if a.shape else a.item()
            _b = b if b.shape else b.item()
            assert type(_a) == type(_b), f"a and b have different types (a: {type(_a)}, b: {type(_b)})"
        if check_meta:
            if hasattr(a, "_meta") and hasattr(b, "_meta"):
                assert_eq(a._meta, b._meta)
            if hasattr(a_original, "_meta"):
                msg = (
                    f"compute()-ing 'a' changes its number of dimensions "
                    f"(before: {a_original._meta.ndim}, after: {a.ndim})"
                )
                assert a_original._meta.ndim == a.ndim, msg
                if a_meta is not None:
                    msg = (
                        f"compute()-ing 'a' changes its type (before: {type(a_original._meta)}, after: {type(a_meta)})"
                    )
                    assert type(a_original._meta) == type(a_meta), msg
                    if not (np.isscalar(a_meta) or np.isscalar(a_computed)):
                        msg = (
                            f"compute()-ing 'a' results in a different type than implied by its metadata "
                            f"(meta: {type(a_meta)}, computed: {type(a_computed)})"
                        )
                        assert type(a_meta) == type(a_computed), msg
            if hasattr(b_original, "_meta"):
                msg = (
                    f"compute()-ing 'b' changes its number of dimensions "
                    f"(before: {b_original._meta.ndim}, after: {b.ndim})"
                )
                assert b_original._meta.ndim == b.ndim, msg
                if b_meta is not None:
                    msg = (
                        f"compute()-ing 'b' changes its type (before: {type(b_original._meta)}, after: {type(b_meta)})"
                    )
                    assert type(b_original._meta) == type(b_meta), msg
                    if not (np.isscalar(b_meta) or np.isscalar(b_computed)):
                        msg = (
                            f"compute()-ing 'b' results in a different type than implied by its metadata "
                            f"(meta: {type(b_meta)}, computed: {type(b_computed)})"
                        )
                        assert type(b_meta) == type(b_computed), msg
        msg = "found values in 'a' and 'b' which differ by more than the allowed amount"
        assert allclose(a, b, equal_nan=equal_nan, **kwargs), msg
        return True
    except TypeError:
        pass

    c = a == b

    if isinstance(c, np.ndarray):
        assert c.all()
    else:
        assert c

    return True
