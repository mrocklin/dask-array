"""Percentile functions for dask arrays."""
from __future__ import annotations

import warnings
from collections.abc import Iterator
from functools import wraps
from numbers import Number

import numpy as np
from tlz import merge

from dask_array._dispatch import empty_lookup, percentile_lookup
from dask.base import tokenize
from dask.utils import derived_from

from dask_array.core import from_graph


@wraps(np.percentile)
def _percentile(a, q, method="linear"):
    n = len(a)
    if not len(a):
        return None, n
    if isinstance(q, Iterator):
        q = list(q)
    if a.dtype.name == "category":
        result = np.percentile(a.cat.codes, q, method=method)
        import pandas as pd

        return pd.Categorical.from_codes(result, a.dtype.categories, a.dtype.ordered), n
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
            result[0] = min(result[0], values.min())
        return result, n
    if not np.issubdtype(a.dtype, np.number):
        method = "nearest"
    return np.percentile(a, q, method=method), n


def _tdigest_chunk(a):
    from crick import TDigest

    t = TDigest()
    t.update(a)

    return t


def _percentiles_from_tdigest(qs, digests):
    from crick import TDigest

    t = TDigest()
    t.merge(*digests)

    return np.array(t.quantile(qs / 100.0))


def merge_percentiles(finalq, qs, vals, method="lower", Ns=None, raise_on_nan=True):
    """Combine several percentile calculations of different data."""
    from dask_array._utils import array_safe

    if isinstance(finalq, Iterator):
        finalq = list(finalq)
    finalq = array_safe(finalq, like=finalq)
    qs = [list(q) for q in qs]
    vals = list(vals)
    if Ns is None:
        vals, Ns = zip(*vals)
    Ns = list(Ns)

    L = list(zip(*((q, val, N) for q, val, N in zip(qs, vals, Ns) if N)))
    if not L:
        if raise_on_nan:
            raise ValueError("No non-trivial arrays found")
        return np.full(len(qs[0]) - 2, np.nan)
    qs, vals, Ns = L

    if vals[0].dtype.name == "category":
        result = merge_percentiles(
            finalq, qs, [v.codes for v in vals], method, Ns, raise_on_nan
        )
        import pandas as pd

        return pd.Categorical.from_codes(result, vals[0].categories, vals[0].ordered)
    if not np.issubdtype(vals[0].dtype, np.number):
        method = "nearest"

    if len(vals) != len(qs) or len(Ns) != len(qs):
        raise ValueError("qs, vals, and Ns parameters must be the same length")

    total_len = sum(len(q) for q in qs)
    counts = empty_lookup.dispatch(type(finalq))(total_len, dtype=finalq.dtype)
    start = 0
    for q, N in zip(qs, Ns):
        length = len(q)
        count = empty_lookup.dispatch(type(finalq))(length, dtype=finalq.dtype)
        count[1:] = np.diff(array_safe(q, like=q[0]))
        count[0] = q[0]
        count *= N
        counts[start : start + length] = count
        start += length

    combined_vals = np.concatenate(vals)
    combined_counts = array_safe(counts, like=combined_vals)
    sort_order = np.argsort(combined_vals)
    combined_vals = np.take(combined_vals, sort_order)
    combined_counts = np.take(combined_counts, sort_order)

    combined_q = np.cumsum(combined_counts)

    finalq = array_safe(finalq, like=combined_vals)
    desired_q = finalq * sum(Ns)

    if method == "linear":
        rv = np.interp(desired_q, combined_q, combined_vals)
    else:
        left = np.searchsorted(combined_q, desired_q, side="left")
        right = np.searchsorted(combined_q, desired_q, side="right") - 1
        np.minimum(left, len(combined_vals) - 1, out=left)
        lower = np.minimum(left, right)
        upper = np.maximum(left, right)
        if method == "lower":
            rv = combined_vals[lower]
        elif method == "higher":
            rv = combined_vals[upper]
        elif method == "midpoint":
            rv = 0.5 * (combined_vals[lower] + combined_vals[upper])
        elif method == "nearest":
            lower_residual = np.abs(combined_q[lower] - desired_q)
            upper_residual = np.abs(combined_q[upper] - desired_q)
            mask = lower_residual > upper_residual
            index = lower
            index[mask] = upper[mask]
            rv = combined_vals[index]
        else:
            raise ValueError(
                "interpolation method can only be 'linear', 'lower', "
                "'higher', 'midpoint', or 'nearest'"
            )
    return rv


def percentile(a, q, method="linear", internal_method="default", **kwargs):
    """Approximate percentile of 1-D array

    Parameters
    ----------
    a : Array
    q : array_like of float
        Percentile or sequence of percentiles to compute, which must be between
        0 and 100 inclusive.
    method : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, optional
        The interpolation method to use when the desired percentile lies
        between two data points.
    internal_method : {'default', 'dask', 'tdigest'}, optional
        What internal method to use. By default will use dask's internal custom
        algorithm (``'dask'``).
    """
    from dask_array._utils import array_safe, meta_from_array
    from dask_array.reductions import quantile

    if a.ndim == 1:
        allowed_internal_methods = {"default", "dask", "tdigest"}

        if method in allowed_internal_methods:
            warnings.warn(
                "The `method=` argument was renamed to `internal_method=`",
                FutureWarning,
            )
            internal_method = method

        if "interpolation" in kwargs:
            warnings.warn(
                "The `interpolation=` argument to percentile was renamed to `method= ` ",
                FutureWarning,
            )
            method = kwargs.pop("interpolation")

        if kwargs:
            raise TypeError(
                f"percentile() got an unexpected keyword argument {kwargs.keys()}"
            )

        q_is_number = False
        if isinstance(q, Number):
            q_is_number = True
            q = [q]
        q = array_safe(q, like=meta_from_array(a))
        token = tokenize(a, q, method)

        dtype = a.dtype
        if np.issubdtype(dtype, np.integer):
            dtype = (array_safe([], dtype=dtype, like=meta_from_array(a)) / 0.5).dtype
        meta = meta_from_array(a, dtype=dtype)

        if internal_method not in allowed_internal_methods:
            raise ValueError(
                f"`internal_method=` must be one of {allowed_internal_methods}"
            )

        if (
            internal_method == "tdigest"
            and method == "linear"
            and (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer))
        ):
            from dask.utils import import_required

            import_required(
                "crick", "crick is a required dependency for using the t-digest method."
            )

            name = "percentile_tdigest_chunk-" + token
            dsk = {
                (name, i): (_tdigest_chunk, key)
                for i, key in enumerate(a.__dask_keys__())
            }

            name2 = "percentile_tdigest-" + token
            dsk2 = {(name2, 0): (_percentiles_from_tdigest, q, sorted(dsk))}

        else:
            zero = empty_lookup.dispatch(type(q))(1, dtype=q.dtype)
            zero[:] = 0

            hundred = empty_lookup.dispatch(type(q))(1, dtype=q.dtype)
            hundred[:] = 100

            calc_q = np.concatenate((zero, q, hundred))
            name = "percentile_chunk-" + token
            dsk = {
                (name, i): (percentile_lookup, key, calc_q, method)
                for i, key in enumerate(a.__dask_keys__())
            }

            name2 = "percentile-" + token
            dsk2 = {
                (name2, 0): (
                    merge_percentiles,
                    q,
                    [calc_q] * len(a.chunks[0]),
                    sorted(dsk),
                    method,
                )
            }
        dsk = merge(dsk, dsk2)
        # Merge the dependency graph with our new tasks
        full_dsk = dict(a.__dask_graph__())
        full_dsk.update(dsk)
        arr = from_graph(full_dsk, meta, ((len(q),),), [(name2, 0)], name2)
        return arr.reshape(()) if q_is_number else arr

    elif a.ndim > 1:
        q = np.true_divide(q, a.dtype.type(100) if a.dtype.kind == "f" else 100)
        return quantile(a, q, method=method, **kwargs)
    else:
        raise NotImplementedError("support for arrays of ndim 0 is not implemented.")


@derived_from(np)
def nanpercentile(a, q, **kwargs):
    from dask_array.reductions import nanquantile

    q = np.true_divide(q, a.dtype.type(100) if a.dtype.kind == "f" else 100)

    return nanquantile(a, q, **kwargs)
