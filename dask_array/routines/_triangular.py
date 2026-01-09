"""Triangular matrix functions for array-expr."""

from __future__ import annotations

import numpy as np

from dask_array._collection import asarray
from dask.utils import derived_from


@derived_from(np)
def tril(m, k=0):
    from dask_array.creation import tri
    from dask_array.routines._where import where
    from dask.array.utils import meta_from_array

    m = asarray(m)
    mask = tri(
        *m.shape[-2:],
        k=k,
        dtype=bool,
        chunks=m.chunks[-2:],
        like=meta_from_array(m),
    )

    return where(mask, m, np.zeros_like(m._meta, shape=(1,)))


@derived_from(np)
def triu(m, k=0):
    from dask_array.creation import tri
    from dask_array.routines._where import where
    from dask.array.utils import meta_from_array

    m = asarray(m)
    mask = tri(
        *m.shape[-2:],
        k=k - 1,
        dtype=bool,
        chunks=m.chunks[-2:],
        like=meta_from_array(m),
    )

    return where(mask, np.zeros_like(m._meta, shape=(1,)), m)


@derived_from(np)
def tril_indices(n, k=0, m=None, chunks="auto"):
    from dask_array.creation import tri
    from dask_array.routines._nonzero import nonzero

    return nonzero(tri(n, m, k=k, dtype=bool, chunks=chunks))


@derived_from(np)
def tril_indices_from(arr, k=0):
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return tril_indices(arr.shape[-2], k=k, m=arr.shape[-1], chunks=arr.chunks)


@derived_from(np)
def triu_indices(n, k=0, m=None, chunks="auto"):
    from dask_array.creation import tri
    from dask_array.routines._nonzero import nonzero

    return nonzero(~tri(n, m, k=k - 1, dtype=bool, chunks=chunks))


@derived_from(np)
def triu_indices_from(arr, k=0):
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return triu_indices(arr.shape[-2], k=k, m=arr.shape[-1], chunks=arr.chunks)
