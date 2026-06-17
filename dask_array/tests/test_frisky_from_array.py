"""Records-path (``to_task_records``) parity for the native FromArray getter layer.

The rest of the suite validates ``to_dask_graph`` indirectly (via ``.compute()``),
but ``to_task_records`` — the path that actually executes on Frisky — is built
separately in Rust (``crates/dask-array-python/src/from_array.rs``). These tests
run the records through a trivial dependency-ordered resolver and assert the
reassembled array equals the numpy ground truth, for a *generic array-like*
source (which takes the getter layer, not the ndarray eager-slice layer): plain,
``_region`` slice-pushdown, ``asarray=False``, and a real zarr array.
"""

from __future__ import annotations

import numpy as np
import pytest

import dask_array as da
from dask._task_spec import TaskRef
from dask.core import flatten
from dask_array._frisky import collect_task_records


class _ArrayLike:
    """A non-ndarray slicing target, so ``from_array`` takes the getter layer
    rather than the eager-slice ndarray layer. Deterministically tokenizable."""

    def __init__(self, base):
        self._base = base

    def __dask_tokenize__(self):
        return ("_ArrayLike", self._base.shape, self._base.dtype.str, self._base.tobytes())

    @property
    def shape(self):
        return self._base.shape

    @property
    def dtype(self):
        return self._base.dtype

    @property
    def ndim(self):
        return self._base.ndim

    def __getitem__(self, idx):
        return np.asarray(self._base[idx])


def _run_records(records):
    """Execute flat ``(key, func, args, kwargs, deps)`` records in dependency
    order; return ``{key_str: value}``. ``TaskRef``s in args resolve to prior
    results — a trivial stand-in for a Frisky worker."""
    by_key = {}

    def resolve(a):
        if isinstance(a, TaskRef):
            return by_key[str(a.key)]
        if isinstance(a, tuple):
            return tuple(resolve(x) for x in a)
        if isinstance(a, list):
            return [resolve(x) for x in a]
        if isinstance(a, dict):
            return {k: resolve(v) for k, v in a.items()}
        return a

    remaining = [(str(k), f, args, kw, [str(d) for d in deps]) for k, f, args, kw, deps in records]
    while remaining:
        still, progressed = [], False
        for key, func, args, kwargs, deps in remaining:
            if all(d in by_key for d in deps):
                by_key[key] = func(*(resolve(a) for a in args), **(kwargs or {}))
                progressed = True
            else:
                still.append((key, func, args, kwargs, deps))
        if not progressed:
            raise AssertionError("records have a cycle or a missing dependency")
        remaining = still
    return by_key


def _reassemble(arr, by_key):
    """Place each computed block into the full array per the chunk grid."""
    full = np.empty(arr.shape, dtype=arr.dtype)
    offsets = [np.concatenate([[0], np.cumsum(c)]).astype(int) for c in arr.chunks]
    for key in flatten(arr.__dask_keys__()):
        idx = key[1:]
        full[tuple(slice(offsets[d][i], offsets[d][i + 1]) for d, i in enumerate(idx))] = by_key[str(key)]
    return full


def _assert_records_match(arr, expected):
    by_key = _run_records(collect_task_records(arr))
    # every advertised output key is actually produced by the records
    assert set(arr.__frisky_output_keys__()) <= set(by_key)
    np.testing.assert_array_equal(_reassemble(arr, by_key), expected)


def test_records_from_arraylike_basic():
    base = np.arange(120.0).reshape(8, 15)
    arr = da.from_array(_ArrayLike(base), chunks=(3, 5))
    # sanity: this really took the native getter layer, not a fallback
    arr.expr.simplify().lower_completely()._frisky_layer()
    _assert_records_match(arr, base)


def test_records_from_arraylike_region_pushdown():
    base = np.arange(120.0).reshape(8, 15)
    arr = da.from_array(_ArrayLike(base), chunks=(3, 5))[2:7, 3:12]
    _assert_records_match(arr, base[2:7, 3:12])


def test_records_from_arraylike_no_asarray():
    base = np.arange(120.0).reshape(8, 15)
    arr = da.from_array(_ArrayLike(base), chunks=(3, 5), asarray=False)
    _assert_records_match(arr, base)


def test_records_from_arraylike_3d():
    base = np.arange(2 * 6 * 4, dtype="f8").reshape(2, 6, 4)
    arr = da.from_array(_ArrayLike(base), chunks=(1, 2, 4))
    _assert_records_match(arr, base)


def test_records_from_zarr():
    zarr = pytest.importorskip("zarr")
    base = np.arange(120.0).reshape(8, 15)
    z = zarr.array(base, chunks=(8, 15))
    arr = da.from_array(z, chunks=(3, 5))
    _assert_records_match(arr, base)
