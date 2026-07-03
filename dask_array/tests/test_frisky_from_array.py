"""Frisky-graph parity for the native FromArray getter layer.

``__frisky_graph__`` emits the records that actually execute on Frisky. These
tests run those records through a trivial dependency-ordered resolver and assert
the reassembled array equals the numpy ground truth, for a *generic array-like*
source (which takes the getter layer, not the ndarray eager-slice layer): plain,
``_region`` slice-pushdown, ``asarray=False``, and a real zarr array.
"""

from __future__ import annotations

import cloudpickle
import importlib.util

import numpy as np
import pytest

import dask_array as da
from dask._task_spec import TaskRef
from dask.core import flatten


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


class _FineChunkedArrayLike:
    """Small-to-serialize source with many output chunks."""

    shape = (195, 389)
    dtype = np.dtype("float64")
    ndim = 2

    def __dask_tokenize__(self):
        return ("_FineChunkedArrayLike", self.shape, self.dtype.str)

    def __getitem__(self, idx):
        shape = tuple(s.stop - s.start for s in idx)
        return np.zeros(shape, dtype=self.dtype)


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
    assert not hasattr(arr, "__frisky_task_records__")
    by_key = _run_records(arr.__frisky_graph__())
    # every advertised output key is actually produced by the records
    assert set(arr.__frisky_output_keys__()) <= set(by_key)
    np.testing.assert_array_equal(_reassemble(arr, by_key), expected)


def _frisky_output_keys(arr):
    return arr.__frisky_output_keys__()


def _dask_keys(arr):
    return arr.__dask_keys__()


@pytest.mark.parametrize(
    "enumerate_keys",
    [_frisky_output_keys, _dask_keys],
    ids=["frisky-output-keys", "dask-keys"],
)
def test_from_array_key_cache_is_not_pickled_after_key_enumeration(enumerate_keys):
    arr = da.from_array(_FineChunkedArrayLike(), chunks=(1, 1)).optimize()
    clean_size = len(cloudpickle.dumps(arr))

    keys = enumerate_keys(arr)
    assert len(list(flatten(keys))) == 195 * 389
    # Collection keys derive from the raw name and block structure; they
    # never populate the expression-level key cache.
    assert "_cached_keys" not in arr.expr.__dict__

    blob = cloudpickle.dumps(arr)
    assert len(blob) < 50_000
    assert len(blob) < clean_size + 25_000
    assert enumerate_keys(cloudpickle.loads(blob)) == keys


def test_records_from_arraylike_basic():
    base = np.arange(120.0).reshape(8, 15)
    arr = da.from_array(_ArrayLike(base), chunks=(3, 5))
    # With the optional Rust extension present, sanity-check that this takes the
    # native getter layer rather than the generic graph adapter. Pure-Python CI
    # still exercises the records behavior below through the adapter fallback.
    if importlib.util.find_spec("dask_array._rust") is not None:
        arr.expr.simplify().lower_completely()._frisky_layer()
    _assert_records_match(arr, base)


def test_records_from_arraylike_region_pushdown():
    base = np.arange(120.0).reshape(8, 15)
    arr = da.from_array(_ArrayLike(base), chunks=(3, 5))[2:7, 3:12]
    _assert_records_match(arr, base[2:7, 3:12])


def test_records_from_region_rechunk_pushdown():
    base = np.arange(25 * 10).reshape(25, 10)
    arr = da.from_array(base, chunks=(8, 2))[1:17].rechunk((4, 4))
    _assert_records_match(arr, base[1:17])


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


def test_numpy_region_pushdown_slices_source_before_frisky_records():
    base = np.arange(1_000_000.0).reshape(1000, 1000)
    arr = da.from_array(base, chunks=(100, 100))[150:160, 230:240].optimize()

    fa = _from_array_expr(arr)
    assert fa.operand("_region") is None
    assert fa.array.shape == (10, 10)
    np.testing.assert_array_equal(fa.array, base[150:160, 230:240])

    records = arr.__frisky_graph__()
    assert len(records) == 1
    np.testing.assert_array_equal(records[0][2][0], base[150:160, 230:240])


def _from_array_expr(arr):
    """The FromArray node underneath a (possibly wrapped) from_array collection."""
    from dask_array.io._from_array import FromArray

    seen = set()

    def walk(e):
        if isinstance(e, FromArray):
            return e
        for op in getattr(e, "operands", []):
            if hasattr(op, "operands") and id(op) not in seen:
                seen.add(id(op))
                r = walk(op)
                if r is not None:
                    return r
        return None

    fa = walk(arr.expr)
    assert fa is not None, "no FromArray node found"
    return fa


class _FlakyTokenSource(_ArrayLike):
    """Array-like whose ``__dask_tokenize__`` returns a *different* value on
    every call — a stand-in for a source (e.g. an xarray/icechunk lazy-indexing
    adapter) whose token is not reproducible. Used to prove the FromArray token
    is *cached* (two calls must agree)."""

    _counter = 0

    def __dask_tokenize__(self):
        type(self)._counter += 1
        return ("flaky", type(self)._counter)


def test_from_array_token_is_cached_not_recomputed():
    """``FromArray.__dask_tokenize__`` must return a cached, stable token rather
    than re-tokenizing its source on every call.

    Parent nodes (e.g. the Blockwise / PartialReduce a reduction lowers to)
    tokenize the FromArray to build *their* keys. If that re-walked an unstable
    source each call, the lowered keys would differ between the client (live
    array) and the scheduler (the same array unpickled in another process) —
    exactly the cross-process output-key mismatch that silently dropped a
    Frisky gather and hung ``.mean().compute()`` while ``.sum()`` happened to
    survive. A source that tokenizes differently every call makes the
    regression deterministic: without caching, two calls disagree.
    """
    base = np.arange(120.0).reshape(8, 15)
    fa = _from_array_expr(da.from_array(_FlakyTokenSource(base), chunks=(3, 5)))
    assert fa.__dask_tokenize__() == fa.__dask_tokenize__()


def test_reduction_output_key_stable_across_processes():
    """The behavioral invariant at the boundary Frisky's expression submission
    relies on: a reduction-over-FromArray must advertise the *same* output key
    in the client (live array) and in the scheduler (the same array cloudpickled
    and re-derived in another process).

    This must run in a real subprocess. An in-process pickle round-trip cannot
    reproduce the bug: ``SingletonExpr._instances`` is keyed by ``_name`` (which
    is pickle-stable via the cached ``deterministic_token``), so the live and
    unpickled FromArray dedupe to one instance and the divergence never shows.
    The source below tokenizes off ``os.getpid()`` — identical within a process,
    different across processes — i.e. exactly the cross-process non-determinism
    that an xarray/icechunk lazy-indexing adapter exhibits in the wild. Without
    the FromArray token cache, the scheduler re-tokenizes the source under its
    own pid and the keys diverge."""
    import subprocess
    import sys

    import cloudpickle

    # Self-contained (no inheritance from a module-level helper) so cloudpickle
    # serializes it by value — the subprocess needs only dask_array + cloudpickle.
    class _PidTokenSource:
        def __init__(self, base):
            self._base = base

        def __dask_tokenize__(self):
            import os

            return ("pid-token", os.getpid())

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

    base = np.arange(120.0).reshape(8, 15)
    arr = da.from_array(_PidTokenSource(base), chunks=(3, 5)).mean()
    client_keys = [str(k) for k in arr.__frisky_output_keys__()]
    blob = cloudpickle.dumps(arr)

    script = (
        "import sys, cloudpickle; arr = cloudpickle.loads(sys.stdin.buffer.read()); "
        "print(chr(10).join(str(k) for k in arr.__frisky_output_keys__()))"
    )
    proc = subprocess.run([sys.executable, "-c", script], input=blob, capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()[-2000:]
    scheduler_keys = [line for line in proc.stdout.decode().splitlines() if line]
    assert client_keys == scheduler_keys
