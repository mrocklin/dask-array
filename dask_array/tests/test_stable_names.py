"""Collection identity is stable: names and keys survive optimization.

Expressions are content-addressed and freely renamed by simplify/lower/fuse.
Collections are not: ``Array._name``/``__dask_keys__`` are the *raw* root
expression's, assigned at construction, and materialization pins the graph's
output keys back to them (``RootAlias``). Everything here guards that
contract — cheap names, keys == graph outputs on every entry point, and
name-preserving persist.
"""

from __future__ import annotations

import dask
import numpy as np
import pytest

import dask_array as da
from dask_array._test_utils import assert_eq


@pytest.fixture
def arr():
    x = da.ones((10, 10), chunks=(5, 5)) + 1
    return (x * 2).sum(axis=0)


def test_name_and_keys_are_cheap(arr):
    """Naming a collection must not lower it — a ``.name`` access that lowers
    makes construction loops O(tree^2)."""
    assert arr.name == arr.expr._name
    arr.__dask_keys__()
    assert "_lowered_expr" not in vars(arr)


def test_graph_produces_advertised_keys(arr):
    graph = arr.__dask_graph__()
    for key in arr.__dask_keys__():
        assert key in graph


def test_zero_dim_keys(arr):
    s = arr.sum()
    assert s.__dask_keys__() == [(s.name,)]
    assert s.__dask_keys__()[0] in s.__dask_graph__()


def test_compute_entry_points_agree(arr):
    expected = np.full((10,), 40.0)
    (via_dask,) = dask.compute(arr)
    np.testing.assert_array_equal(via_dask, expected)
    np.testing.assert_array_equal(arr.compute(), expected)


def test_method_compute_is_fused(arr):
    # Array.compute routes through the materialized expression: fused, plus
    # one alias per output block pinning the collection's keys.
    graph = arr._pinned().__dask_graph__()
    prefixes = {k[0].rsplit("-", 1)[0] for k in graph}
    assert "sum-mul-add-ones" in prefixes  # elemwise chain fused into the reduction


def test_persist_preserves_name_and_keys(arr):
    p = arr.persist(scheduler="threads")
    assert p.name == arr.name
    assert p.__dask_keys__() == arr.__dask_keys__()
    assert_eq(p, arr)


def test_generic_dask_persist_preserves_name(arr):
    # dask.persist on the raw collection takes dask's generic optimizer path;
    # the rebuild locates blocks in the persisted layer by block id.
    (p,) = dask.persist(arr, scheduler="threads")
    assert p.name == arr.name
    assert p.__dask_keys__() == arr.__dask_keys__()
    assert_eq(p, arr)


def test_persist_twice_is_stable(arr):
    p = arr.persist(scheduler="threads")
    p2 = p.persist(scheduler="threads")
    assert p2.name == p.name == arr.name
    assert_eq(p2, arr)


def test_persisted_collection_composes(arr):
    p = arr.persist(scheduler="threads")
    np.testing.assert_array_equal((p + 1).compute(), np.full((10,), 41.0))
    assert (p.sum().compute() == arr.sum().compute()).all()


def test_compute_after_persist_of_same_expression(arr):
    """Persisting must not poison shared caches: materializing a new
    expression over the same raw subtree afterwards has to lower cleanly
    (regression: a pinned node cached under its raw name got spliced into the
    middle of later trees)."""
    arr.persist(scheduler="threads")
    assert arr.sum().compute() == 400.0


def test_optimization_changing_chunks_is_bridged():
    """A rewrite may emit different output chunking (sliding-window
    reductions avoid a padding rechunk); materialization must still deliver
    the advertised chunks."""
    data = np.arange(96.0 * 8).reshape(96, 8)
    x = da.from_array(data, chunks=(24, 4))
    windowed = da.sliding_window_view(x, window_shape=72, axis=0)
    result = windowed.var(axis=-1)
    # raw metadata and optimized plan disagree on purpose
    assert result.chunks != result.expr.simplify().chunks
    graph = result.__dask_graph__()
    from dask.core import flatten

    for key in flatten(result.__dask_keys__()):
        assert key in graph
    expected = np.lib.stride_tricks.sliding_window_view(data, 72, axis=0).var(axis=-1)
    assert_eq(result, expected)


def test_dask_optimize_roundtrip():
    x = da.from_array(np.arange(12), chunks=3).rechunk((4,))
    (optimized,) = dask.optimize(x)
    assert_eq(optimized, np.arange(12))
    assert optimized.chunks == ((4, 4, 4),)
