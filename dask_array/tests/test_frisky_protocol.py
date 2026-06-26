from __future__ import annotations

import importlib.util
import sys

import dask
from dask.core import flatten
from dask.local import get_sync
import numpy as np
import pytest

import dask_array as da
from dask_array._overlap import overlap_internal


def test_dask_graph_does_not_import_frisky_modules():
    for name in list(sys.modules):
        if name == "dask_array._rust" or name.startswith("dask_array._frisky"):
            sys.modules.pop(name)

    x = da.ones((4, 4), chunks=(2, 2)) + 1
    assert "dask_array._frisky" not in sys.modules
    assert "dask_array._rust" not in sys.modules

    x.__dask_graph__()

    assert "dask_array._frisky" not in sys.modules
    assert "dask_array._rust" not in sys.modules


def test_frisky_graph_imports_frisky_modules():
    x = da.ones((4, 4), chunks=(2, 2)) + 1
    records = x.__frisky_graph__()

    assert records
    assert "dask_array._frisky" in sys.modules


def test_sliding_window_overlap_uses_binary_records():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.sliding_window_view(da.ones(12, chunks=4), 3).sum()

    chunks, records = x.__frisky_records_chunks__()

    assert chunks
    assert records == []


def test_overlap_native_layer_matches_legacy_graph():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.from_array(np.arange(4 * 6).reshape(4, 6), chunks=(2, 3))
    y = overlap_internal(x, {0: 1, 1: (0, 2)})
    expr = y.expr
    dep_graph = dict(expr.array.__dask_graph__())
    legacy_graph = expr._layer()
    native_graph = expr._frisky_layer().to_dask_graph()

    for key in flatten(expr.__dask_keys__()):
        expected = get_sync({**dep_graph, **legacy_graph}, key)
        actual = get_sync({**dep_graph, **native_graph}, key)
        np.testing.assert_array_equal(actual, expected)


def _constant_block():
    return np.full((2, 2), 7.0)


def _constant_frisky_graph(self, seen=None):
    return [(self.__frisky_output_keys__()[0], _constant_block, (), {}, [])]


def test_frisky_scheduler_uses_frisky_graph(array_scheduler, monkeypatch):
    if array_scheduler != "frisky":
        pytest.skip("requires --scheduler=frisky")

    x = da.ones((2, 2), chunks=(2, 2)) + 1
    monkeypatch.setattr(type(x), "__frisky_graph__", _constant_frisky_graph)

    (result,) = dask.compute(x)

    np.testing.assert_array_equal(result, np.full((2, 2), 7.0))
