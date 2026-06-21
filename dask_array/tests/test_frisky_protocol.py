from __future__ import annotations

import sys

import dask
import numpy as np
import pytest

import dask_array as da


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


def _constant_block():
    return np.full((2, 2), 7.0)


class _ConstantFriskyGraph:
    def __init__(self, output_key):
        self.output_key = output_key

    def __call__(self, seen=None):
        return [(self.output_key, _constant_block, (), {}, [])]


def test_frisky_scheduler_uses_frisky_graph(array_scheduler, monkeypatch):
    if array_scheduler != "frisky":
        pytest.skip("requires --scheduler=frisky")

    x = da.ones((2, 2), chunks=(2, 2)) + 1
    monkeypatch.setattr(x, "__frisky_graph__", _ConstantFriskyGraph(x.__frisky_output_keys__()[0]))

    (result,) = dask.compute(x)

    np.testing.assert_array_equal(result, np.full((2, 2), 7.0))
