from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import dask_array as da


def test_top_level_compatibility_exports():
    assert da.newaxis is None
    assert np.isnan(da.nan)
    assert da.inf == np.inf
    assert da.pi == np.pi
    assert da.float64 is np.float64
    assert da.int64 is np.int64

    assert callable(da.compute)
    assert callable(da.optimize)
    assert callable(da.register_chunk_type)
    assert callable(da.sliding_window_view)
    assert callable(da.to_hdf5)
    assert callable(da.from_tiledb)
    assert callable(da.to_tiledb)


def test_top_level_optimize_collection():
    x = da.arange(6, chunks=3) + 1

    result = da.optimize(x)

    assert isinstance(result, da.Array)
    np.testing.assert_array_equal(result.compute(), np.arange(6) + 1)


def test_top_level_sliding_window_view():
    x = np.arange(6)

    result = da.sliding_window_view(da.from_array(x, chunks=3), 3)

    assert result.shape == (4, 3)
    assert result.chunks == ((3, 1), (3,))
    np.testing.assert_array_equal(result.compute(), np.lib.stride_tricks.sliding_window_view(x, 3))


def test_random_star_exports_legacy_wrappers():
    namespace = {}
    exec("from dask_array.random import *", namespace)

    assert callable(namespace["normal"])
    assert callable(namespace["random"])
    assert callable(namespace["randint"])
    assert callable(namespace["standard_normal"])


def test_xarray_public_api_when_xarray_is_unavailable():
    code = """
import builtins

real_import = builtins.__import__


def blocked_import(name, *args, **kwargs):
    if name == "xarray" or name.startswith("xarray."):
        raise ImportError("blocked xarray")
    return real_import(name, *args, **kwargs)


builtins.__import__ = blocked_import

import dask_array as da

assert hasattr(da, "xarray")
assert da.xarray.isactive() is False

try:
    da.xarray.register()
except ImportError:
    pass
else:
    raise AssertionError("register() should raise ImportError without xarray")
"""

    subprocess.run([sys.executable, "-c", code], check=True)


def test_plain_import_does_not_load_xarray_or_pandas():
    # Hard requirement: the xarray chunk-manager integration is lazy, so
    # importing dask_array must not pay for (or fail without) xarray/pandas.
    code = """
import sys
import dask_array

assert "xarray" not in sys.modules, "import dask_array pulled in xarray"
assert "pandas" not in sys.modules, "import dask_array pulled in pandas"
assert "dask_array._xarray" not in sys.modules

# isactive() is a passive probe: it must answer without importing xarray.
assert dask_array.xarray.isactive() is False
assert "xarray" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_explicit_xarray_register():
    pytest.importorskip("xarray")
    code = """
import dask_array as da

da.xarray.register()
assert da.xarray.isactive()

from dask_array._xarray import DaskArrayExprManager
from xarray.namedarray.parallelcompat import list_chunkmanagers

assert isinstance(list_chunkmanagers()["dask"], DaskArrayExprManager)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_xarray_entry_point_discovery_engages_dask_array():
    # Without importing dask_array or calling register(), a chunked xarray
    # operation must discover our "dask" chunkmanager through the
    # xarray.chunkmanagers entry point and produce dask_array-backed data.
    pytest.importorskip("xarray")
    code = """
import sys

import numpy as np
import xarray as xr

assert "dask_array" not in sys.modules

ds = xr.Dataset({"a": ("x", np.arange(10.0))}).chunk({"x": 5})

import dask_array as da

assert isinstance(ds["a"].data, da.Array), type(ds["a"].data)
assert da.xarray.isactive()
assert ds["a"].sum().compute().item() == 45.0
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_xarray_discovery_engages_dask_array_after_plain_import():
    # Reverse import order: dask_array imported first (without register()),
    # xarray discovery still picks our manager at first chunked-array use.
    pytest.importorskip("xarray")
    code = """
import dask_array as da
import numpy as np
import xarray as xr

ds = xr.Dataset({"a": ("x", np.arange(10.0))}).chunk({"x": 5})

assert isinstance(ds["a"].data, da.Array), type(ds["a"].data)
assert da.xarray.isactive()
"""
    subprocess.run([sys.executable, "-c", code], check=True)
