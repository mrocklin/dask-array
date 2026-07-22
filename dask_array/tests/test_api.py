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


# register() is normally called before any chunked work, but it is legal to
# call it late -- after xarray has already produced legacy dask.array-backed
# objects. Our manager then owns the "dask" slot and must claim those objects
# and convert them on use, or they would become unusable through xarray. See
# dask_array/_xarray.py's docstring for the guarantees being pinned here.
def test_xarray_register_after_legacy_objects_exist_converts_on_next_use():
    pytest.importorskip("xarray")
    code = """
import dask_array as da
import numpy as np
import xarray as xr

# Chunked before register(): xarray's built-in manager makes this legacy.
ds = xr.Dataset({"a": ("x", np.arange(10.0))}).chunk({"x": 5})
assert not isinstance(ds["a"].data, da.Array), type(ds["a"].data)
assert type(ds["a"].data).__module__.startswith("dask."), type(ds["a"].data)

da.xarray.register()

from dask_array._xarray import DaskArrayExprManager
from xarray.namedarray.parallelcompat import list_chunkmanagers

assert isinstance(list_chunkmanagers()["dask"], DaskArrayExprManager)

# Our manager claims the legacy-backed object: later uses convert it
# instead of failing.
np.testing.assert_array_equal(ds["a"].compute().values, np.arange(10.0))

rechunked = ds.chunk({"x": 2})
assert isinstance(rechunked["a"].data, da.Array), type(rechunked["a"].data)
assert rechunked["a"].data.chunks == ((2, 2, 2, 2, 2),)
np.testing.assert_array_equal(rechunked["a"].compute().values, np.arange(10.0))

# Plain arithmetic never consults the chunk manager (xarray applies the
# operator to the duck array directly), so it stays legacy-backed -- but
# compute on the result still converts and returns correct values.
total = (ds["a"] + 1).sum()
assert total.compute().item() == 55.0

# Every later chunked op engages dask_array directly.
ds2 = xr.Dataset({"a": ("x", np.arange(10.0))}).chunk({"x": 5})
assert isinstance(ds2["a"].data, da.Array), type(ds2["a"].data)
assert ds2["a"].sum().compute().item() == 45.0
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_xarray_register_after_legacy_query_planning_objects_converts():
    # Same late-register scenario, but with dask's array query-planning
    # enabled: the built-in DaskManager then produces dask.array._array_expr
    # collections, which are NOT dask.array.core.Array instances. The
    # structural claim in is_chunked_array must recognise that flavor too, and
    # conversion must round-trip it.
    pytest.importorskip("xarray")
    code = """
import dask

dask.config.set({"array.query-planning": True})

import dask_array as da
import numpy as np
import xarray as xr

ds = xr.Dataset({"a": ("x", np.arange(10.0))}).chunk({"x": 5})

# Backed by the query-planning legacy collection.
assert not isinstance(ds["a"].data, da.Array), type(ds["a"].data)
assert type(ds["a"].data).__module__.startswith("dask.array"), type(ds["a"].data)

da.xarray.register()

np.testing.assert_array_equal(ds["a"].compute().values, np.arange(10.0))

rechunked = ds.chunk({"x": 2})
assert isinstance(rechunked["a"].data, da.Array), type(rechunked["a"].data)
assert rechunked["a"].data.chunks == ((2, 2, 2, 2, 2),)
np.testing.assert_array_equal(rechunked["a"].compute().values, np.arange(10.0))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_xarray_chunkmanager_cache_clear_keeps_objects_usable():
    # xarray test suites call list_chunkmanagers.cache_clear(). Re-enumeration
    # rebuilds the registry from entry points, and we ship none, so the "dask"
    # slot reverts to the built-in manager. Arrays created while we were
    # registered must stay usable: the built-in claims them as duck dask arrays
    # and computes/rechunks them through the dask-collection interface.
    # Calling register() again re-takes the slot.
    pytest.importorskip("xarray")
    code = """
import numpy as np
import xarray as xr
import dask_array as da

da.xarray.register()
ds = xr.Dataset({"a": ("x", np.arange(10.0))}).chunk({"x": 5})
assert isinstance(ds["a"].data, da.Array), type(ds["a"].data)

from xarray.namedarray.parallelcompat import list_chunkmanagers

list_chunkmanagers.cache_clear()
assert not da.xarray.isactive()

np.testing.assert_array_equal(ds["a"].compute().values, np.arange(10.0))
assert ds.chunk({"x": 2})["a"].data.chunks == ((2, 2, 2, 2, 2),)

da.xarray.register()
assert da.xarray.isactive()
assert isinstance(ds.chunk({"x": 2})["a"].data, da.Array)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
