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


# Both xarray and dask_array register a "dask" entry point under
# xarray.chunkmanagers, and whichever enumerates *last* wins the in-flight
# discovery dict -- a per-environment installation detail. The discovery tests
# below patch parallelcompat.entry_points to force each order explicitly
# instead of inheriting install luck. See dask_array/_xarray.py's docstring
# for the guarantees being pinned here.
_FORCE_ENTRY_POINT_ORDER = """
import importlib.metadata

import xarray.namedarray.parallelcompat as pc

_eps = list(importlib.metadata.entry_points(group="xarray.chunkmanagers"))
_ours = [ep for ep in _eps if ep.value.startswith("dask_array")]
_builtin = [ep for ep in _eps if not ep.value.startswith("dask_array")]
assert _ours and _builtin
pc.entry_points = lambda group: {order}  # last entry wins the in-flight dict
"""


def test_xarray_entry_point_discovery_engages_dask_array():
    # Without importing dask_array or calling register(), a chunked xarray
    # operation must discover our "dask" chunkmanager through the
    # xarray.chunkmanagers entry point and produce dask_array-backed data.
    # Forced favorable order: our entry point enumerates last.
    pytest.importorskip("xarray")
    code = f"""
import sys

import numpy as np
import xarray as xr
{_FORCE_ENTRY_POINT_ORDER.format(order="_builtin + _ours")}
assert "dask_array" not in sys.modules

ds = xr.Dataset({{"a": ("x", np.arange(10.0))}}).chunk({{"x": 5}})

import dask_array as da

assert isinstance(ds["a"].data, da.Array), type(ds["a"].data)
assert da.xarray.isactive()
assert ds["a"].sum().compute().item() == 45.0
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_xarray_discovery_adverse_order_first_op_converts_on_next_use():
    # dask_array imported first (no register()), and the built-in "dask"
    # entry point enumerates last: the single op that triggered discovery
    # gets legacy dask.array-backed data (it ran against the in-flight,
    # un-pinned registry). The pinned manager claims that legacy-backed
    # object and converts it on its next use -- values, dtype, and chunks
    # intact -- and every later chunked op engages dask_array directly.
    # register() before first use avoids even the legacy-backed first result.
    pytest.importorskip("xarray")
    code = f"""
import dask_array as da
import numpy as np
import xarray as xr
{_FORCE_ENTRY_POINT_ORDER.format(order="_ours + _builtin")}
ds = xr.Dataset({{"a": ("x", np.arange(10.0))}}).chunk({{"x": 5}})

# The op that triggered discovery saw the in-flight, un-pinned registry.
assert not isinstance(ds["a"].data, da.Array), type(ds["a"].data)
assert type(ds["a"].data).__module__.startswith("dask."), type(ds["a"].data)

# ...but discovery imported dask_array._xarray, which pinned the cache...
from dask_array._xarray import DaskArrayExprManager
from xarray.namedarray.parallelcompat import list_chunkmanagers

assert isinstance(list_chunkmanagers()["dask"], DaskArrayExprManager)

# ...and the pinned manager claims the legacy-backed object: the next uses
# convert it instead of failing.
np.testing.assert_array_equal(ds["a"].compute().values, np.arange(10.0))

rechunked = ds.chunk({{"x": 2}})
assert isinstance(rechunked["a"].data, da.Array), type(rechunked["a"].data)
assert rechunked["a"].data.chunks == ((2, 2, 2, 2, 2),)
np.testing.assert_array_equal(rechunked["a"].compute().values, np.arange(10.0))

# Plain arithmetic never consults the chunk manager (xarray applies the
# operator to the duck array directly), so it stays legacy-backed -- but
# compute on the result still converts and returns correct values.
total = (ds["a"] + 1).sum()
assert total.compute().item() == 55.0

# Every later chunked op engages dask_array directly.
ds2 = xr.Dataset({{"a": ("x", np.arange(10.0))}}).chunk({{"x": 5}})
assert isinstance(ds2["a"].data, da.Array), type(ds2["a"].data)
assert ds2["a"].sum().compute().item() == 45.0
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_xarray_discovery_adverse_order_query_planning_converts():
    # Same adverse window, but with dask's array query-planning enabled: the
    # built-in DaskManager then produces dask.array._array_expr collections,
    # which are NOT dask.array.core.Array instances. The structural claim in
    # is_chunked_array must recognise that flavor too, and conversion must
    # round-trip it.
    pytest.importorskip("xarray")
    code = f"""
import dask

dask.config.set({{"array.query-planning": True}})

import dask_array as da
import numpy as np
import xarray as xr
{_FORCE_ENTRY_POINT_ORDER.format(order="_ours + _builtin")}
ds = xr.Dataset({{"a": ("x", np.arange(10.0))}}).chunk({{"x": 5}})

# The first op is backed by the query-planning legacy collection.
assert not isinstance(ds["a"].data, da.Array), type(ds["a"].data)
assert type(ds["a"].data).__module__.startswith("dask.array"), type(ds["a"].data)

np.testing.assert_array_equal(ds["a"].compute().values, np.arange(10.0))

rechunked = ds.chunk({{"x": 2}})
assert isinstance(rechunked["a"].data, da.Array), type(rechunked["a"].data)
assert rechunked["a"].data.chunks == ((2, 2, 2, 2, 2),)
np.testing.assert_array_equal(rechunked["a"].compute().values, np.arange(10.0))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_xarray_chunkmanager_cache_clear_keeps_objects_usable():
    # xarray test suites call list_chunkmanagers.cache_clear(). Re-enumeration
    # then hands the "dask" slot to whichever entry point enumerates last (our
    # module is already imported, so its pin does not rerun). Existing chunked
    # objects must stay usable under either winner: ours claims-and-converts
    # both array types; the built-in claims our arrays as duck dask arrays
    # and computes/rechunks them through their own dask-collection interface.
    pytest.importorskip("xarray")
    for order in ("_builtin + _ours", "_ours + _builtin"):
        code = f"""
import numpy as np
import xarray as xr
{_FORCE_ENTRY_POINT_ORDER.format(order=order)}
import dask_array as da

da.xarray.register()
ds = xr.Dataset({{"a": ("x", np.arange(10.0))}}).chunk({{"x": 5}})
assert isinstance(ds["a"].data, da.Array), type(ds["a"].data)

from xarray.namedarray.parallelcompat import list_chunkmanagers

list_chunkmanagers.cache_clear()

np.testing.assert_array_equal(ds["a"].compute().values, np.arange(10.0))
assert ds.chunk({{"x": 2}})["a"].data.chunks == ((2, 2, 2, 2, 2),)
"""
        subprocess.run([sys.executable, "-c", code], check=True)


def test_import_after_xarray_pins_manager_eagerly():
    # When xarray is already imported, `import dask_array` pins our manager
    # immediately, so even the eventual first chunked op engages dask_array
    # regardless of entry-point enumeration order (forced adverse here).
    pytest.importorskip("xarray")
    code = f"""
import numpy as np
import xarray as xr
{_FORCE_ENTRY_POINT_ORDER.format(order="_ours + _builtin")}
import dask_array as da

assert da.xarray.isactive()

ds = xr.Dataset({{"a": ("x", np.arange(10.0))}}).chunk({{"x": 5}})
assert isinstance(ds["a"].data, da.Array), type(ds["a"].data)
assert ds["a"].sum().compute().item() == 45.0
"""
    subprocess.run([sys.executable, "-c", code], check=True)
