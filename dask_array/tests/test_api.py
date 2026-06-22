from __future__ import annotations

import subprocess
import sys

import numpy as np

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
