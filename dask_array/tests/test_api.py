from __future__ import annotations

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
    assert callable(da.to_hdf5)
    assert callable(da.from_tiledb)
    assert callable(da.to_tiledb)


def test_top_level_optimize_collection():
    x = da.arange(6, chunks=3) + 1

    result = da.optimize(x)

    assert isinstance(result, da.Array)
    np.testing.assert_array_equal(result.compute(), np.arange(6) + 1)


def test_random_star_exports_legacy_wrappers():
    namespace = {}
    exec("from dask_array.random import *", namespace)

    assert callable(namespace["normal"])
    assert callable(namespace["random"])
    assert callable(namespace["randint"])
    assert callable(namespace["standard_normal"])
