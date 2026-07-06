"""Tests for xarray ChunkManager integration.

This module tests the DaskArrayExprManager which registers as the "dask"
chunk manager, replacing xarray's built-in DaskManager. This allows it to
handle dask_array.Array types.
"""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import dask_array as da
from dask_array._rechunk import TasksRechunk
from dask_array._xarray import DaskArrayExprManager


def _xarray_sliding_window_uses_chunk_manager():
    import inspect
    import xarray.compat.dask_array_compat as compat

    try:
        source = inspect.getsource(compat.sliding_window_view)
    except OSError:
        return False
    return "get_chunked_array_type" in source and ".array_api" in source


requires_xarray_sliding_window_chunk_manager = pytest.mark.skipif(
    not _xarray_sliding_window_uses_chunk_manager(),
    reason="requires xarray sliding_window_view dispatch through the chunk manager array API",
)


def _contains_expr_type(expr, typ):
    if isinstance(expr, typ):
        return True
    return any(_contains_expr_type(dep, typ) for dep in expr.dependencies())


def test_public_xarray_api_available_from_package():
    from dask_array import xarray as da_xarray

    assert da.xarray is da_xarray
    assert callable(da_xarray.register)
    assert callable(da_xarray.isactive)


def test_public_xarray_register_and_isactive(monkeypatch):
    from xarray.namedarray.parallelcompat import list_chunkmanagers

    managers = list_chunkmanagers()
    monkeypatch.setitem(managers, "dask", object())

    assert not da.xarray.isactive()

    da.xarray.register()

    assert da.xarray.isactive()
    assert isinstance(managers["dask"], DaskArrayExprManager)


@requires_xarray_sliding_window_chunk_manager
def test_xarray_rolling_full_time_chunk_avoids_padding_rechunk():
    da.xarray.register()
    x = xr.DataArray(
        da.ones((100, 6, 8), chunks=(100, 3, 4)),
        dims=("time", "latitude", "longitude"),
    )

    result = (x > 0).rolling(time=72, min_periods=72).sum().max("time")
    optimized = result.data.expr.optimize()

    assert not _contains_expr_type(optimized, TasksRechunk)
    np.testing.assert_allclose(result.compute().values, np.full((6, 8), 72.0))


class TestDaskArrayExprManager:
    """Tests for the DaskArrayExprManager class."""

    def test_init(self):
        manager = DaskArrayExprManager()
        assert manager.array_cls is da.Array
        assert manager.available is True

    def test_is_chunked_array(self):
        manager = DaskArrayExprManager()
        arr = da.ones((10, 10), chunks=(5, 5))
        assert manager.is_chunked_array(arr)
        assert not manager.is_chunked_array(np.ones((10, 10)))

    def test_is_chunked_array_legacy_dask(self):
        """Test that manager rejects legacy dask.array.Array."""
        import dask.array as legacy_da

        manager = DaskArrayExprManager()
        arr = legacy_da.ones((10, 10), chunks=(5, 5))
        assert not manager.is_chunked_array(arr)

    def test_chunks(self):
        manager = DaskArrayExprManager()
        arr = da.ones((10, 10), chunks=(5, 5))
        assert manager.chunks(arr) == ((5, 5), (5, 5))

    def test_normalize_chunks(self):
        manager = DaskArrayExprManager()
        result = manager.normalize_chunks((5, 5), shape=(10, 10))
        assert result == ((5, 5), (5, 5))

    def test_from_array(self):
        manager = DaskArrayExprManager()
        x = np.arange(100).reshape(10, 10)
        arr = manager.from_array(x, chunks=(5, 5))
        assert isinstance(arr, da.Array)
        assert arr.chunks == ((5, 5), (5, 5))

    def test_from_array_lazy_indexing_adapter_uses_numpy_meta(self):
        from xarray.core.indexing import (
            ImplicitToExplicitIndexingAdapter,
            LazilyIndexedArray,
            NumpyIndexingAdapter,
            OuterIndexer,
        )

        manager = DaskArrayExprManager()
        base = NumpyIndexingAdapter(np.ones((2, 3)))
        lazy = LazilyIndexedArray(base, OuterIndexer((slice(None), slice(None))))
        adapter = ImplicitToExplicitIndexingAdapter(lazy, OuterIndexer)

        arr = manager.from_array(adapter, chunks=(1, 3))
        out = arr * 1.0

        assert isinstance(arr.expr._meta, np.ndarray)
        assert isinstance(out.expr._meta, np.ndarray)
        assert out.expr._meta.shape == (0, 0)

    def test_compute(self):
        manager = DaskArrayExprManager()
        arr = da.ones((10, 10), chunks=(5, 5))
        (result,) = manager.compute(arr)
        np.testing.assert_array_equal(result, np.ones((10, 10)))

    def test_array_api(self):
        manager = DaskArrayExprManager()
        api = manager.array_api
        assert hasattr(api, "ones")
        assert hasattr(api, "zeros")
        assert hasattr(api, "full_like")


class TestXarrayIntegration:
    """Tests for xarray integration with DaskArrayExprManager."""

    def test_manager_discoverable(self):
        """Test that the manager is discoverable via xarray as 'dask'."""
        from xarray.namedarray.parallelcompat import list_chunkmanagers

        managers = list_chunkmanagers()
        assert "dask" in managers
        # Our manager should be the one registered (replaces built-in)
        assert isinstance(managers["dask"], DaskArrayExprManager)

    def test_get_chunked_array_type_selects_manager_once(self):
        """Test xarray sees dask_array.Array through one chunk manager."""
        from xarray.namedarray.parallelcompat import get_chunked_array_type

        arr = da.ones((10, 10), chunks=(5, 5))

        assert isinstance(get_chunked_array_type(arr), DaskArrayExprManager)

    def test_get_chunked_array_type_survives_legacy_dask_import(self):
        """Importing legacy dask.array must not reclaim xarray's manager."""
        import subprocess
        import sys

        code = (
            "import dask_array as da\n"
            "import dask.array\n"
            "from dask_array._xarray import DaskArrayExprManager\n"
            "from xarray.namedarray.parallelcompat import get_chunked_array_type, list_chunkmanagers\n"
            "arr = da.ones((10, 10), chunks=(5, 5))\n"
            "assert isinstance(list_chunkmanagers()['dask'], DaskArrayExprManager)\n"
            "assert isinstance(get_chunked_array_type(arr), DaskArrayExprManager)\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr

    def test_dask_new_collection_roundtrip(self):
        """Test Dask can rebuild dask_array.Array from its expression."""
        from dask._collections import new_collection

        arr = da.arange(6, chunks=(3,)) + 1

        rebuilt = new_collection(arr.expr)

        assert isinstance(rebuilt, da.Array)
        np.testing.assert_array_equal(rebuilt.compute(), np.arange(6) + 1)

    def test_xarray_register_restores_dask_collection_dispatch_after_legacy_import(self):
        """Test Dask rebuild dispatch survives importing legacy dask.array."""
        import dask.array  # noqa: F401
        from dask._collections import new_collection

        da.xarray.register()
        arr = da.arange(6, chunks=(3,)) + 1

        rebuilt = new_collection(arr.expr)

        assert isinstance(rebuilt, da.Array)
        np.testing.assert_array_equal(rebuilt.compute(), np.arange(6) + 1)

    def test_dataarray_from_dask_array(self):
        """Test creating a DataArray from a dask_array.Array."""
        arr = da.ones((10, 20), chunks=(5, 10))
        da_xr = xr.DataArray(arr, dims=["x", "y"])

        assert da_xr.shape == (10, 20)
        assert da_xr.chunks == ((5, 5), (10, 10))

    def test_dataarray_compute(self):
        """Test computing a DataArray backed by dask_array.Array."""
        arr = da.arange(100, chunks=25).reshape(10, 10)
        da_xr = xr.DataArray(arr, dims=["x", "y"])

        result = da_xr.compute()
        expected = np.arange(100).reshape(10, 10)
        np.testing.assert_array_equal(result.values, expected)

    def test_dataarray_operations(self):
        """Test that DataArray operations work with dask_array.Array."""
        arr = da.ones((10, 20), chunks=(5, 10))
        da_xr = xr.DataArray(arr, dims=["x", "y"])

        # Test arithmetic
        result = (da_xr + 1).compute()
        np.testing.assert_array_equal(result.values, np.full((10, 20), 2.0))

        # Test reduction
        result = da_xr.sum().compute()
        assert result.values == 200.0

    @requires_xarray_sliding_window_chunk_manager
    def test_dataarray_rolling_mean(self):
        arr = da.from_array(np.arange(12.0), chunks=4)
        da_xr = xr.DataArray(arr, dims=["time"])

        result = da_xr.rolling(time=3, center=True).mean()

        assert isinstance(result.data, da.Array)
        expected = np.array([np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan])
        np.testing.assert_allclose(result.compute().values, expected)

    @requires_xarray_sliding_window_chunk_manager
    def test_dataarray_rolling_construct_multi_axis(self):
        data = np.arange(4 * 6.0).reshape(4, 6)
        da_xr = xr.DataArray(
            da.from_array(data, chunks=(2, 3)),
            dims=["time", "x"],
        )
        eager = xr.DataArray(data, dims=["time", "x"])

        result = da_xr.rolling(time=3, x=2, center=True).construct(time="time_window", x="x_window")
        expected = eager.rolling(time=3, x=2, center=True).construct(time="time_window", x="x_window")

        assert isinstance(result.data, da.Array)
        np.testing.assert_allclose(result.compute().values, expected.values)

    def test_dataarray_rechunk(self):
        """Test rechunking a DataArray."""
        arr = da.ones((10, 20), chunks=(5, 10))
        da_xr = xr.DataArray(arr, dims=["x", "y"])

        rechunked = da_xr.chunk({"x": 10, "y": 5})
        assert rechunked.chunks == ((10,), (5, 5, 5, 5))

    def test_dataarray_cftime_auto_rechunk(self):
        pytest.importorskip("cftime")

        years = np.arange(2000, 2120)
        dates = xr.date_range(
            start=f"{years[0]}-01-01",
            end=f"{years[-1]}-12-31",
            freq="1YE",
            use_cftime=True,
        )
        data = np.tile(dates.values, (10, 1))
        da_xr = xr.DataArray(
            data,
            dims=["x", "t"],
            coords={"x": np.arange(10), "t": dates},
        ).chunk({"x": 4, "t": 5})

        rechunked = da_xr.chunk("auto")

        assert isinstance(rechunked.data, da.Array)
        assert rechunked.chunks == ((10,), (120,))
        np.testing.assert_array_equal(rechunked.compute().values, data)

    def test_dataset_from_dask_arrays(self):
        """Test creating a Dataset from dask_array.Arrays."""
        arr1 = da.ones((10, 20), chunks=(5, 10))
        arr2 = da.zeros((10, 20), chunks=(5, 10))

        ds = xr.Dataset(
            {
                "var1": (["x", "y"], arr1),
                "var2": (["x", "y"], arr2),
            }
        )

        assert ds["var1"].shape == (10, 20)
        assert ds["var2"].shape == (10, 20)

        result = ds.compute()
        np.testing.assert_array_equal(result["var1"].values, np.ones((10, 20)))
        np.testing.assert_array_equal(result["var2"].values, np.zeros((10, 20)))

    def test_apply_ufunc(self):
        """Test xr.apply_ufunc with dask_array.Array."""
        arr = da.ones((10, 20), chunks=(5, 10))
        da_xr = xr.DataArray(arr, dims=["x", "y"])

        result = xr.apply_ufunc(
            lambda x: x * 2,
            da_xr,
            dask="parallelized",
            output_dtypes=[float],
        )

        assert isinstance(result.data, da.Array)

        computed = result.compute()
        np.testing.assert_array_equal(computed.values, np.full((10, 20), 2.0))
