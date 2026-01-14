"""Tests for xarray ChunkManager integration.

This module tests the DaskArrayExprManager which registers as the "dask"
chunk manager, replacing xarray's built-in DaskManager. This allows it to
handle both dask_array.Array and legacy dask.array.Array types.
"""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import dask_array as da
from dask_array._xarray import DaskArrayExprManager


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
        """Test that manager also recognizes legacy dask.array.Array."""
        import dask.array as legacy_da

        manager = DaskArrayExprManager()
        arr = legacy_da.ones((10, 10), chunks=(5, 5))
        assert manager.is_chunked_array(arr)

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

    def test_dataarray_rechunk(self):
        """Test rechunking a DataArray."""
        arr = da.ones((10, 20), chunks=(5, 10))
        da_xr = xr.DataArray(arr, dims=["x", "y"])

        rechunked = da_xr.chunk({"x": 10, "y": 5})
        assert rechunked.chunks == ((10,), (5, 5, 5, 5))

    def test_dataset_from_dask_arrays(self):
        """Test creating a Dataset from dask_array.Arrays."""
        arr1 = da.ones((10, 20), chunks=(5, 10))
        arr2 = da.zeros((10, 20), chunks=(5, 10))

        ds = xr.Dataset({
            "var1": (["x", "y"], arr1),
            "var2": (["x", "y"], arr2),
        })

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
