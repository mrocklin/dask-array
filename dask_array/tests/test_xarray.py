"""Tests for xarray ChunkManager integration.

This module tests the DaskArrayExprManager, which takes over the "dask" chunk
manager slot from xarray's built-in DaskManager so that xarray produces
dask_array.Array types. The takeover happens only on an explicit
dask_array.xarray.register(), so most tests here call it first.
"""

import numpy as np
import pytest
import dask

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


def test_no_chunkmanager_entry_point():
    """We ship no "xarray.chunkmanagers" entry point.

    An entry point would activate on install, so anyone with dask-array in
    their environment -- including as a transitive dependency they never
    import -- would silently get expression-backed arrays out of xarray.
    """
    import importlib.metadata

    ours = [
        ep for ep in importlib.metadata.entry_points(group="xarray.chunkmanagers") if ep.value.startswith("dask_array")
    ]
    assert ours == []


# Registration is process-global, and other tests in this module call
# register(), so the "we did not register" assertions need a fresh interpreter.
def _run_isolated(body):
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "-c", body], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_install_alone_leaves_xarray_on_builtin_manager():
    _run_isolated(
        "import numpy as np, xarray as xr\n"
        "import importlib.util\n"
        "assert importlib.util.find_spec('dask_array') is not None, 'test is vacuous'\n"
        "ds = xr.Dataset({'a': ('x', np.arange(10.0))}).chunk({'x': 5})\n"
        "assert type(ds['a'].data).__module__.startswith('dask.array'), type(ds['a'].data)\n"
    )


def test_importing_dask_array_does_not_register():
    """Importing us must not take over xarray, even with xarray already loaded.

    xarray imports dask_array on its own -- Dataset.__dask_exprs__ calls
    import_module("dask_array") whenever it is installed -- so registering on
    import would be an install-time takeover in disguise: it fires the first
    time anybody calls is_dask_collection() on a Dataset.
    """
    _run_isolated(
        "import numpy as np, xarray as xr\n"
        "import dask, dask_array\n"
        "ds = xr.Dataset({'a': ('x', np.arange(10.0))}).chunk({'x': 5})\n"
        "assert not dask_array.xarray.isactive()\n"
        "assert type(ds['a'].data).__module__.startswith('dask.array'), type(ds['a'].data)\n"
        # the import xarray performs behind our back must stay inert too
        "dask.is_dask_collection(ds)\n"
        "assert not dask_array.xarray.isactive()\n"
    )


def test_register_takes_effect_when_imported_before_xarray():
    _run_isolated(
        "import numpy as np\n"
        "import dask_array\n"
        "import xarray as xr\n"
        "dask_array.xarray.register()\n"
        "assert dask_array.xarray.isactive()\n"
        "ds = xr.Dataset({'a': ('x', np.arange(10.0))}).chunk({'x': 5})\n"
        "assert isinstance(ds['a'].data, dask_array.Array), type(ds['a'].data)\n"
    )


@requires_xarray_sliding_window_chunk_manager
def test_xarray_rolling_full_time_chunk_avoids_padding_rechunk():
    da.xarray.register()
    x = xr.DataArray(
        da.ones((100, 6, 8), chunks=(100, 3, 4)),
        dims=("time", "latitude", "longitude"),
    )

    # Force the construct/sliding_window_view path this test guards. With
    # bottleneck installed, xarray's rolling sum takes _bottleneck_reduce
    # instead, whose dask branch (dask_rolling_wrapper, all xarray versions)
    # declares the output dtype via dtypes.maybe_promote — object for bool
    # input, for legacy dask just as for us (legacy merely hides it by
    # computing float64 blocks that contradict its declared meta).
    with xr.set_options(use_bottleneck=False):
        result = (x > 0).rolling(time=72, min_periods=72).sum().max("time")
    optimized = result.data.expr.optimize()

    assert not _contains_expr_type(optimized, TasksRechunk)
    np.testing.assert_allclose(result.compute().values, np.full((6, 8), 72.0))


def test_xarray_rolling_bottleneck_short_first_chunk():
    # xarray's bottleneck rolling path is map_overlap with depth
    # (window - 1, 0) and boundary "none". A first chunk of window - 1 rows
    # gets no left halo, so bottleneck used to reject the block ("Moving
    # window (=30) must between 1 and 29"); the short edge chunk must be
    # merged into its neighbor instead.
    pytest.importorskip("bottleneck")
    da.xarray.register()
    n = 30
    rng = np.random.default_rng(0)
    data = rng.random((n - 1 + 2 * n, 4))
    x = da.from_array(data, chunks=((n - 1, n, n), (4,)))
    result = xr.DataArray(x, dims=["time", "asset"]).rolling(time=n, min_periods=1).sum().compute()
    expected = xr.DataArray(data, dims=["time", "asset"]).rolling(time=n, min_periods=1).sum()
    np.testing.assert_allclose(result.values, expected.values)


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("center", [False, True])
def test_xarray_rolling_long_window_keeps_native_chunks(op, center):
    # DataArray.rolling on dask data takes the bottleneck map_overlap path;
    # a window several times the time chunk must keep the input's native
    # chunking instead of rechunking up to the window.
    pytest.importorskip("bottleneck")
    from dask_array.reductions._sliding_window import MovingWindowReduction

    da.xarray.register()
    rng = np.random.default_rng(0)
    data = rng.normal(size=(13 * 96, 4))
    data[rng.random(data.shape) < 0.15] = np.nan
    x = da.from_array(data, chunks=(96, 4))
    window = 480  # five 96-element chunks

    lazy = getattr(xr.DataArray(x, dims=["time", "asset"]).rolling(time=window, center=center), op)()
    eager = getattr(xr.DataArray(data, dims=["time", "asset"]).rolling(time=window, center=center), op)()

    optimized = lazy.data.expr.optimize()
    assert _contains_expr_type(optimized, MovingWindowReduction)
    assert not _contains_expr_type(optimized, TasksRechunk)
    if not center:  # centered rolling pads, then slices the result back
        assert optimized.chunks == x.chunks
    got = lazy.compute().values
    np.testing.assert_allclose(got, eager.values, rtol=1e-13, atol=1e-13, equal_nan=True)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(eager.values))


def test_xarray_rolling_head_slice_inside_first_window():
    # Slice pushdown used to shrink the rolling's input to window - 1 rows
    # (expanded extent == depth exactly), handing bottleneck a block it must
    # reject: "Moving window (=8640) must between 1 and 8639, inclusive".
    pytest.importorskip("bottleneck")
    da.xarray.register()
    n = 30
    rng = np.random.default_rng(0)
    data = rng.random((5 * n, 4))
    x = da.from_array(data, chunks=((n,) * 5, (4,)))
    r = xr.DataArray(x, dims=["time", "asset"]).rolling(time=n, min_periods=1).sum()
    result = r.isel(time=slice(0, n - 1)).compute()
    expected = xr.DataArray(data, dims=["time", "asset"]).rolling(time=n, min_periods=1).sum()[: n - 1]
    np.testing.assert_allclose(result.values, expected.values)


def test_xarray_rolling_slice_rechunk_map_blocks_receives_full_day_block():
    da.xarray.register()
    samples_per_day = 8
    step_s = 86400 // samples_per_day
    n = 13 * samples_per_day
    time = (
        np.datetime64("2026-06-17") + np.timedelta64(step_s, "s") + np.arange(n) * np.timedelta64(step_s, "s")
    ).astype("datetime64[ns]")
    x = xr.DataArray(
        da.ones((n, 2), chunks=(samples_per_day, 2)),
        dims=("time", "asset"),
        coords={"time": time, "asset": ["A", "B"]},
    )

    adv = x.fillna(0.0).rolling({"time": 5 * samples_per_day}, min_periods=1).sum() * 0.2
    adv = adv.fillna(0.0) + xr.ones_like(x)
    one_day = adv.sel(
        time=slice(
            np.datetime64("2026-06-29"),
            np.datetime64("2026-06-29T23:59:59"),
        )
    )
    arr = one_day.data[:samples_per_day].rechunk((samples_per_day, 2))

    def write_sentinel(block, block_info=None):
        assert block.shape == (samples_per_day, 2)
        return np.array([[1]], dtype="uint8")

    out = arr.map_blocks(
        write_sentinel,
        dtype="uint8",
        chunks=(1, 1),
        meta=np.array((), dtype="uint8"),
    )

    assert arr.chunks == ((samples_per_day,), (2,))
    assert out.chunks == ((1,), (1,))
    for optimize_graph in (True, False):
        result = dask.compute(out, optimize_graph=optimize_graph)[0]
        np.testing.assert_array_equal(result, np.array([[1]], dtype="uint8"))


# =============================================================================
# DaskArrayExprManager
# =============================================================================


def test_manager_init():
    manager = DaskArrayExprManager()
    assert manager.array_cls is da.Array
    assert manager.available is True


def test_manager_is_chunked_array():
    manager = DaskArrayExprManager()
    arr = da.ones((10, 10), chunks=(5, 5))
    assert manager.is_chunked_array(arr)
    assert not manager.is_chunked_array(np.ones((10, 10)))


def test_manager_is_chunked_array_legacy_dask():
    """The manager claims legacy dask.array.Array (and converts it on use).

    register() may be called after xarray has already produced legacy-backed
    objects; since our manager then owns the "dask" registry slot, it must
    recognise those objects or they become permanently unusable through xarray.
    """
    import dask.array as legacy_da

    manager = DaskArrayExprManager()
    arr = legacy_da.ones((10, 10), chunks=(5, 5))
    assert manager.is_chunked_array(arr)


def test_manager_is_chunked_array_does_not_import_legacy_dask():
    """The legacy check is a passive sys.modules probe, never an import.

    Run in a subprocess so nothing that ran earlier in this session has
    already imported dask.array and made the assertions vacuous.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import numpy as np\n"
        "import dask_array as da\n"
        "from dask_array._xarray import DaskArrayExprManager\n"
        "manager = DaskArrayExprManager()\n"
        "assert 'dask.array' not in sys.modules\n"
        "assert manager.is_chunked_array(da.ones((4,), chunks=2))\n"
        "assert not manager.is_chunked_array(np.ones((4,)))\n"
        "assert 'dask.array' not in sys.modules, 'is_chunked_array imported legacy dask.array'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    "op",
    [
        "rechunk",
        "compute",
        "persist",
        "reduction",
        "scan",
        "apply_gufunc",
        "map_blocks",
        "blockwise",
        "unify_chunks",
        "store",
        "shuffle",
    ],
)
def test_manager_methods_convert_legacy_dask(op):
    """Every array-accepting manager method converts legacy inputs.

    Dropping a conversion is not always loud -- e.g. map_blocks on an
    unconverted legacy array used to succeed with silently corrupt metadata --
    so each method is pinned to produce a dask_array-backed result (or plain
    numpy for compute/store) with correct values.
    """
    import dask.array as legacy_da

    manager = DaskArrayExprManager()
    x = np.arange(6.0)
    legacy = legacy_da.from_array(x, chunks=3)

    if op == "rechunk":
        result = manager.rechunk(legacy, (2,))
        assert result.chunks == ((2, 2, 2),)
        expected = x
    elif op == "compute":
        (computed,) = manager.compute(legacy)
        assert type(computed) is np.ndarray
        assert computed.dtype == legacy.dtype
        np.testing.assert_array_equal(computed, x)
        return
    elif op == "persist":
        (result,) = manager.persist(legacy)
        assert result.chunks == legacy.chunks
        expected = x
    elif op == "reduction":
        result = manager.reduction(legacy, np.sum, aggregate_func=np.sum, axis=0, dtype="f8", keepdims=True)
        expected = x.sum(keepdims=True)
    elif op == "scan":
        result = manager.scan(np.cumsum, np.add, 0, legacy, axis=0, dtype="f8")
        expected = np.cumsum(x)
    elif op == "apply_gufunc":
        result = manager.apply_gufunc(np.add, "(),()->()", legacy, 1.0, output_dtypes=["f8"])
        expected = x + 1
    elif op == "map_blocks":
        result = manager.map_blocks(lambda b: b + 1, legacy, dtype="f8")
        assert result.shape == (6,)
        expected = x + 1
    elif op == "blockwise":
        result = manager.blockwise(np.add, "i", legacy, "i", 1.0, None, dtype="f8")
        expected = x + 1
    elif op == "unify_chunks":
        _, (result,) = manager.unify_chunks(legacy, "i")
        expected = x
    elif op == "store":
        target = np.zeros_like(x)
        manager.store([legacy], [target])
        np.testing.assert_array_equal(target, x)
        return
    elif op == "shuffle":
        result = manager.shuffle(legacy, [[3, 1], [2, 0, 4, 5]], 0, "auto")
        expected = x[[3, 1, 2, 0, 4, 5]]

    assert isinstance(result, da.Array), type(result)
    assert result.dtype == legacy.dtype
    np.testing.assert_array_equal(result.compute(), expected)


def test_manager_map_blocks_multi_output_converts_legacy_dask():
    """Legacy inputs to map_blocks_multi_output convert like the siblings."""
    import dask.array as legacy_da

    manager = DaskArrayExprManager()
    x = np.arange(8.0).reshape(4, 2)
    legacy = legacy_da.from_array(x, chunks=(2, 2))

    (double,) = manager.map_blocks_multi_output(
        lambda spec, block: {"double": block * 2},
        [legacy],
        [("x", "y")],
        ("x", "y"),
        {(i, 0): None for i in range(2)},
        [{"key": "double", "indices": ("x", "y"), "chunks": legacy.chunks, "dtype": legacy.dtype}],
        token="convert-legacy",
    )

    assert isinstance(double, da.Array)
    np.testing.assert_array_equal(double.compute(), x * 2)


def test_manager_conversion_never_computes():
    """Conversion is graph-wrapping: a legacy graph whose tasks raise still
    converts; the tasks only run at compute time."""
    import dask.array as legacy_da

    def boom(_):
        raise RuntimeError("task ran during conversion")

    manager = DaskArrayExprManager()
    legacy = legacy_da.map_blocks(
        boom,
        legacy_da.arange(6, chunks=3),
        dtype="f8",
        meta=np.array((), dtype="f8"),
    )

    converted = manager.rechunk(legacy, (6,))
    assert isinstance(converted, da.Array)
    assert converted.dtype == np.dtype("f8")
    with pytest.raises(RuntimeError, match="task ran during conversion"):
        converted.compute()


def test_manager_chunks():
    manager = DaskArrayExprManager()
    arr = da.ones((10, 10), chunks=(5, 5))
    assert manager.chunks(arr) == ((5, 5), (5, 5))


def test_manager_normalize_chunks():
    manager = DaskArrayExprManager()
    result = manager.normalize_chunks((5, 5), shape=(10, 10))
    assert result == ((5, 5), (5, 5))


def test_manager_from_array():
    manager = DaskArrayExprManager()
    x = np.arange(100).reshape(10, 10)
    arr = manager.from_array(x, chunks=(5, 5))
    assert isinstance(arr, da.Array)
    assert arr.chunks == ((5, 5), (5, 5))


def test_manager_from_array_lazy_indexing_adapter_uses_numpy_meta():
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


def test_manager_compute():
    manager = DaskArrayExprManager()
    arr = da.ones((10, 10), chunks=(5, 5))
    (result,) = manager.compute(arr)
    np.testing.assert_array_equal(result, np.ones((10, 10)))


def test_manager_array_api():
    manager = DaskArrayExprManager()
    api = manager.array_api
    assert hasattr(api, "ones")
    assert hasattr(api, "zeros")
    assert hasattr(api, "full_like")


# =============================================================================
# xarray integration with DaskArrayExprManager
# =============================================================================


def test_manager_discoverable():
    """Test that the manager is discoverable via xarray as 'dask'."""
    from xarray.namedarray.parallelcompat import list_chunkmanagers

    da.xarray.register()
    managers = list_chunkmanagers()
    assert "dask" in managers
    # Our manager should be the one registered (replaces built-in)
    assert isinstance(managers["dask"], DaskArrayExprManager)


def test_get_chunked_array_type_selects_manager_once():
    """Test xarray sees dask_array.Array through one chunk manager."""
    from xarray.namedarray.parallelcompat import get_chunked_array_type

    da.xarray.register()
    arr = da.ones((10, 10), chunks=(5, 5))

    assert isinstance(get_chunked_array_type(arr), DaskArrayExprManager)


def test_get_chunked_array_type_survives_legacy_dask_import():
    """Importing legacy dask.array must not reclaim xarray's manager."""
    import subprocess
    import sys

    code = (
        "import dask_array as da\n"
        "da.xarray.register()\n"
        "import dask.array\n"
        "from dask_array._xarray import DaskArrayExprManager\n"
        "from xarray.namedarray.parallelcompat import get_chunked_array_type, list_chunkmanagers\n"
        "arr = da.ones((10, 10), chunks=(5, 5))\n"
        "assert isinstance(list_chunkmanagers()['dask'], DaskArrayExprManager)\n"
        "assert isinstance(get_chunked_array_type(arr), DaskArrayExprManager)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_dask_new_collection_roundtrip():
    """Test Dask can rebuild dask_array.Array from its expression."""
    from dask._collections import new_collection

    arr = da.arange(6, chunks=(3,)) + 1

    rebuilt = new_collection(arr.expr)

    assert isinstance(rebuilt, da.Array)
    np.testing.assert_array_equal(rebuilt.compute(), np.arange(6) + 1)


def test_xarray_register_restores_dask_collection_dispatch_after_legacy_import():
    """Test Dask rebuild dispatch survives importing legacy dask.array."""
    import dask.array  # noqa: F401
    from dask._collections import new_collection

    da.xarray.register()
    arr = da.arange(6, chunks=(3,)) + 1

    rebuilt = new_collection(arr.expr)

    assert isinstance(rebuilt, da.Array)
    np.testing.assert_array_equal(rebuilt.compute(), np.arange(6) + 1)


def test_dataarray_from_dask_array():
    """Test creating a DataArray from a dask_array.Array."""
    arr = da.ones((10, 20), chunks=(5, 10))
    da_xr = xr.DataArray(arr, dims=["x", "y"])

    assert da_xr.shape == (10, 20)
    assert da_xr.chunks == ((5, 5), (10, 10))


def test_dataarray_compute():
    """Test computing a DataArray backed by dask_array.Array."""
    arr = da.arange(100, chunks=25).reshape(10, 10)
    da_xr = xr.DataArray(arr, dims=["x", "y"])

    result = da_xr.compute()
    expected = np.arange(100).reshape(10, 10)
    np.testing.assert_array_equal(result.values, expected)


def test_dataarray_operations():
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
def test_dataarray_rolling_mean():
    arr = da.from_array(np.arange(12.0), chunks=4)
    da_xr = xr.DataArray(arr, dims=["time"])

    result = da_xr.rolling(time=3, center=True).mean()

    assert isinstance(result.data, da.Array)
    expected = np.array([np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan])
    np.testing.assert_allclose(result.compute().values, expected)


@requires_xarray_sliding_window_chunk_manager
def test_dataarray_rolling_construct_multi_axis():
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


def test_dataarray_rechunk():
    """Test rechunking a DataArray."""
    arr = da.ones((10, 20), chunks=(5, 10))
    da_xr = xr.DataArray(arr, dims=["x", "y"])

    rechunked = da_xr.chunk({"x": 10, "y": 5})
    assert rechunked.chunks == ((10,), (5, 5, 5, 5))


def test_dataarray_cftime_auto_rechunk():
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


def test_dataset_from_dask_arrays():
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


def test_apply_ufunc():
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
