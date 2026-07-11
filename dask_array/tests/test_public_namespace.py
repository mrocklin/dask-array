from __future__ import annotations

import pytest

import dask_array as da


def test_package_version_is_exposed():
    assert isinstance(da.__version__, str)
    assert da.__version__


def test_import_star_helpers_do_not_leak_into_package_namespace():
    for name in ["partial", "functools", "np", "re", "merge", "concat", "dask_array"]:
        assert not hasattr(da, name), name

    assert hasattr(da, "add")
    assert hasattr(da, "apply_gufunc")
    assert hasattr(da, "gufunc")
    assert hasattr(da, "unique")


def test_expr_repr_does_not_swallow_unexpected_errors(monkeypatch):
    expr = da.ones((2,), chunks=1).expr

    def broken_table(self, color=True):
        raise RuntimeError("boom")

    monkeypatch.setattr(type(expr), "_table", broken_table)

    with pytest.raises(RuntimeError, match="boom"):
        repr(expr)


def test_star_import_binds_no_submodules():
    # `from dask_array import *` must not bind submodule names: `io` would
    # shadow the stdlib and `xarray` the real package.
    import types

    namespace = {}
    exec("from dask_array import *", namespace)

    modules = [name for name, value in namespace.items() if isinstance(value, types.ModuleType)]
    assert modules == [], modules
    assert "annotations" not in namespace

    assert callable(namespace["from_array"])
    assert callable(namespace["push"])
    assert callable(namespace["where"])


def test_all_names_resolve():
    missing = [name for name in da.__all__ if not hasattr(da, name)]
    assert missing == [], missing


def test_public_chunk_module_importable():
    import dask_array.chunk

    assert da.chunk is dask_array.chunk
    assert callable(dask_array.chunk.coarsen)
    assert callable(dask_array.chunk.getitem)


def test_normalize_chunks_public_surfaces():
    # Upstream parity: dask.array.normalize_chunks and dask.array.core.normalize_chunks.
    import dask_array.core

    assert da.normalize_chunks((5, 5), shape=(10, 10)) == ((5, 5), (5, 5))
    assert dask_array.core.normalize_chunks is da.normalize_chunks
    assert callable(dask_array.core.getter)
    assert callable(dask_array.core.getter_inline)


def test_tree_reduce_importable_but_not_exported():
    from dask_array.reductions import _tree_reduce

    assert callable(_tree_reduce)
    import dask_array.reductions

    assert "_tree_reduce" not in dask_array.reductions.__all__


def test_routines_single_surface():
    # dask_array._routines is gone; dask_array.routines is the one surface.
    with pytest.raises(ModuleNotFoundError):
        import dask_array._routines  # noqa: F401

    from dask_array.routines import Coarsen, iscomplexobj, ptp

    assert da.ptp is ptp
    assert da.iscomplexobj is iscomplexobj
    assert isinstance(Coarsen, type)

    # The array() duplicate is collapsed onto the core implementation.
    from dask_array.core._conversion import array

    assert da.array is array
    assert da.routines.array is array
