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
