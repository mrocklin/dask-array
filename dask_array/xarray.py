"""Public controls for xarray chunk manager integration."""

from __future__ import annotations

__all__ = ("isactive", "register")


def register() -> None:
    """Register dask-array as xarray's active "dask" chunk manager."""
    from dask_array._xarray import _ensure_registered

    _ensure_registered()


def isactive() -> bool:
    """Return whether dask-array is xarray's active "dask" chunk manager."""
    try:
        from dask_array._xarray import DaskArrayExprManager
        from xarray.namedarray.parallelcompat import list_chunkmanagers
    except ImportError:
        return False

    return isinstance(list_chunkmanagers().get("dask"), DaskArrayExprManager)
