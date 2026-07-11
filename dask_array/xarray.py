"""Public controls for xarray chunk manager integration."""

from __future__ import annotations

import sys

__all__ = ("isactive", "register")


def register() -> None:
    """Register dask-array as xarray's active "dask" chunk manager."""
    from dask_array._xarray import _ensure_registered

    _ensure_registered()


def isactive() -> bool:
    """Return whether dask-array is xarray's active "dask" chunk manager."""
    # Our manager can only be registered by importing dask_array._xarray
    # (explicit register() or xarray's entry-point discovery), so if that
    # module was never loaded the answer is no.  Importing it here would
    # register as a side effect -- keep this a passive probe.
    if "dask_array._xarray" not in sys.modules:
        return False

    try:
        from dask_array._xarray import DaskArrayExprManager
        from xarray.namedarray.parallelcompat import list_chunkmanagers
    except ImportError:
        return False

    return isinstance(list_chunkmanagers().get("dask"), DaskArrayExprManager)
