"""Public controls for xarray chunk manager integration."""

from __future__ import annotations

import sys

__all__ = ("isactive", "register")


def register() -> None:
    """Register dask-array as xarray's active "dask" chunk manager.

    Opt-in: nothing else activates it.  Installing or importing dask-array
    leaves xarray on its built-in dask manager, so that adding this package
    to an environment cannot change how other libraries behave.
    """
    from dask_array._xarray import _ensure_registered

    _ensure_registered()


def isactive() -> bool:
    """Return whether dask-array is xarray's active "dask" chunk manager."""
    # register() is the only path in, and it imports dask_array._xarray, so if
    # that module was never loaded the answer is no.  Importing it here would
    # be pointless work -- keep this a passive probe.
    if "dask_array._xarray" not in sys.modules:
        return False

    try:
        from dask_array._xarray import DaskArrayExprManager
        from xarray.namedarray.parallelcompat import list_chunkmanagers
    except ImportError:
        return False

    return isinstance(list_chunkmanagers().get("dask"), DaskArrayExprManager)
