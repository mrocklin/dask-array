"""IO functions for array-expr."""

from __future__ import annotations

from dask_array.io._base import IO
from dask_array.io._from_array import FromArray
from dask_array.io._from_delayed import FromDelayed, from_delayed
from dask_array.io._from_graph import FromGraph
from dask_array.io._from_npy_stack import FromNpyStack, from_npy_stack
from dask_array.io._store import get_scheduler_lock, store
from dask_array.io._to_npy_stack import to_npy_stack
from dask_array.io._zarr import from_zarr, to_zarr

__all__ = [
    "IO",
    "FromArray",
    "FromDelayed",
    "FromGraph",
    "FromNpyStack",
    "from_delayed",
    "from_npy_stack",
    "from_zarr",
    "get_scheduler_lock",
    "store",
    "to_npy_stack",
    "to_zarr",
]
