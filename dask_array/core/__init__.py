"""Core array types and wrapping functions.

This module re-exports the core Array class and conversion functions.
"""

from dask_array.core._blockwise_funcs import blockwise, elemwise
from dask_array.core._conversion import (
    array,
    asanyarray,
    asarray,
    from_array,
)
from dask_array.io._from_graph import from_graph

# Upstream-compatible aliases: dask.array.core exposes these helpers and
# downstream libraries import them from there.
from dask_array._core_utils import (
    getter,
    getter_inline,
    getter_nofancy,
    normalize_chunks,
)


def __getattr__(name):
    """Lazy import of Array to avoid circular imports."""
    if name == "Array":
        from dask_array._collection import Array

        return Array
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Array",
    "from_array",
    "from_graph",
    "asarray",
    "asanyarray",
    "array",
    "blockwise",
    "elemwise",
    "getter",
    "getter_inline",
    "getter_nofancy",
    "normalize_chunks",
]
