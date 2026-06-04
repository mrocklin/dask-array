"""Minimal module for new_collection to avoid circular imports."""

from __future__ import annotations


def new_collection(expr):
    """Create new Array collection from an expression."""
    from dask_array._collection import Array

    return Array(expr)
