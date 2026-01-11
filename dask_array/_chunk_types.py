"""Chunk type registry for duck array support.

This module manages the list of valid chunk types that dask_array can wrap.
By default, this includes numpy.ndarray and numpy.ma.MaskedArray.

See NEP-13 for details on the type casting hierarchy:
https://numpy.org/neps/nep-0013-ufunc-overrides.html#type-casting-hierarchy
"""

from __future__ import annotations

import numpy as np

# Registry of valid chunk types
_HANDLED_CHUNK_TYPES: list[type] = [np.ndarray, np.ma.MaskedArray]


def is_valid_array_chunk(array):
    """Check if given array is of a valid type to operate with."""
    return array is None or isinstance(array, tuple(_HANDLED_CHUNK_TYPES))


def is_valid_chunk_type(type):
    """Check if given type is a valid chunk and downcast array type."""
    try:
        return type in _HANDLED_CHUNK_TYPES or issubclass(type, tuple(_HANDLED_CHUNK_TYPES))
    except TypeError:
        return False


def register_chunk_type(type):
    """Register the given type as a valid chunk and downcast array type.

    Parameters
    ----------
    type : type
        Duck array type to be registered as a type dask_array can safely wrap
        as a chunk and to which dask_array does not defer in arithmetic operations
        and NumPy functions/ufuncs.

    Notes
    -----
    A dask_array.Array can contain any sufficiently "NumPy-like" array in its chunks.
    These are also referred to as "duck arrays" since they match the most important
    parts of NumPy's array API.

    By default, the registry contains:
    * numpy.ndarray
    * numpy.ma.MaskedArray

    Additional types (cupy.ndarray, sparse.SparseArray, etc.) can be registered
    using this function.
    """
    _HANDLED_CHUNK_TYPES.append(type)
