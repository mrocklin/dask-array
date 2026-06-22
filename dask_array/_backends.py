from __future__ import annotations

import numpy as np

from dask._dispatch import get_collection_type

try:
    import sparse

    sparse_installed = True
except ImportError:
    sparse_installed = False


try:
    import scipy.sparse as sp

    scipy_installed = True
except ImportError:
    scipy_installed = False


_previous_collection_type_handlers = {}


def _register_collection_type(type_, handler) -> None:
    try:
        current = get_collection_type.dispatch(type_)
    except TypeError:
        current = None
    if current is not handler:
        _previous_collection_type_handlers[type_] = current
    get_collection_type.register(type_)(handler)


def _fallback_collection(expr):
    handler = _previous_collection_type_handlers.get(type(expr._meta))
    if handler is None:
        handler = _previous_collection_type_handlers.get(object)
    if handler is None:
        raise TypeError(f"Cannot create collection from {type(expr)}")
    return handler(expr._meta)(expr)


def create_array_collection(expr):
    """Create an Array collection from an expression."""
    from dask_array._collection import Array
    from dask_array._expr import ArrayExpr

    if isinstance(expr, ArrayExpr):
        return Array(expr)

    return _fallback_collection(expr)


def get_collection_type_array(_):
    return create_array_collection


if sparse_installed:

    def get_collection_type_sparse(_):
        return create_array_collection


if scipy_installed:

    def get_collection_type_scipy(_):
        return create_array_collection


if scipy_installed and hasattr(sp, "sparray"):

    def get_collection_type_scipy_array(_):
        return create_array_collection


def get_collection_type_object(_):
    return create_scalar_collection


def create_scalar_collection(expr):
    from dask_array._expr import ArrayExpr

    if isinstance(expr, ArrayExpr):
        from dask_array._collection import Array

        return Array(expr)

    return _fallback_collection(expr)


def register_collection_types() -> None:
    _register_collection_type(np.ndarray, get_collection_type_array)
    if sparse_installed:
        _register_collection_type(sparse.COO, get_collection_type_sparse)
    if scipy_installed:
        _register_collection_type(sp.csr_matrix, get_collection_type_scipy)
    if scipy_installed and hasattr(sp, "sparray"):
        _register_collection_type(sp.csr_array, get_collection_type_scipy_array)
    _register_collection_type(object, get_collection_type_object)


register_collection_types()
