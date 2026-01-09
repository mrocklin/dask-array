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


def create_array_collection(expr):
    """Create an Array collection from an expression."""
    from dask_array._collection import Array
    from dask_array._expr import ArrayExpr

    if isinstance(expr, ArrayExpr):
        return Array(expr)

    # For non-ArrayExpr (e.g., from dask-dataframe), wrap in adapter
    # This is a fallback - most cases should be ArrayExpr
    raise TypeError(f"Expected ArrayExpr, got {type(expr)}")


@get_collection_type.register(np.ndarray)
def get_collection_type_array(_):
    return create_array_collection


if sparse_installed:

    @get_collection_type.register(sparse.COO)
    def get_collection_type_sparse(_):
        return create_array_collection


if scipy_installed:

    @get_collection_type.register(sp.csr_matrix)
    def get_collection_type_scipy(_):
        return create_array_collection


if scipy_installed and hasattr(sp, "sparray"):

    @get_collection_type.register(sp.csr_array)
    def get_collection_type_scipy_array(_):
        return create_array_collection


@get_collection_type.register(object)
def get_collection_type_object(_):
    return create_scalar_collection


def create_scalar_collection(expr):
    from dask_array._expr import ArrayExpr

    if isinstance(expr, ArrayExpr):
        from dask_array._collection import Array

        return Array(expr)

    # For other expressions, try to return something sensible
    raise TypeError(f"Cannot create collection from {type(expr)}")
