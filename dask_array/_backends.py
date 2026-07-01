from __future__ import annotations

import functools

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


def ensure_collection_types_registered() -> None:
    """Re-assert our ``get_collection_type`` registration if it has been stolen.

    ``dask._collections.new_collection`` dispatches on ``type(expr._meta)`` -- a
    ``numpy.ndarray`` -- via the single shared ``get_collection_type`` Dispatch.
    Both dask.array's array-expr backend and dask_array register a handler under
    the ``np.ndarray`` (and ``object``) key, and ``Dispatch.register`` is
    last-writer-wins. Whenever dask.array's ``_array_expr._backends`` imports
    *after* us (e.g. during an xarray/frisky ``__dask_exprs__`` composite
    descent), it steals those slots; its ``create_array_collection`` then assumes
    a dask.array expr and reads ``.divisions``, which a dask_array expr lacks.

    dask_array can't win a one-shot registration race, so re-assert our
    registration lazily right before ``new_collection`` dispatches. Our
    ``create_array_collection`` still guards with ``isinstance(expr, ArrayExpr)``
    and falls back to the prior handler for non-dask_array exprs, so reclaiming
    the slot is safe.
    """
    # np.ndarray stands in for every key we own: register_collection_types
    # registers them (np.ndarray, object, sparse/scipy) together, so they are
    # stolen and reclaimed as a set.
    if get_collection_type.dispatch(np.ndarray) is not get_collection_type_array:
        register_collection_types()


def _patch_new_collection() -> None:
    """Wrap ``dask._collections.new_collection`` to re-assert our registration.

    This is the exact dispatch site frisky/xarray use during composite descent:
    a pre-built dask_array expr is handed to dask-core's ``new_collection`` with
    no intervening dask_array call, so re-asserting anywhere else is too early.
    Wrapping ``new_collection`` re-claims the shared ``np.ndarray``/``object``
    dispatch slot immediately before the dispatch, whichever backend imported
    last. Idempotent -- a re-import of this module leaves the single wrapper in
    place.
    """
    import dask._collections as _collections

    original = _collections.new_collection
    if getattr(original, "_dask_array_reassert", False):
        return

    @functools.wraps(original)
    def new_collection(expr):
        ensure_collection_types_registered()
        return original(expr)

    new_collection._dask_array_reassert = True
    _collections.new_collection = new_collection


register_collection_types()
_patch_new_collection()
