"""Array routines for array-expr."""

import numpy as np

from dask.utils import derived_from

# Direct imports from submodules
# Re-exports from other modules
from dask_array._blockwise import outer  # noqa: F401
from dask_array._collection import asanyarray, asarray
from dask_array._ufunc import (  # noqa: F401
    allclose,
    around,
    isclose,
    isnull,
    notnull,
    round,
)
from dask_array.routines._apply import apply_along_axis, apply_over_axes
from dask_array.routines._bincount import bincount
from dask_array.routines._broadcast import broadcast_arrays, unify_chunks
from dask_array.routines._coarsen import aligned_coarsen_chunks, coarsen
from dask_array.routines._diff import diff
from dask_array.routines._gradient import gradient
from dask_array.routines._indexing import ravel_multi_index, unravel_index
from dask_array.routines._insert_delete import (
    append,
    delete,
    ediff1d,
    insert,
)
from dask_array.routines._misc import (
    compress,
    ndim,
    result_type,
    shape,
    take,
)
from dask_array.routines._nonzero import (
    argwhere,
    count_nonzero,
    flatnonzero,
    isnonzero,
    nonzero,
)
from dask_array.routines._search import isin, searchsorted
from dask_array.routines._select import (
    choose,
    digitize,
    extract,
    piecewise,
    select,
)
from dask_array.routines._statistics import average, corrcoef, cov
from dask_array.routines._topk import argtopk, topk
from dask_array.routines._triangular import (
    tril,
    tril_indices,
    tril_indices_from,
    triu,
    triu_indices,
    triu_indices_from,
)
from dask_array.routines._unique import union1d, unique
from dask_array.routines._where import where


@derived_from(np)
def array(x, dtype=None, ndmin=None, *, like=None):
    x = asarray(x, like=like)
    while ndmin is not None and x.ndim < ndmin:
        x = x[None, :]
    if dtype is not None and x.dtype != dtype:
        x = x.astype(dtype)
    return x


__all__ = [
    "aligned_coarsen_chunks",
    "allclose",
    "append",
    "apply_along_axis",
    "array",
    "apply_over_axes",
    "argwhere",
    "argtopk",
    "around",
    "average",
    "bincount",
    "broadcast_arrays",
    "choose",
    "coarsen",
    "compress",
    "corrcoef",
    "count_nonzero",
    "cov",
    "delete",
    "diff",
    "digitize",
    "ediff1d",
    "extract",
    "flatnonzero",
    "gradient",
    "insert",
    "isclose",
    "isin",
    "isnonzero",
    "isnull",
    "ndim",
    "nonzero",
    "notnull",
    "outer",
    "piecewise",
    "ravel_multi_index",
    "result_type",
    "round",
    "searchsorted",
    "select",
    "shape",
    "take",
    "topk",
    "tril",
    "tril_indices",
    "tril_indices_from",
    "triu",
    "triu_indices",
    "triu_indices_from",
    "unify_chunks",
    "union1d",
    "unique",
    "unravel_index",
    "where",
]
