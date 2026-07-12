"""Array routines for array-expr."""

# Direct imports from submodules
# Re-exports from other modules
from dask_array._collection import asanyarray, asarray  # noqa: F401
from dask_array.core._conversion import array
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
from dask_array.routines._coarsen import Coarsen, aligned_coarsen_chunks, coarsen
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
    iscomplexobj,
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
from dask_array.routines._outer import outer
from dask_array.routines._search import isin, searchsorted
from dask_array.routines._select import (
    choose,
    digitize,
    extract,
    piecewise,
    select,
)
from dask_array.routines._statistics import average, corrcoef, cov, ptp
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

__all__ = [
    "Coarsen",
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
    "iscomplexobj",
    "isin",
    "isnonzero",
    "isnull",
    "ndim",
    "nonzero",
    "notnull",
    "outer",
    "piecewise",
    "ptp",
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
