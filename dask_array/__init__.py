from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _dist_version

import dask as _dask
import numpy as _np

# Rechunk's threshold trades data copies for tasks: a higher value means fewer
# intermediate copies and more tasks. Frisky makes tasks cheap, so we raise the
# planner's default from dask's 4 to 32 (benchmarks: single-copy rechunks win
# 15-35%). update_defaults respects an explicit user setting (but a
# dask.config.refresh() would revert it to 4). The separate p2p-vs-tasks choice
# stays at 4 (see _choose_rechunk_method).
# unify-chunks-policy picks the common layout when elemwise operands disagree
# on chunking: "auto" (default) merges nested chunkings up to the coarsest
# operand unless the merge would move more bytes than the operands backing
# that layout justify (then it refines instead -- splits only, no data moves);
# "coarse" always merges; "refine" opts into stock-dask refinement.
# unify-chunks-limit caps how large a chunk a merge may manufacture under any
# policy (above it, unification refines and warns).
_dask.config.update_defaults(
    {
        "array": {
            "rechunk": {"threshold": 32},
            "unify-chunks-limit": "512 MiB",
            "unify-chunks-policy": "auto",
        }
    }
)

from dask_array import _backends as _backends
from dask_array import _chunk as chunk
from dask_array._core_utils import PerformanceWarning
from dask_array._diagnostics import chunk_report, explain, trace_rewrites
from dask.base import compute
from dask_array._chunk_types import register_chunk_type

from dask_array import fft, random
from dask_array._collection import (
    Array,
    array,
    asanyarray,
    asarray,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    block,
    blockwise,
    broadcast_to,
    concatenate,
    dstack,
    elemwise,
    expand_dims,
    flip,
    fliplr,
    flipud,
    from_array,
    hstack,
    moveaxis,
    ravel,
    rechunk,
    reshape,
    reshape_blockwise,
    roll,
    rollaxis,
    rot90,
    squeeze,
    stack,
    swapaxes,
    transpose,
    vstack,
)
from dask_array._einsum import einsum
from dask_array._gufunc import *
from dask_array._histogram import histogram, histogram2d, histogramdd
from dask_array._map_blocks import map_blocks
from dask_array._overlap import map_overlap, overlap, sliding_window_view, trim_overlap
from dask_array._routines import (
    aligned_coarsen_chunks,
    allclose,
    append,
    apply_along_axis,
    apply_over_axes,
    argtopk,
    argwhere,
    around,
    average,
    bincount,
    broadcast_arrays,
    choose,
    coarsen,
    compress,
    corrcoef,
    count_nonzero,
    cov,
    delete,
    digitize,
    ediff1d,
    extract,
    flatnonzero,
    gradient,
    insert,
    isclose,
    iscomplexobj,
    isin,
    isnull,
    ndim,
    nonzero,
    notnull,
    outer,
    piecewise,
    ptp,
    ravel_multi_index,
    result_type,
    round,
    searchsorted,
    select,
    shape,
    take,
    topk,
    tril,
    tril_indices,
    tril_indices_from,
    triu,
    triu_indices,
    triu_indices_from,
    unify_chunks,
    union1d,
    unique,
    unravel_index,
)
from dask_array._shuffle import shuffle
from dask_array._ufunc import *
from dask_array.creation import (
    arange,
    diag,
    diagonal,
    empty,
    empty_like,
    eye,
    fromfunction,
    full,
    full_like,
    indices,
    linspace,
    meshgrid,
    ones,
    ones_like,
    pad,
    repeat,
    tile,
    tri,
    zeros,
    zeros_like,
)
from dask_array.io import (
    from_delayed,
    from_map,
    from_npy_stack,
    from_tiledb,
    from_zarr,
    store,
    to_hdf5,
    to_npy_stack,
    to_tiledb,
    to_zarr,
)
from dask_array.linalg import dot, matmul, tensordot, vdot
from dask_array.reductions import (
    _tree_reduce,
    all,
    any,
    arg_reduction,
    argmax,
    argmin,
    cumprod,
    cumreduction,
    cumsum,
    max,
    mean,
    median,
    min,
    moment,
    nanargmax,
    nanargmin,
    nancumprod,
    nancumsum,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanpercentile,
    nanprod,
    nanquantile,
    nanstd,
    nansum,
    nanvar,
    percentile,
    prod,
    quantile,
    reduction,
    std,
    sum,
    trace,
    var,
)
from dask_array.routines._diff import diff
from dask_array.routines._where import where
from dask_array._expr_flow import expr_flow
from dask_array._visualize import expr_table
from dask_array import xarray

try:
    __version__ = _dist_version("dask-array")
except _PackageNotFoundError:
    __version__ = "0.0.0+unknown"


def optimize(dsk, keys=None, **kwargs):
    """Optimize a dask-array collection.

    Low-level graphs are returned unchanged because this package optimizes
    through Array expressions before graph materialization.
    """
    if isinstance(dsk, Array):
        result = dsk.optimize()
        if keys is not None:
            return result.__dask_graph__()
        return result
    return dsk


newaxis = None
nan = _np.nan
inf = _np.inf
e = _np.e
pi = _np.pi
euler_gamma = _np.euler_gamma

bool = _np.bool
int8 = _np.int8
int16 = _np.int16
int32 = _np.int32
int64 = _np.int64
uint8 = _np.uint8
uint16 = _np.uint16
uint32 = _np.uint32
uint64 = _np.uint64
float32 = _np.float32
float64 = _np.float64
complex64 = _np.complex64
complex128 = _np.complex128

# Ensure our xarray ChunkManager replaces the built-in DaskManager
# regardless of entry point enumeration order. See _xarray.py for details.
try:
    xarray.register()
except ImportError:
    pass
