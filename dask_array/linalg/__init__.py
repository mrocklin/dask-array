"""Linear algebra submodule for array-expr.

This module provides native expression-based implementations of linear algebra
operations for the array-expr system.
"""

from dask_array.linalg._cholesky import cholesky
from dask_array.linalg._lu import lu
from dask_array.linalg._norm import norm
from dask_array.linalg._qr import qr, sfqr, tsqr
from dask_array.linalg._solve import inv, lstsq, solve, solve_triangular
from dask_array.linalg._svd import (
    compression_level,
    compression_matrix,
    svd,
    svd_compressed,
)
from dask_array.linalg._tensordot import dot, matmul, tensordot, vdot

__all__ = [
    "cholesky",
    "compression_level",
    "compression_matrix",
    "dot",
    "inv",
    "lstsq",
    "lu",
    "matmul",
    "norm",
    "qr",
    "sfqr",
    "solve",
    "solve_triangular",
    "svd",
    "svd_compressed",
    "tensordot",
    "tsqr",
    "vdot",
]
