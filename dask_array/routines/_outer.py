from __future__ import annotations

import numpy as np


def outer(a, b):
    """
    Compute the outer product of two vectors.

    This docstring was copied from numpy.outer.

    Some inconsistencies with the Dask version may exist.

    Given two vectors, ``a = [a0, a1, ..., aM]`` and
    ``b = [b0, b1, ..., bN]``,
    the outer product is::

      [[a0*b0  a0*b1 ... a0*bN ]
       [a1*b0    .
       [ ...          .
       [aM*b0            aM*bN ]]

    Parameters
    ----------
    a : (M,) array_like
        First input vector.  Input is flattened if not already 1-dimensional.
    b : (N,) array_like
        Second input vector.  Input is flattened if not already 1-dimensional.

    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``
    """
    from dask_array._collection import asarray, blockwise

    a = asarray(a).flatten()
    b = asarray(b).flatten()

    dtype = np.outer(a.dtype.type(), b.dtype.type()).dtype

    return blockwise(np.outer, "ij", a, "i", b, "j", dtype=dtype)
