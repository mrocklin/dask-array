"""Diag layers (``da.diag``, k=0).

``Diag1D`` builds a diagonal matrix from a 1-D array: diagonal blocks are
``np.diag(x_block)``, off-diagonal blocks ``np.zeros_like(meta, shape=(m, n))``
— the per-block ``shape`` keyword is carried via the neutral form's per-task
kwargs (``Compute::CallKw``). ``Diag2DSimple`` extracts the diagonal of a
square-block 2-D array: block ``i`` is ``np.diag(x[(i, i)])``.
"""

from __future__ import annotations

import numpy as np

from dask_array._frisky.base import Layer, _rust


class Diag1DLayer(Layer):
    def __init__(self, name, meta, dep_name, chunks_1d):
        self._rust = _rust.Diag1DLayer(name, np.diag, np.zeros_like, {}, meta, dep_name, [int(c) for c in chunks_1d])


class Diag2DSimpleLayer(Layer):
    def __init__(self, name, dep_name, nblocks):
        self._rust = _rust.Diag2DSimpleLayer(name, np.diag, {}, dep_name, int(nblocks))
