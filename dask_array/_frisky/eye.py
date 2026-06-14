"""Eye layer: 2-D indexed creation (``da.eye``).

Each output block is either ``np.eye(vchunk, hchunk, local_k, dtype)`` (when the
block straddles the ``k``-diagonal) or ``np.zeros((vchunk, hchunk), dtype)``
otherwise. The per-block diagonal test and ``local_k`` offset are pure integer
arithmetic over the ``(i, j)`` grid, so they run in Rust; Python passes only the
two funcs (``np.eye``/``np.zeros``), the shared ``dtype``, the per-dimension chunk
size lists, the diagonal scale ``chunk_size`` and the diagonal index ``k``.
"""

from __future__ import annotations

import numpy as np

from dask_array import _rust
from dask_array._frisky.base import Layer


class EyeLayer(Layer):
    def __init__(self, name, dtype, vchunks, hchunks, chunk_size, k):
        self._rust = _rust.EyeLayer(
            name,
            np.eye,
            np.zeros,
            {},
            dtype,
            [int(c) for c in vchunks],
            [int(c) for c in hchunks],
            int(chunk_size),
            int(k),
        )
