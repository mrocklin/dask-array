"""BroadcastTo layer (``np.broadcast_to`` applied block-wise).

``BroadcastTo._frisky_layer`` passes the output chunk sizes, the number of new
leading axes, and a per-input-dimension flag marking size-1 (broadcast) dims.
The Rust ``BroadcastLayer`` mirrors ``BroadcastTo._layer`` exactly: for each
output block it resolves the input coord (0 on broadcast dims, output coord
otherwise) and passes the per-block shape as an ``IntTuple``.
"""

from __future__ import annotations

import numpy as np

from dask_array._frisky.base import Layer, _rust


class BroadcastLayer(Layer):
    def __init__(self, name, dep_name, out_chunks, ndim_new, broadcast_dim, kwargs=None):
        self._rust = _rust.BroadcastLayer(
            name,
            np.broadcast_to,
            kwargs or {},
            dep_name,
            [list(c) for c in out_chunks],
            ndim_new,
            list(broadcast_dim),
        )
