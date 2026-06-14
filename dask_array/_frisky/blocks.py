"""Blocks layer: pure-alias block-index selection (``x.blocks[...]``).

``Blocks._frisky_layer`` computes the per-dimension remap lists in Python
(``np.arange(numblocks[d])[index[d]]`` — output position → input block index,
exactly as ``Blocks._layer`` does), converts them to plain ``int`` lists, and
passes them to ``BlocksLayer``. Rust products over output coords and, for each
output block, emits an ``Alias`` to the remapped input block.

No func or kwargs are needed; every task is a ``Compute::Alias``.
"""

from __future__ import annotations

from dask_array import _rust
from dask_array._frisky.base import Layer


class BlocksLayer(Layer):
    def __init__(self, out_name, dep_name, index_maps):
        # index_maps: per-dimension lists of input block indices (plain ints).
        self._rust = _rust.BlocksLayer(
            out_name,
            dep_name,
            [[int(i) for i in m] for m in index_maps],
        )
