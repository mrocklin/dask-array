"""Reshape layer: per-block ``M.reshape(in_block, out_shape)``.

Both ``ReshapeLowered._frisky_layer`` and ``ReshapeBlockwise._frisky_layer``
build the same ``ReshapeLayer``: each input block maps 1:1, by C-order position,
to one output block reshaped to that block's shape. Python plans (the input/
output block grids + per-block output shapes, reusing dask's tested reshape
machinery); the Rust ``ReshapeLayer`` does the O(n_tasks) C-order lockstep
expansion. ``M.reshape`` (``dask.utils.M``) calls ``block.reshape(shape)``.
"""

from __future__ import annotations

from dask_array import _rust
from dask_array._frisky.base import Layer


class ReshapeLayer(Layer):
    def __init__(self, name, dep_name, in_numblocks, out_numblocks, out_shapes):
        from dask.utils import M

        self._rust = _rust.ReshapeLayer(
            name,
            M.reshape,
            {},
            dep_name,
            [int(n) for n in in_numblocks],
            [int(n) for n in out_numblocks],
            [[int(s) for s in shp] for shp in out_shapes],
        )
