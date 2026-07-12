"""Concatenate layer: pure-alias routing from output blocks to input blocks.

``Concatenate._frisky_layer`` computes the per-array block counts along the
concat axis and passes them to ``ConcatenateLayer`` as ``blocks_per_arr``.
Rust builds the ``cum_dims`` table and, for each output block, emits an
``Alias`` to the matching block in the correct source array.

No func or kwargs are needed; every task is a ``Compute::Alias``.
"""

from __future__ import annotations

from dask_array._frisky.base import Layer, _rust


class ConcatenateLayer(Layer):
    def __init__(self, out_name, dep_names, axis, blocks_per_arr, out_numblocks):
        self._rust = _rust.ConcatenateLayer(
            out_name,
            list(dep_names),
            axis,
            list(blocks_per_arr),
            list(out_numblocks),
        )
