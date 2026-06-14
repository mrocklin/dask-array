"""Dimension expansion layer (expand_dims).

Each output block is ``getitem(input_block, indexer)`` where ``indexer`` is the
same for every block. The Rust ``ExpandDimsLayer`` maps each input block coord
to the output coord by inserting ``0`` at each of the sorted expansion ``axes``.
"""

from __future__ import annotations

from dask_array import _rust
from dask_array._frisky.base import Layer


class ExpandDimsLayer(Layer):
    def __init__(self, name, input_name, func, indexer, input_numblocks, axes):
        self._rust = _rust.ExpandDimsLayer(
            name,
            input_name,
            func,
            {},  # kwargs
            indexer,
            list(input_numblocks),
            list(axes),
        )
