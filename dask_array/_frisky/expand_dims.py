"""Dimension expansion layer (expand_dims).

Each output block is ``func(input_block, axes)`` (``func`` is ``np.expand_dims``
and ``axes`` the sorted expansion positions). The Rust ``ExpandDimsLayer`` maps
each input block coord to the output coord by inserting ``0`` at each of the
sorted expansion ``axes``, and carries ``axes`` as a plain int tuple so the
layer serializes to a binary records chunk.
"""

from __future__ import annotations

from dask_array._frisky.base import Layer, _rust


class ExpandDimsLayer(Layer):
    def __init__(self, name, input_name, func, input_numblocks, axes):
        self._rust = _rust.ExpandDimsLayer(
            name,
            input_name,
            func,
            {},  # kwargs
            list(input_numblocks),
            list(axes),
        )
