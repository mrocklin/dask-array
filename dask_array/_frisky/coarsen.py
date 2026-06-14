"""Coarsen layer: apply a fixed reduction over fixed-size neighborhoods.

``Coarsen._frisky_layer`` reconstructs the shared
``functools.partial(chunk.coarsen, reduction, axes=..., trim_excess=...,
**kwargs)`` exactly as the legacy ``_layer`` does and passes the input block
counts to the Rust ``CoarsenLayer``. The Rust expansion emits one task per block
with the output coord equal to the input coord (identity map) and a single
dependency at the same coord — the simplest blockwise shape. Everything is baked
into the partial, so the shared kwargs dict is empty.
"""

from __future__ import annotations

from functools import partial

from dask_array import _chunk as chunk
from dask_array import _rust
from dask_array._frisky.base import Layer


class CoarsenLayer(Layer):
    def __init__(self, name, dep_name, numblocks, reduction, axes, trim_excess, kwargs):
        func = partial(
            chunk.coarsen,
            reduction,
            axes=axes,
            trim_excess=trim_excess,
            **kwargs,
        )
        self._rust = _rust.CoarsenLayer(
            name,
            func,
            {},
            dep_name,
            list(numblocks),
        )
