"""ArgChunk layer: the per-block chunk step of an arg reduction.

``ArgChunk._frisky_layer`` computes the per-block offsets in Python — exactly the
legacy non-ravel ``pluck(axis[0], offsets)`` math — and passes them, along with
the shared ``axis`` and the input block counts, to the Rust ``ArgChunkLayer``.
Rust emits one ``chunk_func(x_block, axis, off)`` task per block, with the output
coord equal to the input coord (identity map) and a single dependency at the same
coord. ``axis`` is shared across blocks; ``off`` varies per block. The ravel case
is not modeled here (its offset is a nested tuple) — the routing raises
``NotImplementedError`` and falls back to legacy dask.
"""

from __future__ import annotations

from dask_array import _rust
from dask_array._frisky.base import Layer


class ArgChunkLayer(Layer):
    def __init__(self, name, chunk_func, axis, dep_name, numblocks, offs):
        self._rust = _rust.ArgChunkLayer(
            name,
            chunk_func,
            {},
            axis,
            dep_name,
            [int(n) for n in numblocks],
            [int(o) for o in offs],
        )
