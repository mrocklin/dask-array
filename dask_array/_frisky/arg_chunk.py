"""ArgChunk layer: the per-block chunk step of an arg reduction.

``ArgChunk._frisky_layer`` computes the per-block offsets in Python — exactly the
legacy non-ravel ``pluck(axis[0], offsets)`` math — and passes them, along with
the shared ``axis`` and the input block counts, to the Rust ``ArgChunkLayer``.
Rust emits one ``chunk_func(x_block, axis, off)`` task per block, with the output
coord equal to the input coord (identity map) and a single dependency at the same
coord. ``axis`` is shared across blocks; ``off`` varies per block. Non-ravel
(``axis=k``) blocks carry a scalar offset; ravel (``axis=None``) blocks carry the
nested ``(per-dim offsets, full shape)`` the ravel ``arg_chunk`` unpacks.
"""

from __future__ import annotations

from dask_array import _rust
from dask_array._frisky.base import Layer


class ArgChunkLayer(Layer):
    def __init__(self, name, chunk_func, axis, dep_name, numblocks, ravel, offs, offset_tuples, shape):
        self._rust = _rust.ArgChunkLayer(
            name,
            chunk_func,
            {},
            axis,
            dep_name,
            [int(n) for n in numblocks],
            bool(ravel),
            [int(o) for o in offs],
            [[int(o) for o in tup] for tup in offset_tuples],
            [int(s) for s in shape],
        )
