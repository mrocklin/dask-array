"""No-dependency creation layer (ones/zeros/empty/full).

Each output block is ``func(block_shape)``; the Rust ``CreationLayer`` reads each
block's shape off the chunk sizes. ``func`` is the wrapped creation function (a
partial carrying dtype/meta/kwargs), shared across all blocks. ``chunks`` must
already be concrete integer sizes per dimension.
"""

from __future__ import annotations

from dask_array import _rust
from dask_array._frisky.base import Layer


class CreationLayer(Layer):
    def __init__(self, name, func, chunks, kwargs=None):
        self._rust = _rust.CreationLayer(name, func, kwargs or {}, chunks)
