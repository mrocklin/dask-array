"""from_map data-source layer.

Each block is ``_map_block(func, value, block_shape, kwargs)`` -- a single call
over a per-block Python ``value`` (a delayed-call bundle, or a user datum),
reshaped to the block's chunk shape. The Rust ``FromMapLayer`` does the
O(n_tasks) expansion in place of the generic ``GraphRecordsLayer`` fallback.

``func``/``value``/``kwargs`` are arbitrary Python objects (shared literals), so
-- like ``from_array`` -- there is no binary ``to_records_chunk``; the base
``Layer``'s raises ``NotImplementedError`` and the walk uses ``to_task_records``.
``values`` is the block grid flattened in C order to align with the Rust layer's
row-major block iteration.
"""

from __future__ import annotations

from dask_array import _rust
from dask_array._frisky.base import Layer


class FromMapLayer(Layer):
    def __init__(self, name, map_block, func, kwargs, values, chunks):
        self._rust = _rust.FromMapLayer(name, map_block, func, kwargs or {}, values, chunks)
