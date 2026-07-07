"""from_map data-source layers.

``FromMapLayer`` is the general case: each block is
``_map_block(func, value, block_shape, kwargs)`` -- a single call over a per-block
Python ``value`` (a delayed-call bundle, or a user datum), reshaped to the block's
chunk shape. The Rust layer does the O(n_tasks) expansion in place of the generic
``GraphRecordsLayer`` fallback. ``func``/``value``/``kwargs`` are arbitrary Python
objects (shared literals), so -- like ``from_array`` -- there is no binary
``to_records_chunk``; the base ``Layer`` raises ``NotImplementedError`` and the
walk uses ``to_task_records``. ``values`` is the block grid flattened in C order
to align with the Rust layer's row-major block iteration.

``FromMapBinaryLayer`` is the pure-Rust special case (see ``FromMap._frisky_layer``):
one shared function across all blocks, with only binary-expressible per-block args
(scalars, strings, int-tuples, lists). ``func`` is the shared block wrapper
``_apply_args`` and ``kwargs`` binds the hoisted user function + its kwargs; the
per-block ``block_args`` are ``(tag, payload)`` descriptors the Rust layer slots
directly, so it emits a binary ``to_records_chunk``.
"""

from __future__ import annotations

from dask_array import _rust
from dask_array._frisky.base import Layer


class FromMapLayer(Layer):
    def __init__(self, name, map_block, func, kwargs, values, chunks):
        self._rust = _rust.FromMapLayer(name, map_block, func, kwargs or {}, values, chunks)


class FromMapBinaryLayer(Layer):
    def __init__(self, name, apply_args, kwargs, block_args, chunks):
        self._rust = _rust.FromMapBinaryLayer(name, apply_args, kwargs, block_args, chunks)
