"""Stack layer: stack multiple arrays along a new axis.

Each output block at coord ``c`` picks input array ``dep_names[c[axis]]`` and
source block coord ``c`` with the new-axis position removed. The task is
``getitem(source_block, indexer)`` where ``indexer`` is the same for every
block (shared literal). The Rust ``StackLayer`` takes the per-input array
names, the output block counts, the new axis, and the shared indexer.
"""

from __future__ import annotations

from dask_array import _rust
from dask_array._frisky.base import Layer


class StackLayer(Layer):
    def __init__(self, name, func, dep_names, out_numblocks, axis, indexer):
        self._rust = _rust.StackLayer(
            name,
            func,
            {},  # kwargs — stack tasks take no kwargs
            list(dep_names),
            list(out_numblocks),
            axis,
            indexer,
        )
