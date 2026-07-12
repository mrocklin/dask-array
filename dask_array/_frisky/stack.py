"""Stack layer: stack multiple arrays along a new axis.

Each output block at coord ``c`` picks input array ``dep_names[c[axis]]`` and
source block coord ``c`` with the new-axis position removed. The task is
``np.expand_dims(source_block, axis)``. The Rust ``StackLayer`` takes the
per-input array names, the output block counts, and the new axis.
"""

from __future__ import annotations

from dask_array._frisky.base import Layer, _rust


class StackLayer(Layer):
    def __init__(self, name, func, dep_names, out_numblocks, axis):
        self._rust = _rust.StackLayer(
            name,
            func,
            {},  # kwargs — stack tasks take no kwargs
            list(dep_names),
            list(out_numblocks),
            axis,
        )
