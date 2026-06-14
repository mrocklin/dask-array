"""Same-grid / broadcast elementwise blockwise layer.

``Blockwise._frisky_layer`` normalizes the operands to the
``("literal", value)`` / ``("array", dep_name, ind, numblocks)`` form, index
labels as ints aligned with ``out_ind``, and rejects anything outside the
supported subset. The Rust ``BlockwiseLayer`` holds the func/kwargs/literals and
does the block-id math.
"""

from __future__ import annotations

from dask_array import _rust
from dask_array._frisky.base import Layer


class BlockwiseLayer(Layer):
    def __init__(self, name, func, numblocks, out_ind, args, kwargs=None):
        self._rust = _rust.BlockwiseLayer(name, func, kwargs or {}, list(numblocks), list(out_ind), list(args))
