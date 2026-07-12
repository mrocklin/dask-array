"""GUfunc leaf layer: the output-splitting step of ``apply_gufunc``.

``GUfuncLeafExpr._frisky_layer`` hands Rust the source (loop-chunked) array grid
plus the leaf's output index and core-dim count. Rust emits one task per source
block — an alias to the block (single output) or ``chunk.getitem(block, i)``
(multiple outputs) — at coord ``(*loop_coord, *core-zeros)``.
"""

from __future__ import annotations

from dask_array._chunk import getitem as chunk_getitem
from dask_array._frisky.base import Layer, _rust


class GUfuncLeafLayer(Layer):
    def __init__(self, name, array_name, numblocks, n_core, nout, i):
        self._rust = _rust.GUfuncLeafLayer(
            name,
            chunk_getitem,
            {},
            array_name,
            [int(n) for n in numblocks],
            int(n_core),
            bool(nout),
            int(i),
        )
