"""from_array data-source layer.

Unlike the computed layers (blockwise, reduction, rechunk, ...), from_array is a
data *source*: each output block is a slice of the backing array. There is no
per-task computation for Rust to accelerate — the work is numpy slicing, which is
inherently Python — so this layer is plain Python, not a Rust layer. (I/O /
source layers are the seam where records originate in Python.)

It mirrors dask's plain-ndarray path: eager per-block slices as data nodes
(`{(name, *idx): array[slc]}`; a single block is copied). The dask path emits the
bare values; the records path wraps each in a `toolz.identity` task, since Frisky
submits tasks, not bare data nodes. Anything outside the plain-ndarray case
(non-ndarray, lock, region, custom getter) falls back to dask — see
`FromArray._frisky_layer`.
"""

from __future__ import annotations

from itertools import product

import toolz

from dask_array._core_utils import slices_from_chunks


class FromArrayLayer:
    def __init__(self, name, array, chunks):
        self._name = name
        self.array = array
        self.chunks = chunks

    def _blocks(self):
        """Yield `(block_index, value)` per output block, matching dask's
        eager-slice (numpy) path: a single block is copied, others are views."""
        slices = slices_from_chunks(self.chunks)
        indices = product(*(range(len(bds)) for bds in self.chunks))
        single = all(len(c) == 1 for c in self.chunks)
        for idx, slc in zip(indices, slices):
            yield idx, (self.array.copy() if single else self.array[slc])

    def to_dask_graph(self):
        return {(self._name, *idx): val for idx, val in self._blocks()}

    def to_task_records(self):
        return [(str((self._name, *idx)), toolz.identity, (val,), {}, []) for idx, val in self._blocks()]
