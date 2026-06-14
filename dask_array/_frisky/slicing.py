"""Basic slicing layer: per output block, ``getitem(input_block, index)``.

``SliceSlicesIntegers._frisky_layer`` runs the intricate per-dimension index
math (``_slice_1d``) in Python — it is O(n_blocks) and runs once per dimension —
and hands the Rust ``SliceLayer`` the resolved per-dimension block slices. Rust
does the O(n_tasks) cartesian-product expansion over them.
"""

from __future__ import annotations

from numbers import Integral

from dask_array import _rust
from dask_array._chunk import getitem
from dask_array._frisky.base import Layer
from dask_array.slicing._utils import _slice_1d


class SliceLayer(Layer):
    def __init__(self, name, dep_name, index, shape, chunks, allow_opt):
        dims = []
        for dim_shape, lengths, ind in zip(shape, chunks, index):
            is_integer = isinstance(ind, Integral)
            reverse = isinstance(ind, slice) and ind.step is not None and ind.step < 0
            # Sorted by input block index, matching SliceSlicesIntegers._layer.
            block_slices = sorted(_slice_1d(dim_shape, lengths, ind).items())
            blocks = [int(b) for b, _ in block_slices]
            elems = [
                (sl.start, sl.stop, sl.step) if isinstance(sl, slice) else (int(sl), None, None)
                for _, sl in block_slices
            ]
            dims.append((is_integer, reverse, blocks, elems))

        self._rust = _rust.SliceLayer(name, getitem, {}, dep_name, bool(allow_opt), dims)
