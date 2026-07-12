"""Overlap neighbor assembly layer.

``OverlapInternal._frisky_layer`` keeps the tested Python planning surface
(depth normalization and boundary construction) and hands Rust the regular block
grid plus per-axis overlap depths. Rust expands the per-block neighbor getitems
and shaped concatenate tasks without materializing Dask's old-style
``ArrayOverlapLayer`` graph.
"""

from __future__ import annotations

import math

import numpy as np

from dask.array.core import concatenate_shaped

from dask_array._chunk import getitem
from dask_array._frisky.base import Layer, _rust


class OverlapLayer(Layer):
    def __init__(self, name, dep_name, numblocks, axes, chunks, dtype):
        axis_depths = []
        for axis, depth in axes.items():
            if isinstance(depth, tuple):
                left, right = depth
            else:
                left = right = depth
            axis_depths.append((int(axis), int(left), int(right)))

        # Source chunk sizes feed only the halo-getitem expected-nbytes stamps;
        # unknown (nan) sizes disable stamping via itemsize=0.
        known = all(not math.isnan(c) for dim in chunks for c in dim)
        itemsize = int(np.dtype(dtype).itemsize) if known else 0

        self._rust = _rust.OverlapLayer(
            name,
            dep_name,
            # chunk.getitem (copy-if-small), not operator.getitem: halo
            # slices are small views of whole neighbor blocks and would
            # otherwise pin them in worker memory until the concatenate
            # runs.  See TasksRechunk._frisky_layer for the full story.
            getitem,
            concatenate_shaped,
            {},
            [int(n) for n in numblocks],
            axis_depths,
            [[0 if math.isnan(c) else int(c) for c in dim] for dim in chunks],
            itemsize,
        )
