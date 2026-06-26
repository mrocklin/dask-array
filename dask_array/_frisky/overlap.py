"""Overlap neighbor assembly layer.

``OverlapInternal._frisky_layer`` keeps the tested Python planning surface
(depth normalization and boundary construction) and hands Rust the regular block
grid plus per-axis overlap depths. Rust expands the per-block neighbor getitems
and shaped concatenate tasks without materializing Dask's old-style
``ArrayOverlapLayer`` graph.
"""

from __future__ import annotations

import operator

from dask.array.core import concatenate_shaped

from dask_array import _rust
from dask_array._frisky.base import Layer


class OverlapLayer(Layer):
    def __init__(self, name, dep_name, numblocks, axes):
        axis_depths = []
        for axis, depth in axes.items():
            if isinstance(depth, tuple):
                left, right = depth
            else:
                left = right = depth
            axis_depths.append((int(axis), int(left), int(right)))

        self._rust = _rust.OverlapLayer(
            name,
            dep_name,
            operator.getitem,
            concatenate_shaped,
            {},
            [int(n) for n in numblocks],
            axis_depths,
        )
