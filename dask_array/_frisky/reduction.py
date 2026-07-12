"""Tree-reduction aggregate layer (``PartialReduce``).

``PartialReduce._frisky_layer`` passes the shared aggregate ``func``, the input
block grid, and the per-dimension ``split_every`` steps (0 for kept axes). The
Rust ``PartialReduceLayer`` reproduces dask's ``lol_tuples`` nesting — reduced
axes become nested lists, kept axes fixed coordinates — as one nested-list
argument per output block.
"""

from __future__ import annotations

from dask_array._frisky.base import Layer, _rust


class PartialReduceLayer(Layer):
    def __init__(self, name, func, dep_name, numblocks, steps, keepdims):
        self._rust = _rust.PartialReduceLayer(name, func, {}, dep_name, list(numblocks), list(steps), keepdims)
