"""Banded sliding/moving-window reduction layers (native-chunk rolling
reductions; see ``dask_array/reductions/_sliding_window.py``).

Each output block applies a shared reduce function to its own input block, a
list of middle-block totals, a list of band blocks, and two per-block scalars
— all expressible in the binary records grammar (``Dep``/``List``/``Scalar``
slots), so these layers go out on the Rust fast path.  The banded plan (band
blocks, offsets, truncation counts) is computed once by the expression
(``_block_plan``) and passed here as plain integer rows; Rust only walks the
block grid and emits.
"""

from __future__ import annotations

from dask_array import _rust
from dask_array._frisky.base import Layer


class SlidingWindowReductionLayer(Layer):
    def __init__(self, name, x_name, reduce_func, total_func, axis, numblocks, keepdims, window_axis, plan):
        # plan rows per sliding-axis block: [out_len, band_offset, band_lo,
        # band_hi]; out_len 0 means the window trim consumed the block.
        self._rust = _rust.SlidingWindowReductionLayer(
            name,
            x_name,
            reduce_func,
            total_func,
            {},
            int(axis),
            [int(n) for n in numblocks],
            bool(keepdims),
            int(window_axis),
            [[int(v) for v in row] for row in plan],
        )


class MovingWindowReductionLayer(Layer):
    def __init__(self, name, x_name, reduce_func, total_func, axis, numblocks, plan):
        # plan rows per sliding-axis block: [n_trunc, band_offset, band_lo,
        # band_hi]; band_lo -1 means no band (the block starting the array).
        self._rust = _rust.MovingWindowReductionLayer(
            name,
            x_name,
            reduce_func,
            total_func,
            {},
            int(axis),
            [int(n) for n in numblocks],
            [[int(v) for v in row] for row in plan],
        )
