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

import math

from dask_array import _rust
from dask_array._frisky.base import Layer


def _stamp_args(chunks, total_itemsize):
    # Chunk sizes feed only the `-total` expected-nbytes stamps (one keepdims
    # hyperplane each); unknown (nan) sizes disable stamping via itemsize=0.
    known = all(not math.isnan(c) for dim in chunks for c in dim)
    return (
        [[0 if math.isnan(c) else int(c) for c in dim] for dim in chunks],
        int(total_itemsize) if known else 0,
    )


class SlidingWindowReductionLayer(Layer):
    def __init__(
        self, name, x_name, reduce_func, total_func, axis, numblocks, keepdims, window_axis, plan, chunks, total_itemsize
    ):
        # plan rows per sliding-axis block: [out_len, band_offset, band_lo,
        # band_hi]; out_len 0 means the window trim consumed the block.
        chunks, total_itemsize = _stamp_args(chunks, total_itemsize)
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
            chunks,
            total_itemsize,
        )


class MovingWindowReductionLayer(Layer):
    def __init__(self, name, x_name, reduce_func, total_func, axis, numblocks, plan, chunks, total_itemsize):
        # plan rows per sliding-axis block: [n_trunc, band_offset, band_lo,
        # band_hi]; band_lo -1 means no band (the block starting the array).
        chunks, total_itemsize = _stamp_args(chunks, total_itemsize)
        self._rust = _rust.MovingWindowReductionLayer(
            name,
            x_name,
            reduce_func,
            total_func,
            {},
            int(axis),
            [int(n) for n in numblocks],
            [[int(v) for v in row] for row in plan],
            chunks,
            total_itemsize,
        )
