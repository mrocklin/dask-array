"""Native task expansion for ``Shuffle``.

Python keeps the tested planning surface (chunk grouping and expression
rewrites). Rust expands the per-block take/concat task geometry.
"""

from __future__ import annotations

import numpy as np
from toolz import identity

from dask_array._dispatch import concatenate_lookup, take_lookup
from dask_array._frisky.base import Layer, _rust


def shuffle_take(block, taker, axis):
    return take_lookup(block, np.asarray(taker), axis=axis)


def shuffle_concat(blocks, sorter, axis):
    array = concatenate_lookup.dispatch(type(blocks[0]))(blocks, axis=axis)
    return take_lookup(array, np.argsort(np.asarray(sorter)), axis=axis)


class ShuffleLayer(Layer):
    def __init__(self, name, dep_name, chunks, axis, new_chunks, dtype):
        # itemsize feeds only the expected-nbytes stamps (splits = take
        # segments, outputs = new chunks); the caller guarantees known sizes.
        self._rust = _rust.ShuffleLayer(
            name,
            dep_name,
            shuffle_take,
            shuffle_concat,
            identity,
            {},
            [list(map(int, dim)) for dim in chunks],
            int(axis),
            [list(map(int, chunk)) for chunk in new_chunks],
            int(np.dtype(dtype).itemsize),
        )
