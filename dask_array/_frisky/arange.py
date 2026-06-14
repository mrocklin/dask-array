"""Arange layer: 1-D indexed creation (``da.arange``).

``Arange._frisky_layer`` computes the per-block ``blockstart``/``blockstop``/
``size`` scalars in Python — the exact same ``start + elem_count*step``
arithmetic the legacy ``_layer`` runs, which preserves the int/float type of
``start``/``step`` — and passes them to the Rust ``ArangeLayer``. Rust emits one
``arange(blockstart, blockstop, step, size, dtype)`` task per block.
"""

from __future__ import annotations

from functools import partial

from dask_array import _rust
from dask_array._chunk import arange as _arange
from dask_array._frisky.base import Layer


class ArangeLayer(Layer):
    def __init__(self, name, start, step, dtype, like, chunks):
        func = partial(_arange, like=like)
        blockstarts, blockstops, sizes = [], [], []
        elem_count = 0
        for bs in chunks:
            blockstarts.append(start + (elem_count * step))
            blockstops.append(start + ((elem_count + bs) * step))
            sizes.append(int(bs))
            elem_count += bs
        self._rust = _rust.ArangeLayer(name, func, {}, step, dtype, blockstarts, blockstops, sizes)
