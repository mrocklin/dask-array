"""Arange layer: 1-D indexed creation (``da.arange``).

``Arange._frisky_layer`` computes the per-block ``blockstart``/``blockstop``/
``size`` scalars in Python — the exact same ``start + elem_count*step``
arithmetic the legacy ``_layer`` runs, which preserves the int/float type of
``start``/``step`` — and passes them to the Rust ``ArangeLayer``. ``step``,
``dtype``, and ``like`` are bound into the shared callable, so Rust emits one
``arange_bound(blockstart, blockstop, size)`` task per block.
"""

from __future__ import annotations

from functools import partial

from dask_array._chunk import arange as _arange
from dask_array._frisky.base import Layer, _rust


def _arange_bound(blockstart, blockstop, size, *, step, dtype, like):
    return _arange(blockstart, blockstop, step, size, dtype, like=like)


class ArangeLayer(Layer):
    def __init__(self, name, start, step, dtype, like, chunks):
        func = partial(_arange_bound, step=step, dtype=dtype, like=like)
        blockstarts, blockstops, sizes = [], [], []
        elem_count = 0
        for bs in chunks:
            blockstarts.append(start + (elem_count * step))
            blockstops.append(start + ((elem_count + bs) * step))
            sizes.append(int(bs))
            elem_count += bs
        self._rust = _rust.ArangeLayer(
            name,
            func,
            {},
            blockstarts,
            blockstops,
            sizes,
        )
