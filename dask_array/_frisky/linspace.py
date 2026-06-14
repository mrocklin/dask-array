"""Linspace layer: 1-D indexed creation (``da.linspace``).

``Linspace._frisky_layer`` computes the per-block ``blockstart``/``blockstop``/
``size`` scalars in Python — the exact same running-``blockstart`` arithmetic the
legacy ``_layer`` runs (advance by ``step*bs`` each block; ``blockstop`` over
``bs - 1`` elements when ``endpoint`` is set), which preserves the int/float type
of ``start``/``step`` — and passes them to the Rust ``LinspaceLayer``. Rust emits
one ``linspace(blockstart, blockstop, size)`` task per block; ``endpoint`` and
``dtype`` are baked into the shared ``functools.partial`` chunk function.
"""

from __future__ import annotations

from functools import partial

from dask_array import _rust
from dask_array._chunk import linspace as _linspace
from dask_array._frisky.base import Layer


class LinspaceLayer(Layer):
    def __init__(self, name, start, step, endpoint, dtype, chunks):
        func = partial(_linspace, endpoint=endpoint, dtype=dtype)
        blockstarts, blockstops, sizes = [], [], []
        blockstart = start
        for bs in chunks:
            bs_space = bs - 1 if endpoint else bs
            blockstops.append(blockstart + (bs_space * step))
            blockstarts.append(blockstart)
            sizes.append(int(bs))
            blockstart = blockstart + (step * bs)
        self._rust = _rust.LinspaceLayer(name, func, {}, blockstarts, blockstops, sizes)
