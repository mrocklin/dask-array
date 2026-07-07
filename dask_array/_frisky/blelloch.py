"""Blelloch parallel cumulative-reduction layer (``method="blelloch"``).

``CumReductionBlelloch._frisky_layer`` binds the per-task functions into
partials so the tasks carry only dependency args (no literal ``func``/``axis``/
``dtype`` the binary grammar can't express): the ``preop`` batch, the ``binop``
combine tree, and the ``_prefixscan_first``/``_prefixscan_combine`` outputs. Rust
(``CumReductionBlellochLayer``) replays the upsweep/downsweep exactly, so the
emitted graph matches ``_layer`` value-for-value. The plan needs only the block
count along the axis, so unknown (nan) chunk sizes are fine here.
"""

from __future__ import annotations

from functools import partial

from dask_array import _rust
from dask_array._frisky.base import Layer
from dask_array.reductions._cumulative import _prefixscan_combine, _prefixscan_first


class CumReductionBlellochLayer(Layer):
    def __init__(self, name, func, preop, binop, axis, dtype, x_name, numblocks):
        preop_partial = partial(preop, axis=axis, keepdims=True)
        first_partial = partial(_prefixscan_first, func, axis=axis, dtype=dtype)
        combine_partial = partial(_prefixscan_combine, func, binop, axis=axis, dtype=dtype)

        self._rust = _rust.CumReductionBlellochLayer(
            name,
            preop_partial,
            binop,
            first_partial,
            combine_partial,
            {},
            x_name,
            int(axis),
            [int(n) for n in numblocks],
        )
