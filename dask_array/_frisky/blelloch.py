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

import math
from functools import partial

import numpy as np

from dask_array import _rust
from dask_array._frisky.base import Layer
from dask_array.reductions._cumulative import _prefixscan_combine, _prefixscan_first


def _itemsize(value):
    return int(np.asarray(value).dtype.itemsize)


def _consistent_itemsize(kind, *values):
    sizes = {_itemsize(value) for value in values}
    if len(sizes) != 1:
        raise NotImplementedError(f"Blelloch {kind} task dtype is not stable")
    return sizes.pop()


def _infer_itemsize_stamps(func, preop, binop, axis, input_dtype, output_dtype, ndim):
    """Infer the dtypes produced by Blelloch's different task families."""
    try:
        sample = np.ones((1,) * ndim, dtype=input_dtype)
        batch = preop(sample, axis=axis, keepdims=True)
        scan = binop(batch, batch)
        first = _prefixscan_first(func, sample, axis=axis, dtype=output_dtype)
        combine = _prefixscan_combine(
            func, binop, batch, sample, axis=axis, dtype=output_dtype
        )
        return (
            _itemsize(batch),
            _consistent_itemsize("scan", scan, binop(scan, batch), binop(batch, scan)),
            _itemsize(first),
            _consistent_itemsize(
                "combine",
                combine,
                _prefixscan_combine(func, binop, scan, sample, axis=axis, dtype=output_dtype),
            ),
        )
    except Exception as exc:
        raise NotImplementedError(
            "could not infer Blelloch task dtypes for expected_nbytes"
        ) from exc


class CumReductionBlellochLayer(Layer):
    def __init__(
        self,
        name,
        func,
        preop,
        binop,
        axis,
        input_dtype,
        dtype,
        x_name,
        numblocks,
        chunks,
    ):
        preop_partial = partial(preop, axis=axis, keepdims=True)
        first_partial = partial(_prefixscan_first, func, axis=axis, dtype=dtype)
        combine_partial = partial(_prefixscan_combine, func, binop, axis=axis, dtype=dtype)

        # Chunk sizes feed only expected-nbytes stamps (outputs = whole blocks,
        # batches/scan nodes = keepdims hyperplanes); unknown (nan) sizes disable
        # stamping via itemsize=0 rather than mis-stamp.
        known = all(not math.isnan(c) for dim in chunks for c in dim)
        if known:
            (
                batch_itemsize,
                scan_itemsize,
                first_itemsize,
                combine_itemsize,
            ) = _infer_itemsize_stamps(func, preop, binop, axis, input_dtype, dtype, len(numblocks))
        else:
            batch_itemsize = scan_itemsize = first_itemsize = combine_itemsize = 0

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
            [[0 if math.isnan(c) else int(c) for c in dim] for dim in chunks],
            batch_itemsize,
            scan_itemsize,
            first_itemsize,
            combine_itemsize,
        )
