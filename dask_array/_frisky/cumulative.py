"""Cumulative-reduction layer (``cumsum``/``cumprod``, sequential algorithm).

``CumReduction._frisky_layer`` builds the shared per-block chunk function — the
``partial(func, axis=…[, dtype=…])`` the legacy ``_layer`` uses, replicating its
``inspect.signature`` dtype decision exactly — and hands the Rust
``CumReductionLayer`` the remaining funcs (an identity-block wrapper, the ``tail``
function, the ``binop``), and the block grid. Rust emits the four task kinds
(chunk / identity / tail / binop) with the sequential carry along the reduction
axis.
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np

from dask_array import _rust
from dask_array._chunk import getitem as chunk_getitem
from dask_array._frisky.base import Layer
from dask_array.reductions._cumulative import _cum_tail


class CumReductionLayer(Layer):
    def __init__(self, name, func, binop, ident, axis, dtype, meta, x_name, numblocks, chunks):
        # Mirror CumReduction._layer's use_dtype decision exactly.
        use_dtype = False
        try:
            import inspect

            use_dtype = "dtype" in inspect.signature(func).parameters
        except ValueError:
            try:
                if isinstance(func.__self__, np.ufunc) and func.__name__ == "accumulate":
                    use_dtype = True
            except AttributeError:
                pass
        chunk_func = partial(func, axis=axis, dtype=dtype) if use_dtype else partial(func, axis=axis)

        identity_func = partial(_full_like_shape, meta, ident, dtype)

        # The per-block carry: the last hyperplane along the axis, or the identity
        # when a block is empty there (so an empty block contributes nothing rather
        # than an unbroadcastable size-0 slice). The non-empty branch uses
        # chunk.getitem (copy-if-small), not operator.getitem: the tail is a
        # one-slice view of the whole previous cumulated block and would otherwise
        # pin it until the next binop runs. See TasksRechunk._frisky_layer.
        ndim = len(numblocks)
        tail_slice = (slice(None),) * axis + (slice(-1, None),) + (slice(None),) * (ndim - axis - 1)
        tail_func = partial(_cum_tail, chunk_getitem, tail_slice, ident, axis)

        # Expected-nbytes stamping needs true chunk sizes; with any unknown (nan)
        # size the placeholder 1s below are only valid for the identity shapes,
        # so pass itemsize=0 to leave those layers' stamps unknown.
        known = all(not math.isnan(c) for dim in chunks for c in dim)
        itemsize = int(np.dtype(dtype).itemsize) if known else 0

        self._rust = _rust.CumReductionLayer(
            name,
            chunk_func,
            identity_func,
            tail_func,
            binop,
            {},
            x_name,
            int(axis),
            [int(n) for n in numblocks],
            # Chunk sizes feed the `extra` identity block's shape, which is 1
            # along the reduction axis and otherwise just broadcasts against real
            # block data — so an unknown (nan) size can become 1 rather than crash
            # `int(nan)`. The plan itself never depended on sizes, only on the fixed
            # block count. (They also feed expected-nbytes stamps, disabled above
            # when any size is unknown.)
            [[1 if math.isnan(c) else int(c) for c in dim] for dim in chunks],
            itemsize,
        )


def _full_like_shape(meta, ident, dtype, shape):
    return np.full_like(meta, ident, dtype=dtype, shape=shape)
