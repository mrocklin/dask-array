"""Cumulative-reduction layer (``cumsum``/``cumprod``, sequential algorithm).

``CumReduction._frisky_layer`` builds the shared per-block chunk function — the
``partial(func, axis=…[, dtype=…])`` the legacy ``_layer`` uses, replicating its
``inspect.signature`` dtype decision exactly — and hands the Rust
``CumReductionLayer`` the remaining funcs (an identity-block wrapper,
``operator.getitem``, the ``binop``), and the block grid. Rust emits the four
task kinds (chunk / identity / tail-getitem / binop) with the sequential carry
along the reduction axis.
"""

from __future__ import annotations

import operator
from functools import partial

import numpy as np

from dask_array import _rust
from dask_array._frisky.base import Layer


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

        self._rust = _rust.CumReductionLayer(
            name,
            chunk_func,
            identity_func,
            operator.getitem,
            binop,
            {},
            x_name,
            int(axis),
            [int(n) for n in numblocks],
            [[int(c) for c in dim] for dim in chunks],
        )


def _full_like_shape(meta, ident, dtype, shape):
    return np.full_like(meta, ident, dtype=dtype, shape=shape)
