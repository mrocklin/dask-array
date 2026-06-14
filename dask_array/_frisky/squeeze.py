"""Squeeze layer: remove size-1 axes from each input block.

``Squeeze._frisky_layer`` normalizes the squeezed-axis set, computes
``chunk_axis`` (the tuple of input-dimension indices to squeeze, shared across
all output blocks), and passes the output block counts + input ndim to the Rust
``SqueezeLayer``. The Rust expansion inserts 0 at each squeezed axis to map
output coords back to input coords.
"""

from __future__ import annotations

import numpy as np

from dask_array import _rust
from dask_array._frisky.base import Layer


class SqueezeLayer(Layer):
    def __init__(self, name, dep_name, numblocks, input_ndim, axis_set):
        # chunk_axis: sorted tuple of input-dim indices being squeezed;
        # passed as shared kwargs so every task calls np.squeeze(..., axis=...).
        chunk_axis = tuple(sorted(axis_set))
        self._rust = _rust.SqueezeLayer(
            name,
            np.squeeze,
            {"axis": chunk_axis},
            dep_name,
            list(numblocks),
            input_ndim,
            sorted(axis_set),
        )
