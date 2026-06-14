"""Task-based rechunk layer (``TasksRechunk``).

``TasksRechunk._frisky_layer`` runs dask's tested ``plan_rechunk`` heuristic
(once) to get the rechunk steps, then hands the Rust ``RechunkLayer`` each step's
old/new chunk sizes and merge/split key names. Rust does the per-block work:
the 1-D chunk intersection and the split (``getitem``) / merge
(``concatenate3`` or alias) expansion.
"""

from __future__ import annotations

from dask_array import _rust
from dask_array._frisky.base import Layer


class RechunkLayer(Layer):
    def __init__(self, getitem, concatenate3, steps):
        self._rust = _rust.RechunkLayer(getitem, concatenate3, {}, steps)
