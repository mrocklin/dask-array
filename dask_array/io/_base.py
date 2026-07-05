from __future__ import annotations

from dask_array._expr import ArrayExpr


class IO(ArrayExpr):
    # Whether rechunk can be pushed into this IO expression by modifying its chunks.
    # False by default since many IO expressions have chunks that affect computation
    # (e.g., Random generates different values with different chunks).
    # Classes that set this True must also dispatch Rechunk parents from their
    # _simplify_up through ArrayExpr._rechunk_pushdown (see FromArray) so the
    # sharing guard applies — a pushed rechunk becomes part of the read.
    _can_rechunk_pushdown = False
