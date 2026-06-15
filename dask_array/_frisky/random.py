"""random data-source layer (``da.random.random``, ``.normal``, ``.poisson``, …).

Like ``from_array``, ``Random`` is a *source*: each output block draws from a
per-block RNG bit-generator. There's no cross-block graph structure to expand in
Rust — the per-block state (a spawned ``SeedSequence`` / RNG state) is the work —
so this is a plain Python layer.

We reuse the expr's own ``_task`` (which already handles every distribution
subclass and the ``extra_chunks`` book-keeping) to build each per-block dask
``Task``, then read its ``func``/``args``/``kwargs`` straight off for the records
path. ``Random._frisky_layer`` falls back to legacy dask when a distribution
parameter is itself an array (a dependency), since that would need task-refs
threaded through the nested arg/kwarg containers.
"""

from __future__ import annotations

from itertools import product


class RandomLayer:
    def __init__(self, expr):
        self.expr = expr

    def _block_ids(self):
        return product(*(range(len(bd)) for bd in self.expr.chunks))

    def to_dask_graph(self):
        e = self.expr
        return {(e._name, *bid): e._task((e._name, *bid), bid) for bid in self._block_ids()}

    def to_task_records(self):
        e = self.expr
        records = []
        for bid in self._block_ids():
            key = (e._name, *bid)
            task = e._task(key, bid)
            # `_task` builds a positional-only Task; read the flat record off it.
            records.append((str(key), task.func, tuple(task.args), task.kwargs or {}, []))
        return records
