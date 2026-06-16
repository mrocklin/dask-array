"""FusedBlockwise records layer.

A ``FusedBlockwise`` is the result of fusing a chain of Blockwise/Elemwise exprs
into one expr — ``Array.optimize()`` does this, and ``Array.compute()`` optimizes
before computing, so idiomatic user code hits it. Its ``_task`` emits exactly ONE
dask ``Task`` per output block::

    _execute_subgraph(subgraph, outkey, inkeys, *source_block_refs)

where ``subgraph`` is a dict of the fused-away inner Tasks (carried as a *literal*
arg and run on the worker by ``_execute_subgraph``), ``inkeys`` are the literal
source-block key-tuples used to seed the subgraph's inputs, and the task's only
real ``dependencies`` are the external source blocks (the inputs to the chain) —
NOT the fused-away ops.

The generic ``GraphRecordsLayer`` adapter mistranslates this: it reads the inner
subgraph's internal key-tuples as graph dependencies, so the completeness check in
``collect_task_records`` sees them dangling and the whole graph falls back to stock
dask. This native layer sidesteps that the same way ``random``/``from_array`` do:
it reuses the expr's own ``_task`` and reads the flat record straight off each dask
``Task`` — so ``deps`` are exactly the real source blocks and the embedded subgraph
rides through as an opaque literal arg.

Frisky ships this faithfully with no changes: its ``lower_dep_refs`` lowers the
top-level source ``TaskRef``s to worker placeholders but leaves the subgraph dict's
inner ``Task`` values (opaque to it) and the literal ``inkeys`` untouched, so
``_execute_subgraph`` resolves the internal refs on the worker exactly as stock
dask does for a fused task.
"""

from __future__ import annotations

from itertools import product


class FusedBlockwiseLayer:
    def __init__(self, expr):
        self.expr = expr

    def _tasks(self):
        e = self.expr
        for bid in product(*(range(n) for n in e.numblocks)):
            key = (e._name, *bid)
            yield key, e._task(key, bid)

    def to_dask_graph(self):
        return {key: task for key, task in self._tasks()}

    def to_task_records(self):
        # ``task.dependencies`` is a frozenset, so ``deps`` order is not stable
        # across processes — that's fine: Frisky matches deps by string, and the
        # ``inkeys``<->source-block alignment ``_execute_subgraph`` relies on lives
        # inside ``task.args`` (both derive from the same ``external_deps`` tuple),
        # independent of this list's order.
        return [
            (str(key), task.func, tuple(task.args), task.kwargs or {}, [str(d) for d in task.dependencies])
            for key, task in self._tasks()
        ]
