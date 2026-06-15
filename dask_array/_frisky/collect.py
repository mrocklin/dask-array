"""Collect plain task records for a whole collection.

Mirrors dask's ``Expr.__dask_graph__`` traversal — a stack over
``dependencies()`` deduped by ``_name`` — but builds ``(key, func, args, kwargs,
deps)`` records instead of a dict. Each expression contributes its records one of
two ways:

  - A native Frisky layer (``_frisky_layer().to_task_records()``): the fast,
    Rust-generated path for the layers that have been ported (blockwise, reduction,
    rechunk, from_array, …).
  - Otherwise — no ``_frisky_layer``, or it raises ``NotImplementedError`` for this
    variant — the generic ``GraphRecordsLayer`` reuses the expression's own legacy
    ``_layer()`` graph (``Task``/``Alias``/``DataNode``) and translates it. This
    covers the specialized tail without a Rust port per op (perf is deferred there);
    the rest of the graph still takes the fast path. If even that can't represent a
    node (an unhandled ``_task_spec`` type), it raises ``NotImplementedError`` and
    the caller falls back to stock dask for the whole graph.
"""

from __future__ import annotations

from dask_array._frisky.graph_records import GraphRecordsLayer


def collect_task_records(collection):
    expr = collection.expr.lower_completely()
    stack = [expr]
    seen = set()
    records = []
    while stack:
        e = stack.pop()
        if e._name in seen:
            continue
        seen.add(e._name)

        make_layer = getattr(e, "_frisky_layer", None)
        layer = None
        if make_layer is not None:
            try:
                layer = make_layer()
            except NotImplementedError:
                layer = None
        if layer is None:
            layer = GraphRecordsLayer(e)
        records.extend(layer.to_task_records())

        stack.extend(e.dependencies())
    return records
