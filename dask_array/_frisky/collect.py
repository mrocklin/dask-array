"""Collect plain task records for a whole collection.

Mirrors dask's ``Expr.__dask_graph__`` traversal — a stack over
``dependencies()`` deduped by ``_name`` — but calls each expression's
``_frisky_layer().to_task_records()`` instead of ``_layer()``. Returns a flat
list of ``(key, func, args, kwargs, deps)`` records (a Rust-built mirror of the
dask Task, one per task), which the Frisky client serializes as it would any
task graph.

Raises ``NotImplementedError`` if any expression has no Frisky layer, so a
caller can fall back to the dask path for graphs not yet fully covered.
"""

from __future__ import annotations


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
        if make_layer is None:
            raise NotImplementedError(f"{type(e).__name__} has no _frisky_layer")
        records.extend(make_layer().to_task_records())

        stack.extend(e.dependencies())
    return records
