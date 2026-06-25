"""Collect plain task records for a whole collection.

Mirrors dask's ``Expr.__dask_graph__`` traversal — a stack over
``dependencies()`` deduped by ``_name`` — but builds ``(key, func, args, kwargs,
deps)`` records instead of a dict. Each expression contributes its records one of
two ways:

  - A native Frisky layer (``_frisky_layer().to_task_records()``): the fast,
    Rust-generated path for the layers that have been ported (blockwise, reduction,
    rechunk, from_array, …).
  - Otherwise — no ``_frisky_layer``, or it raises ``NotImplementedError`` /
    ``ImportError`` for this variant — the generic ``GraphRecordsLayer`` reuses the
    expression's own legacy ``_layer()`` graph (``Task``/``Alias``/``DataNode``)
    and translates it. This covers the specialized tail without a Rust port per op
    (perf is deferred there); the rest of the graph still takes the fast path. If
    even that can't represent a node (an unhandled ``_task_spec`` type), it raises
    ``NotImplementedError`` and the caller falls back to stock dask for the whole
    graph.

Finally the assembled graph is checked for completeness: every dependency must be
produced by some record. A dangling reference means the translation wasn't faithful
— e.g. an *optimized/fused* expr (``x.sum().compute()`` fuses mul→add→chunk into a
SubgraphCallable) whose ``_layer()`` still references the fused-away block keys. The
records path can't express that, so we raise ``NotImplementedError`` and fall back to
stock dask rather than submit an incomplete graph (which would silently hang).
"""

from __future__ import annotations

from dask_array._frisky.graph_records import GraphRecordsLayer


def _walk_records(roots, seen, records):
    """Depth-first walk over already-lowered ``roots``, appending each node's
    records to ``records`` and deduping every node by ``_name`` against the
    shared ``seen`` set. Passing several roots with one ``seen`` is what lets a
    subgraph shared across collections be expanded exactly once."""
    stack = list(roots)
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
            except (NotImplementedError, ImportError):
                layer = None
        if layer is None:
            layer = GraphRecordsLayer(e)
        records.extend(layer.to_task_records())

        stack.extend(e.dependencies())


def _walk_record_chunks(roots, seen, chunks, records):
    """Like ``_walk_records`` but each node contributes EITHER a binary records
    LAYER chunk (``to_records_chunk``, the fast Rust-to-Rust path) appended to
    ``chunks``, OR plain ``(key, func, args, kwargs, deps)`` records appended to
    ``records`` (the from_array source, generic ``GraphRecordsLayer`` fallback, or
    any layer whose Rust backend doesn't yet emit chunks). Frisky decodes both and
    unions them under one ``dask.order`` pass, so a graph need not be all-or-nothing
    to get the speedup on the layers that support it."""
    stack = list(roots)
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
            except (NotImplementedError, ImportError):
                layer = None
        if layer is None:
            layer = GraphRecordsLayer(e)

        chunk = None
        to_chunk = getattr(layer, "to_records_chunk", None)
        if to_chunk is not None:
            try:
                chunk = to_chunk()
            except NotImplementedError:
                chunk = None
        if chunk is not None:
            chunks.append(chunk)
        else:
            records.extend(layer.to_task_records())

        stack.extend(e.dependencies())


def collect_record_chunks(collection, seen=None):
    """Hybrid binary/Python records for ``collection``: ``(chunks, records)``.

    ``chunks`` is a list of binary records LAYER chunks (bytes); ``records`` is a
    list of plain task records for the layers that didn't go binary. Same
    ``seen`` semantics as :func:`collect_task_records` — pass a shared set across
    a multi-collection submission so a shared subgraph is expanded once.
    Completeness is checked by the caller over the combined union (Frisky)."""
    if seen is None:
        seen = set()
    chunks = []
    records = []
    _walk_record_chunks([collection._lowered_expr], seen, chunks, records)
    return chunks, records


def _check_complete(records):
    """Every dependency must be produced by some record, else the translation
    wasn't faithful (e.g. a fused expr referencing fused-away keys) — raise so
    the caller falls back to stock dask rather than submit an incomplete graph."""
    produced = {r[0] for r in records}
    dangling = {dep for r in records for dep in r[4]} - produced
    if dangling:
        raise NotImplementedError(f"records graph has {len(dangling)} dangling dep(s), e.g. {next(iter(dangling))}")


def collect_task_records(collection, seen=None):
    """Flat ``(key, func, args, kwargs, deps)`` records for ``collection``.

    ``seen`` is the set of already-walked expr ``_name``s. Pass a shared set
    across several collections in one submission (``dask.compute(x, y)``) and a
    subgraph common to them is expanded exactly once — the costly per-node
    ``_frisky_layer``/``GraphRecordsLayer`` materialization is skipped for nodes
    an earlier collection already produced. In that shared mode this returns only
    *this* collection's contribution and does **not** check completeness: its
    records reference shared keys produced by another collection, so completeness
    is the caller's job over the combined union. With ``seen=None`` (single
    collection / legacy callers) a fresh set is used and completeness is checked
    here as before."""
    shared = seen is not None
    if seen is None:
        seen = set()
    records = []
    # Reuse the collection's lowered expression so records and __dask_keys__ agree,
    # including after a pickle round trip where the sender's lowering policy was
    # preserved to keep advertised Frisky outputs stable across processes.
    _walk_records([collection._lowered_expr], seen, records)
    if not shared:
        _check_complete(records)
    return records
