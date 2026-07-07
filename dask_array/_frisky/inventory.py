"""Inventory a collection's lowered layers by graph-generation TIER — the
data-driven burn-down list for pushing graph generation onto the Rust path.

Mirrors the production records walk (:func:`_walk_record_chunks` in
``collect.py``): for every node in the lowered expression tree, which of four
ascending client-side cost tiers do its records take? Cluster-free and fast —
it lowers, walks, and *attempts* record generation, but never computes.

  - ``binary``         Rust ``to_records_chunk`` -> one ``bytes`` blob per
                       layer, O(1) Python objects regardless of block count.
                       The goal.
  - ``native_tuples``  a native ``_frisky_layer`` exists but declines the
                       binary chunk, so ``to_task_records`` builds N Python
                       tuples in Rust. O(tasks).
  - ``adapter``        no native layer -> ``GraphRecordsLayer`` reuses the
                       expr's Python ``_layer()``. O(tasks), the most work.
  - ``fallback``       ``_check_frisky_supported`` rejects the whole graph ->
                       stock dask builds every task in Python. Worst case;
                       one rejected node taints its entire collection.

:func:`classify` weighs each layer by block count, so a single million-block
Blockwise on ``native_tuples`` outranks forty one-block tail nodes on
``adapter`` — the ranking reflects real client-side build cost. It threads one
``seen`` set across all collections, the way a real ``dask.compute(x, y, ...)``
submission dedups a shared subgraph, so a leaf shared by hundreds of
collections is classified once (both faithful and fast on wide fan-outs like
the quantity closure).

The reject signal is per-collection here (which op rejects, and how many
collections) rather than production's per-submission all-or-nothing — more
useful for a burn-down list. Because ``seen`` is shared, the FALLBACK task
counts are approximate when a rejected and a healthy collection share nodes;
``result["rejected"]`` is the exact whole-graph-reject signal.
"""

from __future__ import annotations

import math
from collections import Counter

BINARY = "binary"
NATIVE_TUPLES = "native_tuples"
ADAPTER = "adapter"
FALLBACK = "fallback"
TIERS = (BINARY, NATIVE_TUPLES, ADAPTER, FALLBACK)


def _numtasks(expr):
    try:
        return math.prod(int(n) for n in expr.numblocks) or 1
    except Exception:
        return 1


def classify_node(expr):
    """The ``(tier, reason)`` one lowered node's records take, mirroring
    ``_walk_record_chunks``. Only asks whether the binary chunk is *available*
    — it never materializes the tuple/adapter path (the expensive one we are
    trying to eliminate) — so the walk stays fast. ``reason`` is "" for
    ``binary`` and a short diagnostic otherwise."""
    make_layer = getattr(expr, "_frisky_layer", None)
    if make_layer is None:
        return ADAPTER, "no _frisky_layer"
    try:
        layer = make_layer()
    except (NotImplementedError, ImportError) as exc:
        return ADAPTER, f"_frisky_layer declined: {type(exc).__name__}"
    if layer is None:
        return ADAPTER, "_frisky_layer returned None"
    to_chunk = getattr(layer, "to_records_chunk", None)
    if to_chunk is None:
        return NATIVE_TUPLES, "layer has no to_records_chunk"
    try:
        chunk = to_chunk()
    except NotImplementedError as exc:
        return NATIVE_TUPLES, str(exc)[:60] or "to_records_chunk declined"
    # Production (_walk_record_chunks) treats a None chunk as "declined" and
    # falls back to per-task records, so match that rather than assuming any
    # non-raising return means binary.
    if chunk is None:
        return NATIVE_TUPLES, "to_records_chunk returned None"
    return BINARY, ""


def classify(collections, seen=None):
    """Task-weighted tier inventory across one or more collections.

    ``collections`` is a single dask-array collection or an iterable of them.
    Returns ``{"tiers": Counter tier->tasks, "culprits": Counter (tier, cls,
    reason)->tasks, "nodes": int unique-nodes-walked, "rejected": Counter
    reject_reason->collections}``.
    """
    if hasattr(collections, "_lowered_expr"):
        collections = [collections]
    if seen is None:
        seen = set()
    tiers, culprits, rejected = Counter(), Counter(), Counter()
    nodes = 0
    for collection in collections:
        try:
            collection._check_frisky_supported()
            reject = None
        except NotImplementedError as exc:
            reject = str(exc)[:80]
            rejected[reject] += 1
        stack = [collection._lowered_expr]
        while stack:
            expr = stack.pop()
            if expr._name in seen:
                continue
            seen.add(expr._name)
            nodes += 1
            tasks = _numtasks(expr)
            tier, reason = (FALLBACK, reject) if reject is not None else classify_node(expr)
            tiers[tier] += tasks
            if tier != BINARY:
                culprits[(tier, type(expr).__name__, reason)] += tasks
            try:
                stack.extend(expr.dependencies())
            except Exception:
                pass
    return {"tiers": tiers, "culprits": culprits, "nodes": nodes, "rejected": rejected}


def python_groups(collections):
    """Rows ``(expr_class, reason, ntasks)`` for every NON-binary lowered layer
    across ``collections``, ranked by task volume (descending) — the layers
    that would be built in Python if these were submitted. ``binary`` layers
    are omitted (already on the Rust path). This is the burn-down list."""
    result = classify(collections)
    rows = [
        (cls, f"{tier}: {reason}" if reason else tier, ntasks)
        for (tier, cls, reason), ntasks in result["culprits"].items()
    ]
    rows.sort(key=lambda row: -row[2])
    return rows
