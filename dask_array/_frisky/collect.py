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

import json
import math

import numpy as np

from dask_array._frisky.graph_records import GraphRecordsLayer

_I64_MAX = (1 << 63) - 1


def _jsonify(v, depth=0):
    """Render a parameter value as a bounded, JSON-safe form for display. Anything
    we don't model becomes a truncated ``repr`` — never the raw object, so a stray
    numpy array or closure can't bloat (or break) the blob."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, np.dtype):
        return str(v)
    if isinstance(v, np.generic):  # numpy scalar
        return _jsonify(v.item(), depth)
    if isinstance(v, np.ndarray):  # summarize, never embed the data
        return f"<ndarray shape={tuple(v.shape)} dtype={v.dtype}>"
    if isinstance(v, slice):
        return {"start": _jsonify(v.start), "stop": _jsonify(v.stop), "step": _jsonify(v.step)}
    if isinstance(v, (list, tuple)):
        if depth >= 3:
            return repr(v)[:200]
        out = [_jsonify(x, depth + 1) for x in v[:32]]
        if len(v) > 32:
            out.append(f"...(+{len(v) - 32} more)")
        return out
    if isinstance(v, dict):
        if depth >= 3:
            return repr(v)[:200]
        return {str(k): _jsonify(val, depth + 1) for k, val in list(v.items())[:32]}
    if callable(v):
        return getattr(v, "__name__", None) or repr(v)[:200]
    return repr(v)[:200]


def _expr_params(e):
    """The expression's scalar/config *parameters* — op, axis, chunks, dtype,
    indexer, kwargs, … — NOT its array inputs (child expressions, whether named
    operands or trailing varargs). Each value is rendered JSON-safe and bounded by
    :func:`_jsonify`. Defensive: any hiccup yields ``{}`` rather than sinking the
    whole metadata blob."""
    try:
        from dask._expr import Expr

        names = getattr(e, "_parameters", None) or []
        operands = getattr(e, "operands", None) or []
        params = {}
        for name, val in zip(names, operands):
            if isinstance(val, Expr):
                continue  # a single array input
            if isinstance(val, (list, tuple)) and any(isinstance(x, Expr) for x in val):
                continue  # a collection of array inputs (e.g. stack/concatenate)
            params[name] = _jsonify(val)
        return params
    except Exception:
        return {}


# A dimension with more than this many chunks is summarized rather than listed —
# real arrays can have 100k+ chunks per dim, which would bloat the metadata blob
# (built here, stored per-group on the scheduler, shipped on every poll, and
# rendered). Small dims are listed in full so the common case stays legible.
_MAX_CHUNKS_PER_DIM = 16

# Hard cap on the whole metadata blob. Matches Frisky's scheduler-side cap
# (MAX_GROUP_METADATA_BYTES, transitions.rs). `chunks` is summarized and
# `_jsonify` bounds each param, so realistic layers land far under this; the cap
# only bites a pathological nested `params`, and we drop just `params` (keeping
# op/shape/chunks/dtype) rather than letting the scheduler drop the whole blob.
_MAX_METADATA_BYTES = 16 * 1024


def _summarize_chunks(chunks):
    """Bounded per-dimension chunk description: the full size list when a dim has
    few chunks, else a compact ``{nchunks, min, max}`` so the blob stays small no
    matter how finely the array is chunked."""
    out = []
    for dim in chunks:  # each `dim` is already a tuple of ints — don't copy it
        if len(dim) <= _MAX_CHUNKS_PER_DIM:
            out.append([int(b) for b in dim])
        else:
            out.append({"nchunks": len(dim), "min": int(min(dim)), "max": int(max(dim))})
    return out


def _layer_metadata(e):
    """A small JSON blob describing the expression/layer — operation kind, shape,
    chunking, dtype, and the expression's scalar *parameters* — for Frisky to
    display against the layer's *group*. Opaque to Frisky (it stores the string
    verbatim and never parses it); this keeps the scheduler array-agnostic. One
    blob per layer (not per task).

    Bounded by construction: `chunks` is summarized for finely-chunked dims (see
    `_summarize_chunks`) and `params` values are capped by `_jsonify`, so the blob
    can't blow up on a 100k-chunk array. As a hard backstop, an oversized blob
    drops `params` (a pathological nested param is the only way to get there).

    Returns a JSON string, or ``None`` when it can't be built JSON-safely (a node
    without these attrs, unknown/``nan`` dims, or non-coercible values) — in which
    case the group simply has no metadata."""
    try:
        meta = {
            "op": type(e).__name__,
            "shape": [int(s) for s in e.shape],
            "numblocks": [int(n) for n in e.numblocks],
            "chunks": _summarize_chunks(e.chunks),
            "dtype": str(e.dtype),
            "params": _expr_params(e),
        }
        # allow_nan=False → invalid-JSON NaN/Inf raise instead of being emitted.
        blob = json.dumps(meta, allow_nan=False)
        if len(blob) > _MAX_METADATA_BYTES:
            meta.pop("params", None)
            blob = json.dumps(meta, allow_nan=False)
        return blob
    except (AttributeError, TypeError, ValueError):
        return None


def _expected_nbytes_metadata(e):
    """Return ``(name, chunks, itemsize)`` for Rust-side nbytes stamping.

    Binary records are still array-agnostic. The collector is the one place that
    has both the emitted layer chunk and the array expression metadata, so it can
    stamp final output tasks for any binary-capable layer with known
    ``chunks``/``dtype``. Helper tasks (rechunk splits, scan carries, overlap halo
    getitems, …) are stamped earlier, at layer expansion time, where their exact
    extents are in hand — this pass fills only stamps that are still zero. The
    byte-level task walk runs in Rust; Python only normalizes
    O(numblocks-per-axis) metadata.
    """
    try:
        name = e._name
        itemsize = int(np.dtype(e.dtype).itemsize)
        if itemsize <= 0:
            return None
        chunks = []
        for dim in e.chunks:
            sizes = []
            for size in dim:
                try:
                    if math.isnan(size):
                        sizes.append(0)
                        continue
                except TypeError:
                    pass
                size = int(size)
                if size < 0:
                    return None
                if size > _I64_MAX:
                    size = _I64_MAX
                sizes.append(size)
            chunks.append(sizes)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None

    return name, chunks, itemsize


def _stamp_expected_nbytes(chunk, e):
    meta = _expected_nbytes_metadata(e)
    if meta is None:
        return chunk

    # Via base, not dask_array._rust directly, so the build-freshness check in
    # base.py is guaranteed to have run before we call into the extension.
    from dask_array._frisky.base import _rust

    return _rust.stamp_expected_nbytes(chunk, *meta)


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


def _walk_record_chunks(roots, seen, chunks, records, chunk_groups):
    """Like ``_walk_records`` but each node contributes EITHER a binary records
    LAYER chunk (``to_records_chunk``, the fast Rust-to-Rust path) appended to
    ``chunks``, OR plain ``(key, func, args, kwargs, deps)`` records appended to
    ``records`` (the from_array source, generic ``GraphRecordsLayer`` fallback, or
    any layer that declines the binary chunk — by design for literal-carrying
    layers; see ``Layer.to_records_chunk``). Frisky decodes both and
    unions them under one ``dask.order`` pass, so a graph need not be all-or-nothing
    to get the speedup on the layers that support it.

    ``chunk_groups`` is kept parallel to ``chunks``: for each emitted chunk, the
    producing expr's ``(_name, metadata_json, upstream_group_names)`` — its stable
    layer identity (which a key prefix can't always recover), an opaque JSON
    description, and its child layers' ``_name``s (the layer-DAG edges, so Frisky
    can draw layer→layer data flow without scanning tasks). Frisky ties that
    layer's tasks to this group. Residual ``records`` carry no group entry (Frisky
    key-derives them; an edge pointing at such a layer still resolves, since its
    key-derived group shares the same ``_name``)."""
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
        deps = e.dependencies()
        if chunk is not None:
            chunk = _stamp_expected_nbytes(chunk, e)
            chunks.append(chunk)
            # Upstream group names = this layer's child exprs' _names (deduped).
            upstream = sorted({c._name for c in deps})
            chunk_groups.append((e._name, _layer_metadata(e), upstream))
            # A layer whose chunk references a shared data node it can't put in the
            # chunk (from_array's source array) emits it here as a plain record, so
            # the array ships once and the chunk's Dep slots resolve to it.
            side = getattr(layer, "chunk_side_records", None)
            if side is not None:
                records.extend(side())
        else:
            records.extend(layer.to_task_records())

        stack.extend(deps)


def collect_record_chunks(collection, seen=None):
    """Hybrid binary/Python records for ``collection``:
    ``(chunks, records, chunk_groups)``.

    ``chunks`` is a list of binary records LAYER chunks (bytes); ``records`` is a
    list of plain task records for the layers that didn't go binary;
    ``chunk_groups`` is parallel to ``chunks`` — each ``(layer _name,
    metadata_json, upstream_group_names)`` so Frisky groups the layer's tasks by
    their true identity, can show the layer's shape/chunks/dtype, and can draw the
    layer-DAG edges. Same ``seen`` semantics as
    :func:`collect_task_records` — pass a shared set across a multi-collection
    submission so a shared subgraph is expanded once. Completeness is checked by
    the caller over the combined union (Frisky)."""
    if seen is None:
        seen = set()
    chunks = []
    records = []
    chunk_groups = []
    _walk_record_chunks([collection._lowered_expr], seen, chunks, records, chunk_groups)
    return chunks, records, chunk_groups


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
