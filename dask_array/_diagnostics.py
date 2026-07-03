"""Chunk-layout diagnostics over an expression graph.

``chunk_report`` is the tool to reach for when a mysteriously large (or
shattered) task shows up in a computation: it walks the *unlowered* expression
graph — pure metadata, nothing computes — and shows where each chunk layout
enters, so the op that changed the layout is named instead of hunted for.
"""

from __future__ import annotations

import math

from dask.utils import format_bytes

from dask_array._expr import ArrayExpr


def _signature(chunks):
    return " x ".join(f"{len(c)}ch({c[0]}{'' if len(set(c)) == 1 else '..' + str(c[-1])})" for c in chunks)


def _max_chunk_bytes(node):
    size = node.dtype.itemsize * math.prod(max(c) for c in node.chunks)
    return None if math.isnan(size) else size


def chunk_report(*arrays, limit=8):
    """Summarize chunk layouts across the expression graph(s) of ``arrays``.

    Returns a printable string with a histogram of chunk layouts (count of
    nodes per layout, its largest chunk, and an example op) plus the
    ``limit`` largest-chunk nodes.  Metadata only — nothing computes.

    >>> import dask_array as da
    >>> x = da.ones((100, 10), chunks=(10, 10))
    >>> print(da.chunk_report(x + 1))  # doctest: +SKIP
    """
    nodes = {}
    for x in arrays:
        for node in getattr(x, "expr", x).walk():
            if not isinstance(node, ArrayExpr) or node._name in nodes:
                continue
            try:
                chunks = node.chunks
            except Exception:
                continue
            if not chunks:  # scalar
                continue
            size = _max_chunk_bytes(node)
            if size is not None:
                nodes[node._name] = (size, _signature(chunks), node)

    if not nodes:
        return "chunk report: no array nodes with known chunks"

    per_layout = {}  # signature -> (count, max bytes, example op)
    for size, sig, node in nodes.values():
        count, worst, op = per_layout.get(sig, (0, -1.0, ""))
        if size > worst:
            worst, op = size, type(node).__name__
        per_layout[sig] = (count + 1, worst, op)

    lines = [f"chunk report: {len(nodes)} array nodes, {len(per_layout)} layouts"]
    for sig, (count, worst, op) in sorted(per_layout.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"  {count:5d}  {sig:38s} <= {format_bytes(worst):>10s}  e.g. {op}")
    lines.append(f"largest chunks ({limit}):")
    for size, sig, node in sorted(nodes.values(), key=lambda t: -t[0])[:limit]:
        lines.append(f"  {format_bytes(size):>10s}  {type(node).__name__:20s} {sig}")
    return "\n".join(lines)
