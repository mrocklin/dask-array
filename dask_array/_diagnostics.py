"""Optimizer and chunk-layout diagnostics over an expression graph.

``chunk_report`` is the tool to reach for when a mysteriously large (or
shattered) task shows up in a computation: it walks the *unlowered* expression
graph — pure metadata, nothing computes — and shows where each chunk layout
enters, so the op that changed the layout is named instead of hunted for.

``trace_rewrites`` records every rewrite rule that fires while an expression
simplifies or lowers. Rules that decline (return None) leave no record, so
"why didn't my rewrite fire" is answered by its absence here plus the
surviving node in the tree.

``explain`` runs the whole optimization pipeline phase by phase (raw →
simplified → lowered → fused) and reports how each phase changed the graph:
node and task counts, bytes read at the leaves, transfer estimates, and the
rules responsible.
"""

from __future__ import annotations

import functools
import math
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

from dask._expr import Expr
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


# ---------------------------------------------------------------------------
# Rewrite tracing
# ---------------------------------------------------------------------------

# Hooks the rewrite framework calls on every node, and the phase they belong
# to.  ``_accept_slice``/``_accept_rechunk``/``_accept_shuffle`` are invoked
# *from* these hooks, so wrapping the hooks captures pushdowns too — the rule
# string names the class whose hook fired.
_REWRITE_HOOKS = {
    "_simplify_down": "simplify",
    "_simplify_up": "simplify",
    "_lower": "lower",
}


@dataclass
class RewriteRecord:
    """One rewrite that fired: ``rule`` turned ``before`` into ``after``.

    For ``_simplify_down``/``_lower`` the rule's class is the node that
    rewrote itself; for ``_simplify_up`` it is the child that rewrote its
    parent (``before`` is the parent).
    """

    phase: str  # "simplify" or "lower"
    rule: str  # e.g. "Elemwise._simplify_up"
    before_type: str
    after_type: str
    before_name: str
    after_name: str


def _rule_lines(records):
    counts = Counter((r.rule, r.before_type, r.after_type) for r in records)
    return [f"  {n:4d} x {rule}: {before} -> {after}" for (rule, before, after), n in counts.most_common()]


class Trace:
    """Rewrites recorded by a :func:`trace_rewrites` context, in firing order."""

    def __init__(self):
        self.records: list[RewriteRecord] = []

    def __len__(self):
        return len(self.records)

    def __repr__(self):
        if not self.records:
            return "trace: no rewrites"
        return "\n".join([f"trace: {len(self.records)} rewrites"] + _rule_lines(self.records))


def _expr_classes():
    seen = set()
    stack = [ArrayExpr]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    return seen


def _traced_hook(orig, hook, phase, out_trace):
    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if out is None:
            return out
        before = args[0] if hook == "_simplify_up" else self
        after_name = getattr(out, "_name", None)
        if after_name != before._name:
            out_trace.records.append(
                RewriteRecord(
                    phase=phase,
                    rule=f"{type(self).__name__}.{hook}",
                    before_type=type(before).__name__,
                    after_type=type(out).__name__,
                    before_name=before._name,
                    after_name=after_name if after_name is not None else repr(out),
                )
            )
        return out

    return wrapper


@contextmanager
def trace_rewrites():
    """Record every rewrite rule that fires inside the block.

    Wraps the ``_simplify_down``/``_simplify_up``/``_lower`` hooks on every
    ``ArrayExpr`` subclass for the duration of the context and yields a
    :class:`Trace` of what fired.  Diagnostic tool — patches classes, so not
    thread-safe, and expression classes first imported inside the block are
    not captured.

    >>> import dask_array as da
    >>> x = da.ones((10, 10), chunks=(5, 5))
    >>> with da.trace_rewrites() as t:  # doctest: +SKIP
    ...     ((x + 1)[:2]).expr.simplify()
    >>> t  # doctest: +SKIP
    trace: 3 rewrites
       1 x Elemwise._simplify_up: SliceSlicesIntegers -> Elemwise
       ...
    """
    out_trace = Trace()
    patched = []
    for cls in _expr_classes():
        for hook, phase in _REWRITE_HOOKS.items():
            if hook in cls.__dict__:
                orig = cls.__dict__[hook]
                setattr(cls, hook, _traced_hook(orig, hook, phase, out_trace))
                patched.append((cls, hook, orig))
    try:
        yield out_trace
    finally:
        for cls, hook, orig in reversed(patched):
            setattr(cls, hook, orig)


# ---------------------------------------------------------------------------
# Phase-by-phase optimization report
# ---------------------------------------------------------------------------


@dataclass
class PhaseStats:
    name: str  # "raw" | "simplified" | "lowered" | "fused"
    nodes: int
    tasks: int | None  # None for un-lowered phases
    read_bytes: float  # sum of leaf nbytes (nan when unknown)
    transfer_min: float  # sum of node transfer_bytes estimates
    transfer_max: float
    seconds: float | None  # time to produce this phase from the previous


def _phase_stats(name, expr, tasks=None, seconds=None):
    nodes = [n for n in expr.walk() if isinstance(n, ArrayExpr)]
    read = 0.0
    tmin = 0.0
    tmax = 0.0
    for n in nodes:
        try:
            transfer = n.transfer_bytes
            tmin += transfer.min
            tmax += transfer.max
            if not any(isinstance(d, ArrayExpr) for d in n.dependencies()):
                read += n.nbytes
        except Exception:
            read = tmin = tmax = math.nan  # unknown beats wrong
            break
    return PhaseStats(name, len(nodes), tasks, read, tmin, tmax, seconds)


def _task_count(lowered_expr):
    return len(Expr.__dask_graph__(lowered_expr))


def _fmt_bytes(v):
    return "?" if math.isnan(v) else format_bytes(int(v))


@dataclass(repr=False)
class Explain:
    phases: list[PhaseStats]
    simplify_rules: Counter  # rule -> count, e.g. {"Elemwise._simplify_up": 9}
    lower_rules: Counter
    fusion_groups: list[int]  # ops covered by each FusedBlockwise
    _simplify_trace: Trace
    _lower_trace: Trace

    def __repr__(self):
        lines = [
            f"{'phase':<11} {'nodes':>5} {'tasks':>7} {'read':>10} {'transfer(min)':>13} {'transfer(max)':>13} {'time':>7}"
        ]
        for p in self.phases:
            tasks = "-" if p.tasks is None else str(p.tasks)
            seconds = "-" if p.seconds is None else f"{p.seconds * 1000:.0f} ms"
            lines.append(
                f"{p.name:<11} {p.nodes:>5} {tasks:>7} {_fmt_bytes(p.read_bytes):>10} "
                f"{_fmt_bytes(p.transfer_min):>13} {_fmt_bytes(p.transfer_max):>13} {seconds:>7}"
            )
        for title, phase_trace in (
            ("simplify rules:", self._simplify_trace),
            ("lower rules:", self._lower_trace),
        ):
            if phase_trace.records:
                lines.append(title)
                lines.extend(_rule_lines(phase_trace.records))
        if self.fusion_groups:
            groups = ", ".join(f"{n} ops" for n in sorted(self.fusion_groups, reverse=True))
            lines.append(f"fusion: {len(self.fusion_groups)} group(s) ({groups})")
        return "\n".join(lines)


def explain(x):
    """Report how each optimization phase changes an expression.

    Runs simplify → lower → fuse on ``x`` (an Array or ArrayExpr) and returns
    an :class:`Explain` whose repr shows, per phase, the node count, task
    count, bytes read at the leaves, transfer-byte estimates (see
    ``ArrayExpr.transfer_bytes`` — optimistic before lowering, since
    alignment rechunks don't exist yet), and the rewrite rules that fired.
    "read" sums the nbytes of dependency-free nodes, so synthetic sources
    (``ones``, ``random``) count as if they were read.

    Builds task graphs for the raw and optimized forms to count tasks, so it
    costs about as much as graph construction — nothing computes.

    >>> import dask_array as da
    >>> x = da.ones((100, 100), chunks=(10, 10))
    >>> print(da.explain((x + 1).sum(axis=0)[:20]))  # doctest: +SKIP
    """
    expr = x.expr if hasattr(x, "expr") else x

    with trace_rewrites() as simplify_trace:
        t0 = time.perf_counter()
        simplified = expr.simplify()
        simplify_seconds = time.perf_counter() - t0
    with trace_rewrites() as lower_trace:
        t0 = time.perf_counter()
        lowered = simplified.lower_completely()
        lower_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    fused = lowered.fuse()
    fuse_seconds = time.perf_counter() - t0

    from dask_array._blockwise import FusedBlockwise

    fusion_groups = [len(n.exprs) for n in fused.walk() if isinstance(n, FusedBlockwise)]

    simplify_rules = Counter(r.rule for r in simplify_trace.records if r.phase == "simplify")
    lower_rules = Counter(r.rule for r in lower_trace.records if r.phase == "lower")

    return Explain(
        phases=[
            _phase_stats("raw", expr, tasks=_task_count(expr.lower_completely())),
            _phase_stats("simplified", simplified, seconds=simplify_seconds),
            _phase_stats("lowered", lowered, tasks=_task_count(lowered), seconds=lower_seconds),
            _phase_stats("fused", fused, tasks=_task_count(fused), seconds=fuse_seconds),
        ],
        simplify_rules=simplify_rules,
        lower_rules=lower_rules,
        fusion_groups=fusion_groups,
        _simplify_trace=simplify_trace,
        _lower_trace=lower_trace,
    )
