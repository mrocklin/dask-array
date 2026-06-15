"""Generic records adapter: reuse an expr's own legacy ``_layer()`` graph.

Most of the *specialized tail* (diagonal, setitem, unique, bincount, histogram,
fancy/boolean indexing, …) lowers to standard ``dask._task_spec`` nodes — ``Task``,
``Alias``, ``DataNode``. Rather than reimplement each one's task structure in Rust
(perf is deferred for the tail), an expr can opt into the records path by reusing
the graph its ``_layer()`` already builds and translating those nodes into the
flat ``(key, func, args, kwargs, deps)`` records Frisky submits.

This is the generalization of the "reuse the expr's own ``_task``/``_layer``"
technique that ported ``random``: zero reconstruction, faithful by construction
(the suite validates ``_layer`` directly), and it keeps the rest of a graph on the
fast Rust path — only this node materializes its (small) subgraph in Python.

``collect_task_records`` falls back to this adapter for any node that has no
native ``_frisky_layer`` (or whose layer raises ``NotImplementedError`` for the
given variant), so no per-expr wiring is needed. It calls the expr's ``_layer()``
directly; that method builds the legacy graph without re-entering the records
path, so there is no recursion and the dask/correctness path is unchanged.

Key normalization: dask-expr emits block coordinates as ``np.int64``; the Rust /
from_array layers emit plain ``int``. Frisky matches a dependency by the *string*
of its key, and ``str(('x', np.int64(0)))`` != ``str(('x', 0))``. So every key —
the output key, each dep key, and every ``TaskRef`` embedded in args/kwargs — is
normalized to plain ints to match the canonical form the other layers produce.
"""

from __future__ import annotations

import numbers

import toolz

from dask._task_spec import Alias, DataNode, GraphNode, NestedContainer, Task, TaskRef


def _norm_key(key):
    """Canonicalize a dask key: numpy ints in a tuple coord become plain ints."""
    if isinstance(key, tuple):
        return tuple(int(k) if isinstance(k, numbers.Integral) else k for k in key)
    return key


def _resolve(arg):
    """Resolve a ``_task_spec`` argument into a plain Python value the Frisky
    records path understands: a ``TaskRef`` (canonical key) marks a dependency,
    ``NestedContainer`` becomes the plain list/tuple/set/dict, ``DataNode`` its
    value, and literals pass through. An inline nested ``Task``/``Alias`` can't be
    expressed in a flat record, so raise ``NotImplementedError`` — the caller then
    falls back to stock dask for the whole graph."""
    if isinstance(arg, TaskRef):
        return TaskRef(_norm_key(arg.key))
    if isinstance(arg, NestedContainer):
        resolved = [_resolve(a) for a in arg.args]
        if arg.klass is dict:
            return dict(zip(resolved[::2], resolved[1::2]))
        return arg.klass(resolved)
    if isinstance(arg, DataNode):
        return arg.value
    if isinstance(arg, (Task, Alias)):
        raise NotImplementedError(f"inline {type(arg).__name__} in args")
    if isinstance(arg, list):
        return [_resolve(a) for a in arg]
    if isinstance(arg, tuple):
        return tuple(_resolve(a) for a in arg)
    if isinstance(arg, dict):
        return {k: _resolve(v) for k, v in arg.items()}
    return arg


def _record(key, node):
    """Translate one ``_task_spec`` graph node into a Frisky task record."""
    out_key = str(_norm_key(key))
    if isinstance(node, Task):
        deps = sorted(str(_norm_key(k)) for k in node.dependencies)
        args = tuple(_resolve(a) for a in node.args)
        kwargs = {k: _resolve(v) for k, v in (node.kwargs or {}).items()}
        return (out_key, node.func, args, kwargs, deps)
    if isinstance(node, Alias):
        return (out_key, toolz.identity, (TaskRef(_norm_key(node.target)),), {}, [str(_norm_key(node.target))])
    if isinstance(node, DataNode):
        return (out_key, toolz.identity, (node.value,), {}, [])
    if isinstance(node, NestedContainer):
        # The key's value is itself a container of refs (e.g. a list of blocks);
        # identity over the resolved container yields it once deps are filled in.
        deps = sorted(str(_norm_key(k)) for k in node.dependencies)
        return (out_key, toolz.identity, (_resolve(node),), {}, deps)
    if isinstance(node, GraphNode):
        raise NotImplementedError(f"unhandled GraphNode {type(node).__name__}")
    if isinstance(node, tuple):
        # A bare tuple is ambiguous (could be a legacy (func, *args) task); don't
        # guess — fall back to stock dask for this graph.
        raise NotImplementedError("bare tuple graph value")
    # A bare non-node value is data (e.g. setitem stores a plain ndarray block).
    return (out_key, toolz.identity, (node,), {}, [])


class GraphRecordsLayer:
    def __init__(self, expr):
        self.expr = expr

    def to_dask_graph(self):
        # The legacy graph itself — used only if an expr routes its dask path
        # through _frisky_layer (most opt-in exprs don't; they keep _layer legacy).
        return self.expr._layer()

    def to_task_records(self):
        return [_record(key, node) for key, node in self.expr._layer().items()]
