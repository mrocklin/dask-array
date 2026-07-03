"""Generic records adapter: reuse an expr's own legacy ``_layer()`` graph.

The *specialized tail* (diagonal, setitem, unique, bincount, histogram, fancy/
boolean indexing, overlap, percentile, apply_along_axis, …) is long, and perf is
deferred there. Rather than port a bespoke Rust layer for each, this adapter reuses
the graph an expr's ``_layer()`` already builds and translates it into the flat
``(key, func, args, kwargs, deps)`` records Frisky submits. It's the generalization
of the "reuse the expr's own ``_task``/``_layer``" technique that ported ``random``:
zero reconstruction, faithful by construction (the suite validates ``_layer``
directly), and the rest of a graph stays on the fast Rust path — only this node
materializes its (small) subgraph in Python.

``collect_task_records`` falls back to this adapter for any node with no native
``_frisky_layer`` (or whose layer raises ``NotImplementedError`` for the variant),
so no per-expr wiring is needed. It calls ``_layer()`` directly; that builds the
legacy graph without re-entering the records path, so there's no recursion and the
dask/correctness path is unchanged.

Translating the graph involves three wrinkles:

  - *Key normalization.* dask-expr emits block coords as ``np.int64``; the Rust /
    from_array layers emit plain ``int``, and Frisky matches deps by the *string* of
    the key (``str(('x', np.int64(0)))`` != ``str(('x', 0))``). So every key — output,
    each dep, and every embedded ``TaskRef`` — is normalized to plain ints.
  - *Inline subtasks.* ``_task_spec`` nests freely (``concatenate3([[Task(getitem,
    …)]])``) but a Frisky record is flat. An inline ``Task`` is lifted into its own
    synthesized record and replaced by a ``TaskRef``; an inline ``Alias`` becomes a
    ref to its target (see ``_Flattener``).
  - *Old-style layers.* a few (e.g. overlap) still emit legacy ``(func, *args)``
    tuples whose cross-layer block refs are bare key tuples. ``convert_legacy_graph``,
    given the full keyset (this layer's keys + every dependency's block keys),
    resolves the tuple-vs-key ambiguity; it's idempotent on already-converted graphs.

If a node still can't be expressed (an unhandled ``GraphNode`` type), the adapter
raises ``NotImplementedError`` and the whole graph falls back to stock dask.

Key normalization: dask-expr emits block coordinates as ``np.int64``; the Rust /
from_array layers emit plain ``int``. Frisky matches a dependency by the *string*
of its key, and ``str(('x', np.int64(0)))`` != ``str(('x', 0))``. So every key —
the output key, each dep key, and every ``TaskRef`` embedded in args/kwargs — is
normalized to plain ints to match the canonical form the other layers produce.
"""

from __future__ import annotations

import numbers

import toolz

from dask._task_spec import Alias, DataNode, GraphNode, NestedContainer, Task, TaskRef, convert_legacy_graph


def _norm_key(key):
    """Canonicalize a dask key: numpy ints in a tuple coord become plain ints."""
    if isinstance(key, tuple):
        return tuple(int(k) if isinstance(k, numbers.Integral) else k for k in key)
    return key


class _Flattener:
    """Resolve a node's args/kwargs into flat record form, collecting the keys it
    depends on and emitting ``extra`` records for any inline subtask.

    Frisky records are flat — args hold ``TaskRef`` deps, plain containers, and
    literals, but no inline-computed subtask. dask's ``_task_spec`` nests freely
    (e.g. ``concatenate3([[Task(getitem, ...), ...]])``). So an inline ``Task`` is
    lifted into its own synthesized record (key = ``"<parent>-subN"``, unique:
    parent keys are unique and N increments) and replaced by a ``TaskRef`` to it;
    an inline ``Alias`` is just a reference to its target."""

    def __init__(self, parent_key):
        self.parent_key = parent_key
        self.extra = []
        self._n = 0

    def resolve(self, arg, deps):
        if isinstance(arg, TaskRef):
            k = _norm_key(arg.key)
            deps.add(str(k))
            return TaskRef(k)
        if isinstance(arg, Alias):
            k = _norm_key(arg.target)
            deps.add(str(k))
            return TaskRef(k)
        if isinstance(arg, DataNode):
            return arg.value
        # NestedContainer subclasses Task, so this must precede the Task check.
        # Inline a list/tuple here — Frisky's lower_dep_refs recurses both and
        # resolves embedded TaskRefs, so no extra task is needed (concatenate3 etc.
        # nest these heavily). dict/set fall through to the generic Task lift below:
        # lower_dep_refs recurses dict *values* but not dict *keys* or sets, so a
        # ref there wouldn't be resolved; the lifted subtask instead runs
        # `to_container` over its already-resolved args, rebuilding them safely.
        if isinstance(arg, NestedContainer) and arg.klass in (list, tuple):
            return arg.klass(self.resolve(a, deps) for a in arg.args)
        if isinstance(arg, Task):
            self._n += 1
            sub_key = f"{self.parent_key}-sub{self._n}"
            sub_deps = set()
            sub_args = tuple(self.resolve(a, sub_deps) for a in arg.args)
            sub_kwargs = {k: self.resolve(v, sub_deps) for k, v in (arg.kwargs or {}).items()}
            self.extra.append((sub_key, arg.func, sub_args, sub_kwargs, sorted(sub_deps)))
            deps.add(sub_key)
            return TaskRef(sub_key)
        if isinstance(arg, GraphNode):
            raise NotImplementedError(f"unhandled inline {type(arg).__name__}")
        if isinstance(arg, list):
            return [self.resolve(a, deps) for a in arg]
        if isinstance(arg, tuple):
            return tuple(self.resolve(a, deps) for a in arg)
        if isinstance(arg, dict):
            return {k: self.resolve(v, deps) for k, v in arg.items()}
        return arg


def _records(key, node):
    """Translate one ``_task_spec`` graph node into one or more Frisky records
    (one for the node, plus any lifted inline subtasks)."""
    out_key = str(_norm_key(key))
    if isinstance(node, Alias):
        target = str(_norm_key(node.target))
        if target == out_key:
            # Self-alias. A name-preserving persisted block is a live Future
            # stored under its own key, which ``convert_legacy_graph`` turns
            # into ``Alias(key -> key)``. The scheduler already holds the
            # data — emit no record (a self-referential one would deadlock);
            # the frisky client registers live future keys as external deps.
            return []
        return [(out_key, toolz.identity, (TaskRef(_norm_key(node.target)),), {}, [target])]
    if isinstance(node, DataNode):
        # A persisted collection's FromGraph maps each block key to the frisky
        # Future holding its data. Emit a real dependency edge to that future's
        # key (so the worker resolves it) rather than a literal arg with no edge,
        # which would flow an unresolved placeholder into the consuming op. The
        # future's ``.key`` is already the scheduler's registered key string.
        # Import Future lazily: this module only runs under Frisky, but a
        # future-free graph must not require Frisky to be importable.
        try:
            from frisky import Future

            is_future = isinstance(node.value, Future)
        except ImportError:
            is_future = False
        if is_future:
            future_key = str(node.value.key)
            if future_key == out_key:
                # Name-preserving persist: this block IS the live future for
                # this very key. It is external data the scheduler already
                # holds — emit no record (a self-referential record would
                # deadlock). The frisky client registers live future keys as
                # external deps, so completeness checks allow references.
                return []
            return [(out_key, toolz.identity, (TaskRef(future_key),), {}, [future_key])]
        return [(out_key, toolz.identity, (node.value,), {}, [])]
    fl = _Flattener(out_key)
    deps = set()
    # NestedContainer subclasses Task — handle the list/tuple case (the key's value
    # is itself a container of block refs) before the Task check: identity over the
    # inlined container yields it once deps are filled in. dict/set containers, and
    # genuine inline subtasks, go through the Task branch (generic to_container lift).
    if isinstance(node, NestedContainer) and node.klass in (list, tuple):
        resolved = fl.resolve(node, deps)
        return [(out_key, toolz.identity, (resolved,), {}, sorted(deps)), *fl.extra]
    if isinstance(node, Task):
        args = tuple(fl.resolve(a, deps) for a in node.args)
        kwargs = {k: fl.resolve(v, deps) for k, v in (node.kwargs or {}).items()}
        return [(out_key, node.func, args, kwargs, sorted(deps)), *fl.extra]
    if isinstance(node, GraphNode):
        raise NotImplementedError(f"unhandled GraphNode {type(node).__name__}")
    # A bare non-node value is data (e.g. setitem stores a plain ndarray block).
    # convert_legacy_graph wraps real tasks in nodes, so anything left is data.
    return [(out_key, toolz.identity, (node,), {}, [])]


class GraphRecordsLayer:
    def __init__(self, expr):
        self.expr = expr

    def to_task_records(self):
        # Normalize the legacy graph to _task_spec nodes first: some layers (e.g.
        # overlap) still emit old-style (func, *args) tuples whose cross-layer block
        # references are bare key tuples. convert_legacy_graph resolves the
        # tuple-vs-key ambiguity against a keyset, so pass it this layer's own keys
        # plus every dependency's block keys (else a cross-layer ref looks like data
        # and gets passed literally). It's idempotent on already-converted graphs.
        from dask.core import flatten

        local = self.expr._layer()
        all_keys = set(local)
        for dep in self.expr.dependencies():
            all_keys.update(flatten(dep.__dask_keys__()))
        dsk = convert_legacy_graph(local, all_keys)
        return [rec for key, node in dsk.items() for rec in _records(key, node)]
