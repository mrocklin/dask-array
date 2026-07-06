"""FusedBlockwise records layer.

A ``FusedBlockwise`` is the result of fusing a chain of Blockwise/Elemwise exprs
into one expr — ``Array.optimize()`` does this, and ``Array.compute()`` optimizes
before computing, so idiomatic user code hits it. Its ``_task`` emits exactly ONE
dask ``Task`` per output block::

    _execute_subgraph(subgraph, outkey, inkeys, *source_block_refs)

where ``subgraph`` is a dict of the fused-away inner Tasks (carried as a *literal*
arg and run on the worker by ``_execute_subgraph``), ``inkeys`` are the literal
source-block key-tuples used to seed the subgraph's inputs, and the task's only
real ``dependencies`` are the external source blocks (the inputs to the chain).

The generic ``GraphRecordsLayer`` adapter mistranslates this (it reads the inner
subgraph's internal key-tuples as graph dependencies), so this native layer reads
the flat record straight off each dask ``Task``.

Fast path
---------
Building one fully-materialized fused ``Task`` per output block (``_task`` ->
inner ``_task`` x N -> ``Task.fuse``) is the dominant graph-build cost on
million-block ERA5 graphs (~30x the structural/unfused path). But the inner
subgraph (funcs, scalar args, internal wiring) is *identical* for every output
block — only the external input blocks vary. So we build the subgraph ONCE (from
block 0) and ship it as the task's FUNC, a shared :class:`_FusedSubgraph` callable
that Frisky pickles once (its func cache keys on object identity); each block's
record is then just the cheap per-block source refs. ``_execute_subgraph`` seeds
the (block-0) input labels with whatever data the per-block refs resolve to and
runs the subgraph, so the result is identical to the per-block task.

Two things must hold for this to be correct, and both are *verified* (not assumed)
before the fast path is taken — otherwise we fall back to the per-block path:

  * **Block-independence:** the subgraph differs across blocks only in its keys.
    Checked by canonicalizing each block's subgraph (renaming internal keys
    to their expr name and external inputs to their source name) and comparing it
    to block 0's — a block-dependent func/literal/wiring makes them differ.
  * **Input mapping:** each output block's external source blocks come directly
    from that block's real fused ``_task`` input labels, then are aligned back to
    block 0's input-label order before building the shared callable args.
"""

from __future__ import annotations

from itertools import product

from dask._task_spec import TaskRef, _execute_subgraph

from dask_array._blockwise import _broadcast_block_id

# One shared empty-kwargs dict so Frisky's per-task func cache (keyed on the
# (func, kwargs) object identities) hits across every fused record.
_EMPTY_KWARGS: dict = {}


class _FusedSubgraph:
    """A block-independent fused subgraph as one picklable callable.

    Holds the inner subgraph + its output key + input labels (all from block 0).
    Calling it seeds the input labels with the block's resolved dependency data
    and runs the subgraph — exactly what the per-block ``_execute_subgraph`` task
    does, but built once and shared (so Frisky serializes the subgraph a single
    time instead of once per output block)."""

    __slots__ = ("subgraph", "outkey", "inkeys")

    def __init__(self, subgraph, outkey, inkeys):
        self.subgraph = subgraph
        self.outkey = outkey
        self.inkeys = inkeys

    def __call__(self, *dependencies):
        return _execute_subgraph(self.subgraph, self.outkey, self.inkeys, *dependencies)

    def __reduce__(self):
        return (_FusedSubgraph, (self.subgraph, self.outkey, self.inkeys))


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
        fast = self._fast_records()
        if fast is not None:
            return fast
        return self._slow_records()

    def to_records_chunk(self):
        fast = self._fast_spec()
        if fast is None:
            raise NotImplementedError
        shared, dep_names, dep_slots = fast
        try:
            from dask_array._frisky.base import _rust
        except ImportError as exc:
            raise NotImplementedError from exc
        return _rust.FusedBlockwiseLayer(
            self.expr._name,
            shared,
            [int(n) for n in self.expr.numblocks],
            dep_names,
            dep_slots,
        ).to_records_chunk()

    def _slow_records(self):
        # ``task.dependencies`` is a frozenset, so ``deps`` order is not stable
        # across processes — that's fine: Frisky matches deps by string, and the
        # ``inkeys``<->source-block alignment ``_execute_subgraph`` relies on lives
        # inside ``task.args`` (both derive from the same ``external_deps`` tuple),
        # independent of this list's order.
        return [
            (str(key), task.func, tuple(task.args), task.kwargs or {}, [str(d) for d in task.dependencies])
            for key, task in self._tasks()
        ]

    def _fast_records(self):
        fast = self._fast_spec()
        if fast is None:
            return None
        shared, dep_names, dep_slots = fast
        name = self.expr._name
        numblocks = self.expr.numblocks
        records = []
        for bid, slots in zip(product(*(range(n) for n in numblocks)), dep_slots):
            dep_keys = [self._dep_key(dep_names, slot) for slot in slots]
            refs = [TaskRef(dep_key) for dep_key in dep_keys]
            records.append((str((name, *bid)), shared, tuple(refs), _EMPTY_KWARGS, dep_keys))
        return records

    def _fast_spec(self):
        """Shared fused callable + source mapping, or ``None`` to fall back.

        Returns ``None`` unless the block-0 task is the expected
        ``_execute_subgraph`` shape, every external input is a direct dependency,
        and the block-independence + input-mapping checks pass for every block."""
        e = self.expr
        numblocks = e.numblocks
        if not numblocks:
            return None

        block0 = (0,) * len(numblocks)
        task0 = e._task((e._name, *block0), block0)
        if task0.func is not _execute_subgraph or len(task0.args) < 3:
            return None
        subgraph0, outkey0, inkeys0 = task0.args[0], task0.args[1], task0.args[2]

        try:
            inkeys0 = tuple(inkeys0)
        except TypeError:
            return None
        labels0 = tuple(self._input_label(ik) for ik in inkeys0)
        if any(label is None for label in labels0) or len(set(labels0)) != len(labels0):
            return None
        dep_names = [d._name for d in e.dependencies()]
        dep_idx_by_name = {name: i for i, name in enumerate(dep_names)}

        canon0 = self._canonical(task0)
        shared = _FusedSubgraph(subgraph0, outkey0, inkeys0)

        broadcast = self._broadcast_spec(inkeys0, dep_idx_by_name)
        if broadcast is not None and self._validate_broadcast(canon0, broadcast, numblocks):
            dep_slots = [
                [
                    (dep_idx, tuple(int(c) for c in _broadcast_block_id(source_numblocks, bid)))
                    for dep_idx, _source_name, source_numblocks in broadcast
                ]
                for bid in product(*(range(n) for n in numblocks))
            ]
            return shared, dep_names, dep_slots

        dep_slots = []
        for bid in product(*(range(n) for n in numblocks)):
            task = e._task((e._name, *bid), bid)
            if task.func is not _execute_subgraph or len(task.args) < 3:
                return None
            try:
                inkeys = tuple(task.args[2])
            except TypeError:
                return None
            labels = tuple(self._input_label(ik) for ik in inkeys)
            if any(label is None for label in labels) or len(set(labels)) != len(labels):
                return None
            if set(labels) != set(labels0):
                return None
            if set(inkeys) != set(task.dependencies):
                return None
            if self._canonical(task) != canon0:
                return None
            slots_by_label = {}
            for ik, label in zip(inkeys, labels):
                slot = self._dep_slot(ik, dep_idx_by_name)
                if slot is None:
                    return None
                slots_by_label[label] = slot
            dep_slots.append([slots_by_label[label] for label in labels0])

        return shared, dep_names, dep_slots

    # --- validation -------------------------------------------------------

    def _broadcast_spec(self, inkeys0, dep_idx_by_name):
        deps_by_name = {d._name: d for d in self.expr.dependencies()}
        sources = []
        for ik in inkeys0:
            if not isinstance(ik, tuple) or not ik:
                return None
            dep = deps_by_name.get(ik[0])
            if dep is None:
                return None
            sources.append((dep_idx_by_name[dep._name], dep._name, dep.numblocks))
        return sources

    def _validate_broadcast(self, canon0, sources, numblocks):
        e = self.expr
        for bid in self._probe_blocks(numblocks):
            task = e._task((e._name, *bid), bid)
            if task.func is not _execute_subgraph:
                return False
            expected = {(name, *_broadcast_block_id(source_numblocks, bid)) for _, name, source_numblocks in sources}
            if expected != set(task.dependencies):
                return False
            if self._canonical(task) != canon0:
                return False
        return True

    @staticmethod
    def _canonical(task):
        """Key-independent form of a fused ``_execute_subgraph`` task: rename
        internal subgraph keys to their expr name and external input keys to their
        source name, so two blocks' subgraphs compare equal iff they differ only
        in block ids."""
        subgraph, outkey, inkeys = task.args[0], task.args[1], task.args[2]
        rename = {}
        for k in subgraph:
            rename[k] = (k[0],) if isinstance(k, tuple) else (k,)
        for ik in inkeys:
            rename[ik] = FusedBlockwiseLayer._input_label(ik)
        canon = {rename[k]: t.substitute(rename, key=rename[k]) for k, t in subgraph.items()}
        canon_out = rename.get(outkey, outkey)
        return canon, canon_out

    @staticmethod
    def _input_label(key):
        if not isinstance(key, tuple) or not key:
            return None
        return ("__in__", key[0])

    @staticmethod
    def _dep_slot(key, dep_idx_by_name):
        if not isinstance(key, tuple) or not key:
            return None
        dep_idx = dep_idx_by_name.get(key[0])
        if dep_idx is None:
            return None
        try:
            coord = tuple(int(c) for c in key[1:])
        except (TypeError, ValueError):
            return None
        return dep_idx, coord

    @staticmethod
    def _dep_key(dep_names, slot):
        dep_idx, coord = slot
        return str((dep_names[dep_idx], *coord))

    @staticmethod
    def _probe_blocks(numblocks):
        zero = tuple(0 for _ in numblocks)
        probes = {zero, tuple(n - 1 for n in numblocks)}
        for i, n in enumerate(numblocks):
            if n > 1:
                for v in (n - 1, n // 2):
                    b = list(zero)
                    b[i] = v
                    probes.add(tuple(b))
        probes.add(tuple(min(i, n - 1) for i, n in enumerate(numblocks)))
        return probes
