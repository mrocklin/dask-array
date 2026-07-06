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
on a spread of probe blocks before the fast path is taken — otherwise we fall back
to the per-block path:

  * **Block-independence:** the subgraph differs across blocks only in its keys.
    Checked by canonicalizing each probe block's subgraph (renaming internal keys
    to their expr name and external inputs to their source name) and comparing it
    to block 0's — a block-dependent func/literal/wiring makes them differ.
  * **Input mapping:** each output block's external source blocks are
    ``_broadcast_block_id(source.numblocks, block_id)``. Checked against the real
    ``_task``'s dependencies (a transpose/contraction, or a source referenced at
    several blocks per output, makes them differ).
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
        shared, sources = fast
        try:
            from dask_array._frisky.base import _rust
        except ImportError as exc:
            raise NotImplementedError from exc
        return _rust.FusedBlockwiseLayer(
            self.expr._name,
            shared,
            [int(n) for n in self.expr.numblocks],
            [(name, [int(n) for n in numblocks]) for name, numblocks in sources],
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
        shared, sources = fast
        name = self.expr._name
        numblocks = self.expr.numblocks
        records = []
        for bid in product(*(range(n) for n in numblocks)):
            refs = []
            dep_keys = []
            for src_name, src_nb in sources:
                sk = (src_name, *_broadcast_block_id(src_nb, bid))
                dep_key = str(sk)
                refs.append(TaskRef(dep_key))
                dep_keys.append(dep_key)
            records.append((str((name, *bid)), shared, tuple(refs), _EMPTY_KWARGS, dep_keys))
        return records

    def _fast_spec(self):
        """Shared fused callable + source mapping, or ``None`` to fall back.

        Returns ``None`` unless the block-0 task is the expected
        ``_execute_subgraph`` shape, every external input is a direct dependency,
        and the block-independence + input-mapping checks pass on probe blocks."""
        e = self.expr
        numblocks = e.numblocks
        if not numblocks:
            return None

        block0 = (0,) * len(numblocks)
        task0 = e._task((e._name, *block0), block0)
        if task0.func is not _execute_subgraph or len(task0.args) < 3:
            return None
        subgraph0, outkey0, inkeys0 = task0.args[0], task0.args[1], task0.args[2]

        deps_by_name = {d._name: d for d in e.dependencies()}
        # Each input label -> (source name, source numblocks) for broadcast mapping.
        sources = []
        for ik in inkeys0:
            name = ik[0] if isinstance(ik, tuple) else ik
            dep = deps_by_name.get(name)
            if dep is None:
                return None
            sources.append((dep._name, dep.numblocks))

        if not self._validate(task0, sources, numblocks):
            return None

        shared = _FusedSubgraph(subgraph0, outkey0, inkeys0)
        return shared, sources

    # --- validation -------------------------------------------------------

    def _validate(self, task0, sources, numblocks):
        """Both fast-path assumptions must hold on a spread of probe blocks:
        the subgraph is block-independent (same up to key renaming) and the
        external source blocks are the broadcast-mapped ones."""
        e = self.expr
        canon0 = self._canonical(task0)
        for bid in self._probe_blocks(numblocks):
            task = e._task((e._name, *bid), bid)
            if task.func is not _execute_subgraph:
                return False
            mine = {(n, *_broadcast_block_id(nb, bid)) for n, nb in sources}
            if mine != set(task.dependencies):
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
            rename[ik] = ("__in__", ik[0] if isinstance(ik, tuple) else ik)
        canon = {rename[k]: t.substitute(rename, key=rename[k]) for k, t in subgraph.items()}
        canon_out = rename.get(outkey, outkey)
        return canon, canon_out

    @staticmethod
    def _probe_blocks(numblocks):
        """A small spread of blocks that varies each axis independently (plus the
        corners and a diagonal), to catch axis-dependent mappings/subgraphs."""
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
