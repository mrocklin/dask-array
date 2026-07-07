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
        """Shared fused callable + per-block source mapping, or ``None`` to fall
        back to ``_slow_records``.

        Three derivations, cheapest first:

        - ``_analytical_site`` infers each input's per-dimension coord projection
          from a few probe blocks and generates every block by arithmetic — no
          per-block ``_task()``. Handles both distinct- and repeated-source
          contractions (matmul/tensordot/einsum, Gram ``A@A.T``), the common case
          on large graphs.
        - ``_uniform`` / ``_site_based`` are the exact per-block derivations, kept
          as correctness fallbacks for shapes the projection can't model (a
          reversed axis and other non-index-preserving block maps). They read each
          block's real ``_task``, so they are O(blocks) but validate on probes so
          they are no longer dominated by per-block canonical comparisons."""
        return self._analytical_site_spec() or self._fast_spec_uniform() or self._site_based_spec()

    def _fast_spec_uniform(self):
        """Shared fused callable + source mapping for the case where each output
        block reads any one source at most once, or ``None`` to fall back.

        Returns ``None`` unless the block-0 task is the expected
        ``_execute_subgraph`` shape, every external input is a direct dependency,
        and the block-independence + input-mapping checks pass for every block.
        A source read *twice* in one block (Gram matrix, ``x.T + x``) collides on
        the source-name label here and returns ``None`` — ``_site_based_spec``
        handles it."""
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

        # Contraction / non-broadcast fallback: read each block's source coords
        # exactly. Block-independence (canonical) is validated on a PROBE sample
        # only — a per-block canonical comparison is a cloudpickle tokenize per
        # inner Task (``Task.__eq__``) that dominates the derivation ~10x on large
        # contractions; the fused subgraph is uniform by construction, the same
        # bet ``_validate_broadcast`` makes on the broadcast path just above.
        for bid in self._probe_blocks(numblocks):
            if self._canonical(e._task((e._name, *bid), bid)) != canon0:
                return None

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
            slots_by_label = {}
            for ik, label in zip(inkeys, labels):
                slot = self._dep_slot(ik, dep_idx_by_name)
                if slot is None:
                    return None
                slots_by_label[label] = slot
            dep_slots.append([slots_by_label[label] for label in labels0])

        return shared, dep_names, dep_slots

    def _site_based_spec(self):
        """Fast-path spec that also admits a source read more than once per block.

        Binds each output block's inputs by their structural *site* in the fused
        subgraph (the ordered positions that reference an external source block)
        rather than by source name, so two reads of one source no longer collide.
        The shared subgraph is taken from a *maximal* block — one whose sites are
        all distinct — so ``_execute_subgraph`` has a seed label per site. A
        coincident block (two sites landing on the same source block, e.g. a Gram
        matrix's diagonal) passes the same ref for both sites; the subgraph seeds
        both labels with that data, which is exactly what the deduped per-block
        task computes. Returns ``None`` (fall back to ``_slow_records``) on any
        shape it can't verify."""
        e = self.expr
        numblocks = e.numblocks
        if not numblocks:
            return None
        dep_names = [d._name for d in e.dependencies()]
        dep_idx_by_name = {name: i for i, name in enumerate(dep_names)}
        all_bids = list(product(*(range(n) for n in numblocks)))

        # Block-independence check: canonicalize a reference block and compare a
        # few PROBE blocks to it. Comparing canonical subgraphs is a cloudpickle
        # tokenize per inner Task (``Task.__eq__``) and dominates the whole
        # derivation at scale, so we do it on the probe sample only — the fused
        # subgraph is uniform by construction, the same bet ``_validate_broadcast``
        # makes. The per-block slot pass below still reads each block's *exact*
        # source coords, so nothing is inferred; only the wiring is trusted.
        ref_task = e._task((e._name, *all_bids[0]), all_bids[0])
        if ref_task.func is not _execute_subgraph or len(ref_task.args) < 3:
            return None
        canon0 = self._canonical(ref_task)
        n_sites = None
        for bid in self._probe_blocks(numblocks):
            task = e._task((e._name, *bid), bid)
            if task.func is not _execute_subgraph or len(task.args) < 3:
                return None
            if self._canonical(task) != canon0:
                return None
            sites = self._walk_sites(task.args[0], task.args[1], set(task.args[2]))
            if sites is None:
                return None
            if n_sites is None:
                n_sites = len(sites)
            elif len(sites) != n_sites:
                return None
        if not n_sites:
            return None

        # Per-block source slots (exact, cheap) + a maximal block (all sites
        # distinct) whose subgraph becomes the shared seed.
        maximal = None
        dep_slots = []
        for bid in all_bids:
            task = e._task((e._name, *bid), bid)
            if task.func is not _execute_subgraph or len(task.args) < 3:
                return None
            subgraph, outkey, inkeys = task.args[0], task.args[1], task.args[2]
            try:
                inkeys_set = set(inkeys)
            except TypeError:
                return None
            sites = self._walk_sites(subgraph, outkey, inkeys_set)
            # Faithful traversal: the sites must reproduce the block's external
            # inputs exactly (else the arg list would be wrong), and stay uniform.
            if sites is None or len(sites) != n_sites or set(sites) != inkeys_set:
                return None
            slots = []
            for key in sites:
                slot = self._dep_slot(key, dep_idx_by_name)
                if slot is None:
                    return None
                slots.append(slot)
            dep_slots.append(slots)
            if maximal is None and len(set(sites)) == n_sites:
                maximal = (subgraph, outkey, tuple(sites))

        if maximal is None:
            return None
        return _FusedSubgraph(*maximal), dep_names, dep_slots

    def _analytical_site_spec(self):
        """Analytical fast path: infer each input SITE's per-dimension coord
        projection from probe blocks, then generate every block's slots by integer
        arithmetic — no per-block ``_task()``. Subsumes the uniform and site-based
        exact loops for the common shapes.

        ``_walk_sites`` gives a block-independent site order, so site j is a fixed
        subgraph position across blocks even when two sites coincide (a Gram
        diagonal); each site gets its own projection, keyed by index not source
        name. A projection maps an output-block dim to a source-block dim
        one-to-one (``source_block[d] = block_id[o]``) or holds it constant. Any
        map that isn't a per-dim index-preserving projection (e.g. a reversed
        axis, where ``source_block[d] = n-1-block_id[o]``) makes inference bail to
        the exact path.

        Returns ``None`` (fall through) unless: every probe block is the expected
        ``_execute_subgraph`` shape with a stable canonical subgraph
        (block-independence), and the inferred projections reproduce each probe
        block's exact reads. Both are checked on the probe sample only — the same
        trust the exact site-based path and ``_validate_broadcast`` use — so the
        cost stays O(probes) not O(blocks)."""
        e = self.expr
        nb = e.numblocks
        if not nb:
            return None
        dep_names = [d._name for d in e.dependencies()]
        dep_idx = {name: i for i, name in enumerate(dep_names)}
        zero = (0,) * len(nb)

        t0 = e._task((e._name, *zero), zero)
        if t0.func is not _execute_subgraph or len(t0.args) < 3:
            return None
        sites0 = self._walk_sites(t0.args[0], t0.args[1], set(t0.args[2]))
        if not sites0:
            return None
        n_sites = len(sites0)
        site_src = [k[0] for k in sites0]
        if any(s not in dep_idx for s in site_src):
            return None
        canon0 = self._canonical(t0)
        proj = [[("const", int(c)) for c in k[1:]] for k in sites0]

        # Infer each site's projection by bumping one output dim at a time and
        # seeing which source dim tracks it (delta +1) vs stays constant.
        for o in range(len(nb)):
            if nb[o] <= 1:
                continue
            bid = list(zero)
            bid[o] = 1
            t = e._task((e._name, *bid), tuple(bid))
            if t.func is not _execute_subgraph or len(t.args) < 3:
                return None
            sites = self._walk_sites(t.args[0], t.args[1], set(t.args[2]))
            if sites is None or len(sites) != n_sites:
                return None
            for j in range(n_sites):
                if sites[j][0] != site_src[j]:
                    return None  # a site's source must be stable across blocks
                base_coord = tuple(int(c) for c in sites0[j][1:])
                bumped_coord = tuple(int(c) for c in sites[j][1:])
                for d, (cv, bv) in enumerate(zip(base_coord, bumped_coord)):
                    if bv != cv:
                        if bv - cv == 1:
                            proj[j][d] = ("bid", o)
                        else:
                            return None  # non-index-preserving map

        projections = [(dep_idx[site_src[j]], proj[j]) for j in range(n_sites)]
        # If two sites share a source AND a projection they resolve to the same
        # block for every output block, so no block ever has all-distinct sites
        # (no maximal seed block exists). Bail now rather than scan every block
        # fruitlessly — the uniform path handles these (e.g. A*A, x**2).
        if len({(dep_i, tuple(pj)) for dep_i, pj in projections}) != n_sites:
            return None

        def block_slots(bid):
            return [(dep_i, tuple(bid[co] if kind == "bid" else co for kind, co in pj)) for dep_i, pj in projections]

        # Validate on probes: block-independence AND that the projections
        # reproduce the exact reads. Catches a mis-inferred projection before it
        # can emit a wrong graph.
        for bid in self._probe_blocks(nb):
            t = e._task((e._name, *bid), bid)
            if t.func is not _execute_subgraph or self._canonical(t) != canon0:
                return None
            sites = self._walk_sites(t.args[0], t.args[1], set(t.args[2]))
            if sites is None:
                return None
            actual = sorted((dep_idx[k[0]], tuple(int(c) for c in k[1:])) for k in sites)
            if sorted(block_slots(bid)) != actual:
                return None

        # Shared subgraph from a maximal block (projected sites all distinct), so
        # _execute_subgraph has a seed label per site; coincident blocks pass a
        # ref twice, exactly as _site_based_spec does.
        maximal_bid = None
        for bid in product(*(range(n) for n in nb)):
            if len({(dep_i, c) for dep_i, c in block_slots(bid)}) == n_sites:
                maximal_bid = bid
                break
        if maximal_bid is None:
            return None
        tm = e._task((e._name, *maximal_bid), maximal_bid)
        inkeys_m = tuple(tm.args[2])
        sm = self._walk_sites(tm.args[0], tm.args[1], set(inkeys_m))
        if sm is None or len(set(sm)) != n_sites or set(sm) != set(inkeys_m):
            return None
        # Order the slots by the maximal block's inkey order (``args[2]``), not the
        # structural ``_walk_sites`` order. Both are permutations of the same
        # distinct sites, but emitting in inkey order keeps the binary chunk's
        # slot layout identical to the exact per-block paths (block-0 inkey order
        # for distinct sources) — otherwise the two orders differ by hash seed and
        # the layout is non-deterministic. ``projections[j]`` and ``sm[j]`` are the
        # same structural site, so map each inkey to its projection through ``sm``.
        site_of = {sm[j]: j for j in range(n_sites)}
        ordered = [projections[site_of[ik]] for ik in inkeys_m]
        shared = _FusedSubgraph(tm.args[0], tm.args[1], inkeys_m)

        def ordered_slots(bid):
            return [(dep_i, tuple(bid[co] if kind == "bid" else co for kind, co in pj)) for dep_i, pj in ordered]

        dep_slots = [ordered_slots(bid) for bid in product(*(range(n) for n in nb))]
        return shared, dep_names, dep_slots

    @staticmethod
    def _walk_sites(subgraph, outkey, inkeys_set):
        """External-input reference sites of a fused subgraph, in a deterministic,
        block-independent order: depth-first from ``outkey`` through the internal
        wiring, in argument order, appending each external (inkey) reference as it
        is encountered. One entry per *occurrence* (a source used at two sites
        appears twice); internal nodes are walked once. Returns ``None`` if the
        subgraph shape can't be traversed as expected."""
        sites = []
        visited = set()

        def scan(arg):
            if isinstance(arg, TaskRef):
                key = arg.key
                if key in inkeys_set:
                    sites.append(key)
                elif key in subgraph:
                    visit(key)
                return
            if isinstance(arg, (list, tuple)):
                for a in arg:
                    scan(a)
            elif isinstance(arg, dict):
                for a in arg.values():
                    scan(a)

        def visit(key):
            if key in visited:
                return
            visited.add(key)
            task = subgraph.get(key)
            if task is None:
                return
            for a in getattr(task, "args", ()):
                scan(a)

        if outkey not in subgraph:
            return None
        visit(outkey)
        return sites

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
