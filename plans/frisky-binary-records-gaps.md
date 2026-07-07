# Close the binary-records gaps (graph-build cost at scale)

Follows on from `frisky-rust-task-gen.md`. That plan moved task-graph
*generation* into Rust. This one is narrower and cost-driven: for very large
graphs the client-side pain is building millions of task objects in Python, so
the goal is to maximize the fraction of tasks emitted as **binary records
chunks** (`to_records_chunk` — one `bytes` blob per layer, O(1) Python objects)
and minimize anything that materializes O(N) Python record tuples.

## The four build tiers (ascending client cost)

Each lowered layer contributes its records one of four ways (see
`_walk_record_chunks` in `dask_array/_frisky/collect.py`):

- `binary` — Rust `to_records_chunk` → one `bytes` blob. The goal.
- `native_tuples` — Rust layer exists but declines binary → `to_task_records`
  builds N Python tuples in Rust. O(tasks).
- `adapter` — no native layer → `GraphRecordsLayer` runs the expr's Python
  `_layer()` and translates it. O(tasks), most work.
- `fallback` — `_check_frisky_supported` rejects the whole graph → stock dask.

## Feedback system (already built — use it, extend it)

- **`bench/tier_probe.py`** — cluster-free (<1s). Walks a collection's lowered
  tree and reports the **task-weighted** tier split plus the top non-binary
  layers by task volume. This is the primary signal: a slice succeeds when the
  target op's `binary %` jumps. Drop your own expressions into `corpus()`.
- **`bench/diff_layers.py`** — cluster-free (<1s) distinct-data per-key
  differential of the records path vs dask. The correctness net for a new
  binary layer; add cases as you port.
- **`bench/roundtrip_layers.py`** — on-cluster, asserts Frisky actually ran
  (not a silent fallback). Note: its engagement spy (`_frisky_compute_collections`)
  and `diff_adversarial.py`'s have **bit-rotted** against frisky's current API
  (2-arg vs 3-arg) — refresh the spy when you touch them.
- **Whole suite under `--scheduler=frisky`** (autouse fixture,
  `dask_array/tests/conftest.py`) — the standing correctness backstop. A passing
  test proves correctness, *not* engagement (an unknown-chunk test passes by
  falling back). Pair suite runs with `tier_probe` for engagement.

## What the instrument found (calibrate before trusting these ratios)

The tier is **operand-sensitive** — two easy mistakes bias the numbers, both hit
during this investigation:

- `ones @ ones` and `A @ A` are *self*-contraction (same source twice). A
  matmul of **distinct** sources (`A @ B`, `A≠B`) is **already binary**. Only
  repeated operands collide.
- `da.from_array(np.array(...))` embeds data (eager slice); a plain-ndarray
  source is a different path from a lazy getter (zarr/h5py). And two
  `np.zeros(same_shape)` tokenize identically → same `_name` → accidental
  self-op.

With those controlled, the real non-binary tasks concentrate in three places,
in the agreed order of work:

### Slice 1 — FusedBlockwise repeated-operand fan-in — DONE (2026-07-06)

Landed: `_site_based_spec` + `_walk_sites` in `dask_array/_frisky/fused_blockwise.py`
put `A@A`, `A@A.T` (Gram), `x.T + x`, `x*x.T` on the binary path (parity with
distinct-source contraction). Verified: new tests
`test_repeated_operand_fused_blockwise_uses_binary_records`, value-correct on a
real Frisky cluster with non-symmetric data (arg order right), and the **full
suite is green under `--scheduler=frisky`** (3133 passed). Rust needed no change
(the `FusedBlockwiseLayer` `dep_slots` grammar already allows repeated sources).

**But profiling (`bench/tier_probe.py` + a derivation timer) found the real
contraction build cost, and it is NOT tuples-vs-binary — see Slice 1b.**

### Slice 1b — probe-only canonical validation — DONE (2026-07-06)

The contraction derivation bottleneck turned out NOT to be per-block
`_task()`/`_canonical()` *construction* (cheap) but the per-block **`canon !=
canon0` comparison**: comparing canonical subgraphs invokes `Task.__eq__` →
`tokenize` → `cloudpickle.dumps` per inner Task (profiled: 18 of 21s). Both the
`uniform` path (line ~214, hit by distinct-source `A@B`) and the new
`_site_based_spec` did it per block.

Fix: validate block-independence (canonical) on a **probe sample** only
(`_probe_blocks`), keeping the exact per-block coord reads (nothing inferred) —
the same bet `_validate_broadcast` already makes on the broadcast path. Result,
measured: distinct `A@B` 6477ms→429ms (15×) and same-source `A@A.T` →670ms at
27k blocks; a 216k-block matmul 60s→3.8s. Verified: `diff_layers` value-correct,
full suite green under `--scheduler=frisky` (3133 passed). The remaining cost is
the honest O(blocks) `_task()` floor (same as `_slow_records`).

Follow-up if million-block graphs need sub-second: the fully **analytical**
derivation below removes even the per-block `_task()`.

### Slice 1b-analytical — remove the per-block `_task()` floor — DONE (2026-07-06)

`_analytical_site_spec` in `_frisky/fused_blockwise.py`: infers each input SITE's
per-dimension coord projection (`source_block[d] = out_id[o]` or const) from
one-dim bump probes, then generates every block's slots by arithmetic — no
per-block `_task()`. `_walk_sites`'s block-independent site order lets it key
projections by site index, so it handles distinct- AND same-source contractions
in one path (matmul/tensordot/einsum, Gram `A@A.T`, `A.T@A`). Gated to fall
through to the exact `_fast_spec_uniform`/`_site_based_spec` on any non-index-
preserving map (reversed axis, stride/offset) via a `Δ==1` inference check +
probe-canonical + probe-projection validation. `_fast_spec` is now
`_analytical_site_spec() or _fast_spec_uniform() or _site_based_spec()`.

Measured (matmul & gram, 27k blocks): `to_records_chunk` ~36ms (was 414–672ms
exact, ~6.5s original — ~180×). Correctness: exact ordered match to the
value-verified `_site_based_spec` across all tested shapes; `diff_layers` green;
full suite green under `--scheduler=frisky` (3133 passed). Reviewed by an
independent agent focused on projection-inference soundness.

### Slice 1b-analytical notes (superseded)

`to_records_chunk` for *any* contraction (distinct-source `A@B` via the `uniform`
path **and** same-source via `_site_based_spec`) is O(blocks) with a per-block
`e._task(bid)` materialization **and** a per-block `_canonical()` — measured ~234ms
/1000 blocks, ~6.5s/27k, ~60s/216k, i.e. ~10× slower to *derive* than
`_slow_records` (which skips `_canonical`). The binary chunk is compact to *ship*
but slow to *build*, so contractions don't yet get the graph-build win at scale.
This is **pre-existing for all contractions**; Slice 1 reached parity with it, it
did not cause it.

The fix (precedented by `_broadcast_spec`/`_validate_broadcast`, which already do
this for the elementwise/broadcast case): prove block-independence on a few
**probe** blocks instead of all, derive a per-site **coord projection** (the affine
map output-block-id → source-block-id) from the probes, then generate every
block's `dep_slots` by integer arithmetic — no per-block `_task()`/`_canonical()`.
Should cut contraction derivation from O(blocks × task-materialization) to
O(blocks × cheap-arithmetic), a large constant-factor win on matmul/tensordot/
einsum. Correctness risk is the projection inference — pin it with `tier_probe`
(build time) + a working cluster-free value differential (the current
`bench/diff_layers.py` resolver is bit-rotted; refresh it or use the on-cluster
value check from this session).

### Slice 1c — FusedBlockwise repeated-operand fan-in (original notes)

Bites `A@A`, `A@A.T` (Gram matrices — common in stats/ML), `A*A`, `x.T + x`,
and map_overlap. Distinct-source contractions are fine; this is specifically
*the same source read more than once per fused block*.

Root cause: `_fast_spec` (`dask_array/_frisky/fused_blockwise.py:162`) labels
each fused input by **source name only** (`_input_label = ("__in__", key[0])`),
so two inputs from one source collide (`len(set(labels)) != len(labels)`) and it
bails to `_slow_records` (O(N) Python tuples).

The byte grammar is **not** the blocker — `common.rs` already has `ArgSlot::List`
and `Dep{name_idx, COORD}`, so multiple source blocks under one name are
expressible. The work is Python `_fast_spec` (+ the Rust `FusedBlockwiseLayer`
builder / `dep_slots`): disambiguate inputs by position and emit ordered `Dep`
slots. **Likely no protocol bump.**

Known wrinkle to design for: at the **diagonal** (e.g. `A@A.T`, block `i==j`)
the two source blocks coincide and dask dedups the deps, so fan-in drops 2→1.
The fast-spec's uniformity check must admit *per-block-varying* fan-in, not just
relabel. Confirm with the `why_matmul`/`trace_fastspec` style probes (a fused
block at `(0,0,1)` shows the collision; block `(0,0,0)` hides it).

TDD entry point: pick `A@A.T`, `x.T + x`, `A*A` at a few block counts. Behavior
to pin first — `tier_probe` shows them `binary` (today `native_tuples`), and
`diff_layers` + `roundtrip_layers` stay correct. Align on those cases before
touching `_fast_spec`.

Related cleanup (same file): `_fast_spec` *derivation* is O(blocks) Python even
when the result is binary (the per-block loop), unless `_broadcast_spec`'s
fast-path validates. Widening that fast-path cuts a hidden build cost on the
happy path.

### Slice 2 — ArgChunk binary

argmax/argmin chunk step (`dask_array/reductions/_arg_reduction.py`), one task
per block, always `native_tuples`. Verify the decline reason (compute shape or
per-task kwarg) and port if small. Self-contained.

### Slice 3 — FromArray-getter binary (pervasive; needs frisky)

Every lazy-IO read block is a Python tuple today: `FromArrayGetterLayer`
(`crates/dask-array-python/src/from_array.rs`) implements `to_task_records` but
has **no `to_records_chunk`** — it deliberately bypasses `common.rs` because the
data node has a bare-string key (`original-<name>`) and a literal array value,
and `Literal` slots aren't binary-expressible.

Approach (mrocklin): **don't** encode the array into the grammar. Split into (a)
a tiny **Python holder record** — the single `(original-<name>, identity,
(array,), {}, [])` data node, left on the plain-records side of
`collect_record_chunks`'s `(chunks, records, chunk_groups)` return — and (b) a
**Rust binary layer for the N getitem tasks**: each `getter(TaskRef("original-
<name>"), (slice, ...))`. The slices are derived from chunk sizes (Rust already
does this in `dim_slices`); the func (`getter`) is shared; only the per-block
slice varies (`ArgSlot::Index`, already supported).

Open design points for slice 3:
- The getter tasks reference the holder by a **bare-string key**, but `Dep`
  slots reference `(name_idx, COORD)` → `str((name, *coord))`. Need a
  bare-name dep (new SLOT tag or a data-node-key convention) — a small grammar
  addition, `RECORDS_PROTOCOL_VERSION` bump + matching `records_proto.rs`
  (`CHUNK_GRAMMAR_VERSION`) in the frisky repo.
- `_walk_record_chunks` currently treats a node as *either* a chunk *or* plain
  records. From_array needs to contribute **both** (one holder record + one
  binary getter chunk). Decide: let a layer return `(chunk, extra_records)`, or
  lower from_array into two exprs.

## Scope

Editing the frisky repo (`records_proto.rs`, `CHUNK_GRAMMAR_VERSION`) is in
scope for the protocol-bumping work (slice 3). Slices 1–2 should need no
protocol change — keep them dask-array-only and land them first.

## Non-goals

- The adapter tail (setitem/histogram/diagonal/vindex) — `tier_probe` shows it's
  ~0.7% of tasks; irrelevant to build cost. Leave on the adapter.
- The unknown-chunk fallback (`bool mask`, `unique`) — real but second-order
  (~a few %); the guard in `_check_frisky_supported` is over-broad (records
  generate fine with `nan` sizes, since structure is known), but it's a separate
  track. Note it, don't bundle it here.
