# Rewrite dask graph generation in Rust

We're migrating the Dask-Array project to use Rust for task graph generation,
and for more efficient handing off between dask-array and frisky.

For context, Frisky is a reimplementation of dask.distributed in Rust and
Dask-Array is a reimplementation of Dask.array with high level query
optimization.  Both are trying to modernize Dask for higher performance.

## Current state and problem

Right now Dask collections generate task graphs in Python with Python Task
objects, and then they hand them to Frisky, which walks over the graph and
serializes them.  This has several steps:

-  Generate
-  Serialize / translate
-  Communicate over a wire
-  Scheduler ingestion

Eventually we'll move the entire expression up to the scheduler, and generate
scheduler state directly from it, really reducing overhead.  For now though,
we're going to be less ambitious and more generic.

Our goal is to generate task graphs in Rust rather than in Python, and then
hand it to the Frisky Client to serialize (no translation hopefully necessary)
and then communicate up to the scheduler.  We're looking for a 5-10x
improvement, not 100x.

## Challenges:

-  dask-array was pure python before, so we now have to include rust
-  dask graphs can be arbitrarily complex, so attempts to abstract around them (like map or reduce layers) tend to fail


# Frisky Rust task generation for dask-array — plan & state

Generate dask-array's array task graphs in **Rust** and feed them to the Frisky
scheduler efficiently, instead of building millions of Python
`dask._task_spec.Task` objects in `_layer()` and then re-serializing them.

Branch: `rust-layers` (worked in the **main dask-array checkout**, not a
worktree — both venvs resolve `dask_array` via an editable install that beats
PYTHONPATH, so a worktree breaks imports). **Committed** as of `47dc790`
(blockwise + creation); Frisky side committed on `array-bulk-client` (`9bf5760`).
Subsequent work should keep committing as it lands (owner authorized).

## Scope decisions (with rationale)

- **Lever 1 + 2, Frisky stays generic.** Generate ordinary tasks in Rust (kill
  the Python codegen) + lean per-task encoding. Frisky only ever consumes
  "a batch of tasks sharing a function" — never anything array-shaped. We
  accept O(N) scheduler ingest for now (not chasing the prior `frisky-bulk-graph`
  branch's O(1) compact protocol, which made Frisky array-aware).
- **No compile-time dependency** between dask-array's Rust crate and any Frisky
  crate. The contract is a data interface, not a shared Rust type.
- **`to_dask_graph` is the correctness oracle.** Each expr's `_layer()` routes
  through the Rust layer's `to_dask_graph`; dask-array's existing single-threaded
  test suite (≈2810 passing) validates the Rust expansion. No separate
  correctness tests needed for the Rust→dask path.
- First targets: **blockwise + creation (ones/zeros/...)**, both done. The
  per-layer file structure exists so more layer types can be farmed to parallel
  agents later (from_array, rechunk, reductions, slicing, transpose, fused...).

## Architecture (current, decided)

A layer = a compact description of one expression's subgraph. The **graph code
lives in Rust** (`crates/dask-array-python/src/`). One pure-Rust `expand()` per
layer produces a neutral **form** held as a Rust `Vec`:

- `Expanded` (common.rs): shared `func`/`kwargs`/`literals` (the only `PyObject`s,
  one set per layer) + `dep_names: Vec<String>` + `tasks: Vec<(coord, Vec<ArgSlot>)>`.
- `ArgSlot` = `Literal(idx)` | `Dep { name_idx, coord }` | `IntTuple(Vec<i64>)`.
  Everything per-task (coords, dep refs) is raw Rust; nothing per-task is Python.

Two **generic** Rust converters consume the form (common.rs):

- `to_dask_graph` → `{key: dask Task}` (builds `Task`/`TaskRef` via pyo3). The
  correctness/legacy path that `_layer()` routes through — validated by the suite.
- `to_task_records` → a plain list of `(key, func, args, kwargs, deps)` records,
  one per task (a Rust-built **mirror of the dask Task**): `key` is
  `str((name, *coord))`, `func`/`kwargs` are the shared layer Python objects,
  `args` is a Python tuple whose dependency positions hold a dask
  `TaskRef(dep_key_tuple)` (literals/values inline), and `deps` is the dep key
  strings. No shared template, no packed buffers — each task is self-contained,
  so any layer kind emits records the same way. (Earlier rounds had a compact
  template + a protocol-2 pickle-splice "bulk" client — removed as too clever /
  non-general; that cleverness is a *future* optimization, not round 1.)

**Per-expr wiring**: `Blockwise._frisky_layer()` (dask_array/_blockwise.py) and
`BroadcastTrick._frisky_layer()` (dask_array/creation/_ones_zeros.py) validate +
normalize, then build the Rust layer. `_layer()` is
`try: self._frisky_layer().to_dask_graph() except NotImplementedError: <legacy>`.
`_frisky_layer` raises NotImplementedError (→ legacy fallback) for: contractions,
new_axes, concatenate, adjust_chunks, ArrayBlockwiseDep, embedded dask
collections, ndim mismatch, **unaligned/un-lowered chunks**, nan chunks.

**Hand-off to Frisky:** `collect_task_records(collection)`
(dask_array/_frisky/collect.py) mirrors dask's `__dask_graph__` walk
(`expr.lower_completely()`, stack over `dependencies()`, dedup by `_name`) calling
`_frisky_layer().to_task_records()` per expr → one flat list of task records.
`Client.submit_tasks` (Frisky, client.rs) serializes them exactly as
`submit`/`map` do: the function is serialized once per distinct `(func, kwargs)`
(cached), `kwargs` bound via `functools.partial`, dependency references
(`TaskRef`/`Future`) in `args` replaced with worker placeholders
(`lower_dep_refs`), then `[u32 func_len][func][args]`, submitted via the existing
`UpdateGraph`. Frisky's scheduler/worker are unchanged.

## What's built & verified

- Rust crate (pyo3 0.23, maturin, cdylib `dask_array._rust`, **no frisky dep**):
  `common.rs` (ArgSlot/Expanded + the `to_dask_graph` and `to_task_records`
  converters), `blockwise.rs`, `creation.rs`, `lib.rs` (module root +
  `protocol_revision`). `PROTOCOL_REVISION = 7`, checked on import in
  `_frisky/base.py` (fails loudly on a stale `.so`).
- Python `dask_array/_frisky/`: `base.py` (thin `Layer` delegating to `_rust`),
  `blockwise.py`, `creation.py` (build the Rust layer), `collect.py`
  (`collect_task_records`), `__init__.py`.
- Verified: **suite 2810 passed** (1 pre-existing masked-array tokenize flake:
  `test_weighted_reduction` / `test_slice_pushdown::test_masked_array`, fail
  identically on `main` — NOT ours). `bench/e2e_transparent.py` green on a real
  `LocalCluster`. Frisky `test_dask`/`test_basic`/`test_hijack` 101 passed.

## File map

- dask-array Rust: `crates/dask-array-python/src/{common,blockwise,creation,lib}.rs`.
- dask-array Python: `dask_array/_frisky/{base,blockwise,creation,collect,__init__}.py`;
  `Array.__frisky_task_records__` in `dask_array/_collection.py`; routing in
  `dask_array/_blockwise.py` (`_frisky_layer`, `_layer`) and
  `dask_array/creation/_ones_zeros.py`. Build: `pyproject.toml` (maturin backend).
- Frisky (branch `array-bulk-client` off `main`): `crates/frisky-python/src/client.rs`
  (`Client::submit_tasks`, `lower_dep_refs`, `dispatch`); `python/frisky/dask.py`
  (the `dask.compute`/`persist` patch); `python/frisky/_dask_config.py` (installs it).
- Feedback: dask-array `bench/{e2e_transparent,bench_records}.py`.

## Feedback systems (the inner loop)

- Build: `cd ~/workspace/dask-array && .venv/bin/maturin develop` (debug; rebuilds
  in ~1s after first compile). Use `--release` before perf-benching.
- Correctness (no Frisky): `.venv/bin/python -m pytest dask_array/tests/ -q -n auto`.
- Frisky e2e + perf (need Frisky's venv + PYTHONPATH; `dask_array` resolves from
  this checkout):
  `PYTHONPATH=/Users/mrocklin/workspace/dask-array /Users/mrocklin/workspace/frisky/.venv/bin/python bench/e2e_frisky.py`
  `... bench/bench_frisky_path.py` (old dask-translate vs new Frisky-native, per stage + wire bytes).

## Key measured findings (720k-task unfused ones+1)

- Old dask path (`__dask_graph__` → `translate_graph`): gen ~2.5s + serialize
  ~12.8s. The whole-Task-per-node pickle dominates.
- New Frisky-native (prototype): gen ~1.8s + serialize ~2.4s → **~3.6× total,
  serialize ~5×**. Wire bytes only ~1.1× smaller.
- Why the remaining gaps: generation still materializes a per-task key/dep
  **string** to hand to the *Python* prototype client; wire is dominated by those
  per-task key strings (`('add-<32hex>', i, j)`). Both vanish once the client is
  Rust (consumes the form directly) and the wire ships the layer name once +
  coords (Level-2).
- `to_dask_graph` in Rust regressed the suite ~9s→~14s with no benefit.

## DONE (this session): boring task-records path + transparent hijack

After a first round built a clever compact "bulk" client (protocol-2 pickle
template + per-task byte-splice in Rust), the owner steered to **boring &
general**: generation produces a plain per-task mirror of the dask Task; the
Frisky client serializes it as it would any graph; cleverness is deferred. The
clever code (`bulk.rs`, `frisky/bulk.py`, `to_frisky_batch`, the splice + de-risk
scripts) was **removed**. Work is on Frisky branch `array-bulk-client` off `main`
(main checkout — NOT `frisky-bulk-graph`); dask-array on `rust-layers`. All
uncommitted.

- **Generation (dask-array, Rust):** `to_task_records` (`common.rs`) → a flat
  Python list of `(key, func, args, kwargs, deps)` per task. `key`/`deps` are
  Rust strings; `func`/`kwargs` are the shared layer Python objects; `args` is a
  Python tuple with dask `TaskRef(dep_key_tuple)` at dependency positions and
  literals/values inline. `collect_task_records` (`_frisky/collect.py`) walks the
  lowered expr tree and concatenates. Exposed to Frisky via
  `Array.__frisky_task_records__()` (`_collection.py`), duck-typed.
- **Serialization (Frisky client, Rust):** `Client.submit_tasks` (`client.rs`) —
  per task: serialize the function once per distinct `(func, kwargs)` (local
  cache; a layer reuses one func object across its tasks), bind `kwargs` via
  `functools.partial`, replace `TaskRef`/`Future` refs in `args` with worker
  placeholders (`lower_dep_refs`, sibling of `extract_futures`), `dumps_value` the
  args, `[u32 func_len][func][args]`, submit via the existing `UpdateGraph`
  (`Client::dispatch`, shared with `submit_graph`). **Worker + scheduler
  untouched.**
- **Transparent:** Frisky patches `dask.compute`/`dask.persist` (+ `dask.base`
  originals) from `install_dask_defaults` (every Frisky `Client`; idempotent;
  inert without a Frisky scheduler → real dask collections fall through). When the
  scheduler is a Frisky client (duck-typed `submit_tasks`) and the single
  collection yields records, submit via `submit_tasks` + repack
  (`__dask_postcompute__`) / rebuild (`__dask_postpersist__`); else the saved
  original dask function. Multi-collection / unsupported layer / extra kwargs →
  fall back.
- **Correctness:** dask-array suite 2810 passed (1 pre-existing masked flake);
  frisky `test_dask`/`test_basic`/`test_hijack` 101 passed;
  `bench/e2e_transparent.py` — `dask.compute(x)`, `dask.persist(x)`, `a+b` take
  the records path and match numpy; fused `x.compute()` falls back, still
  matches. `PROTOCOL_REVISION` → 7.
- **Perf (`bench/bench_records.py`, inproc, submission-to-futures):** records
  path **~4.2–4.7× faster** than the materialized (`__dask_graph__` +
  `translate_graph`) path (e.g. 180k tasks: ~3260ms → ~690ms) — even though it's
  "boring." The win comes from Rust generation (lighter records vs Python dask
  Tasks), the func cache, and a lean run_spec (no `_execute_dask_task` wrapper).
  Note the records path also skips `dask.order` (insertion-order priorities) —
  part of the speed, but a scheduling-quality tradeoff to revisit. Further
  speedups (avoid per-task Python `args`) are explicit future cleverness.

Caveats (same as before):
- `x.compute()` optimizes (fuses) first → `FusedBlockwise` (no `_frisky_layer`) →
  falls back. The records path fires mainly for the `dask.compute(x)` free
  function (raw, unfused collection).
- The records path doesn't run dask's graph optimize (works from the raw lowered
  collection) → unfused, same result, more tasks.

## NEXT STEP candidates

- **`FusedBlockwise` Frisky layer** (highest leverage): unlocks the records path
  for real `x.compute()` graphs (which fuse) and lets it keep dask's fusion.
- More layer types: `from_array` (numpy eager-slice care), rechunk, reductions,
  slicing, transpose — each just emits `to_task_records` like the others.
- Performance cleverness, *once the general path is in place and measured*:
  avoid per-task Python `args` construction; reuse Frisky's `dask.order` for
  priorities (currently insertion order); cross-collection dedup (shared `_name`)
  so `dask.compute(a, b)` can use the records path instead of falling back.

## Adding a layer — the recipe (READ THIS BEFORE ADDING LAYERS)

The leverage that makes this scale: a new layer only implements **`expand()`**
(its output-block → task mapping into the neutral `Expanded` form) plus
**conservative routing**. Both converters — `to_dask_graph` (correctness,
validated by the suite) and `to_task_records` (the Frisky path) — are generic
and work for free. Mirror `blockwise.rs` (deps) and `creation.rs` (no deps) as
exemplars.

1. **Rust layer** `crates/dask-array-python/src/<layer>.rs`: a `#[pyclass]`
   holding the layer's *compact* params (name, func, kwargs, dep_names, + shape
   info — NOT per-task data); `#[new]` parses the normalized params from Python;
   `to_dask_graph` / `to_task_records` just delegate to `common::`; a private
   `expand(&self) -> Expanded` produces the neutral form: `tasks: Vec<(coord,
   Vec<ArgSlot>)>` where `ArgSlot` ∈ `Literal(idx)` | `Dep { name_idx, coord }` |
   `IntTuple(vals)`. Register with `mod <layer>;` + `m.add_class::<...>()` in
   `lib.rs`.
2. **Python wrapper** `dask_array/_frisky/<layer>.py`: a thin `Layer` subclass
   building `self._rust = _rust.<Layer>(...)`; export in `_frisky/__init__.py`.
3. **Routing** on the expr (e.g. `Transpose`, `Rechunk`, the reduction `Expr`):
   `_layer()` = `try: self._frisky_layer().to_dask_graph() except
   NotImplementedError: <legacy>`. `_frisky_layer()` validates + normalizes the
   operands into the layer's `#[new]` form and **raises `NotImplementedError` for
   any shape it doesn't fully handle** — the safety valve: when unsure, raise and
   fall back to the correct legacy dask path. Bump `PROTOCOL_REVISION` (lib.rs +
   `_frisky/base.py`) whenever the Rust↔Python method surface changes.
4. **Verify**: the dask-array suite (`pytest dask_array/tests/ -q -n auto`) is the
   correctness oracle for `to_dask_graph` with `_frisky_layer` active;
   `bench/e2e_transparent.py` confirms the records path round-trips on a real
   cluster. Done = suite green + a numpy-matching e2e case for the new layer.

**Slot-vocabulary note (the one real shared dependency):** if a layer needs an
arg shape `ArgSlot` can't express — an arbitrary per-block Python object, or a
variable-length dep list (tree reduction, rechunk-concat) — extend `ArgSlot` in
`common.rs` ONCE; both converters must learn it. Settle this centrally before
fanning out, so parallel agents don't each extend it differently.

## Layer backlog (difficulty / what each exercises)

- **transpose** — index permutation only (deps are a relabel of the output
  coord). Simplest non-elemwise; **recommended first** to confirm the recipe
  generalizes. No new slot kinds.
- **FusedBlockwise** — *highest leverage*: unblocks `x.compute()` (which fuses
  before dispatch) and lets the records path keep dask's fusion. The func is a
  fused sub-expression; args are the fused layer's external deps. Most of the
  work is normalizing the fused expr in `_frisky_layer`.
- **from_array** — eager numpy source; each block slices the backing array. Care:
  the per-task func/arg must do the slice; don't ship the whole array to every
  worker. May need a non-int per-task arg.
- **rechunk** (tasks method) — each output block concatenates/slices several
  input blocks → **variable deps per task** + per-task slice indices. First user
  of an extended slot vocabulary.
- **reductions** (tree reduce) — partial aggregation funcs + variable fan-in.
- **slicing** (`getitem`) — per-block index math; mostly Dep + IntTuple.

Recommended order to nail the process: **transpose**, then **FusedBlockwise**.
rechunk/reductions follow once variable-arity deps are settled in `ArgSlot`.

## Parallel-agent workflow (for the bulk of layers)

Per-layer files keep contention low, so layers fan out — with coordination:

- **Shared files:** `lib.rs` (mod + add_class) and `_frisky/__init__.py` (export)
  are append-only, low-conflict. `common.rs` `ArgSlot` is the real shared point —
  settle the slot vocabulary BEFORE fanning out. Expr-routing files usually
  differ per layer.
- **Editable-install trap (important):** the venv resolves `dask_array` +
  `dask_array._rust` to the *main checkout* via an editable install that beats
  `PYTHONPATH`, so an agent editing a bare worktree can't import/test its work.
  Either give each parallel agent its **own worktree with its own venv +
  `maturin develop`** (imports resolve locally), or have agents return diffs and
  the **lead applies + builds + tests sequentially** on the main checkout. Never
  run multiple agents editing the main checkout at once (they collide); never
  `git stash` in an agent (shared stash stack).
- **Fan-out:** (A, lead) lock the recipe + slot vocabulary + the layer set;
  (B, parallel) one agent per layer — reads this § + the exemplars, implements
  `expand()` + routing + an e2e case, runs the suite in its worktree, reports
  diff + results; (C, lead) integrate the append-only edits, run full suite +
  `e2e_transparent` + a fresh review.
- **Per-agent task template:** "Implement the `<X>` layer for Frisky task
  generation in dask-array. Follow `plans/frisky-rust-task-gen.md` § Adding a
  layer; mirror `blockwise.rs`/`creation.rs`. Your files:
  `crates/dask-array-python/src/<x>.rs`, `dask_array/_frisky/<x>.py`, routing in
  `<expr file>`; append-only edits to `lib.rs`/`_frisky/__init__.py`; don't touch
  other layers' files. Validate conservatively (NotImplementedError when unsure).
  Verify: suite green + a numpy-matching `e2e_transparent`-style case for `<X>`.
  Report the diff + test output."

## Open questions / known issues

- **Slot vocabulary for variable-arity deps** (rechunk/reductions): does a
  per-task `Vec<ArgSlot>` already suffice (just more `Dep` slots per task), or do
  we want a dedicated representation? Decide before those layers.
- **`from_array` eager-slice**: keep the backing array out of per-task args.
- **`dask.order` priorities**: the records path uses insertion order; wire
  Frisky's `dask.order` port if scheduling quality matters on real workloads.
- **Monkeypatch stacking**: the `dask.compute`/`persist` patch assumes nothing
  else patches `dask.base.compute` after Frisky (distributed integrates via
  config, not patching — low risk in practice).
- **Multi-collection** `dask.compute(a, b)`: currently falls back (no
  cross-collection `_name` dedup yet).
