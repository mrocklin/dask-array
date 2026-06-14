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

## Challenges

-  dask-array was pure python before, so we now have to include rust
-  dask graphs can be arbitrarily complex, so attempts to abstract around them (like map or reduce layers) tend to fail
-  We still need to send Python objects in the task, so we'll need to have pyo3 code deal with python objects
-  There are many expressions / layers to translate

## Benefits

-  We have a good python test suite
-  Frisky and Rust are pretty fast

## Rough plan

We build Rust-side Layer objects, like Blockwise, Creation, PartialReduce, Rechunk, Slice, etc., that more or less correspond to our layers.  These  layer objects will build a Rust implementation of the Dask graph of that layer.  That dask graph will contain some Rust Task object that holds Python objects for func/args/kwargs, but otherwise uses rust concepts where it can for efficiency (dependencies, etc.)

We also have a generic function that takes this layer and translates it, task by task, to the Python/Dask representation suitable for a dask scheduler.  We can use these then to drive tests.  The dask-array Expression class has a _layer method that can try to create the frisky layer if its around, then expand to rust tasks, then translate those tasks to Python/Dask to return to Dask.  We can then use the existing test suite to verify correctness.

That Rust Layer can also be used by the frisky client to bypass Python and go straight to something that we can send up to the scheduler as normal (no scheduler-side improvements are expected in this work).

## Process

We're going to do a few of these in one smart / Opus agent to make sure that this process is solid.  Then once we get it to a point where it's mechanical we'll do a parallel agent workflow to work through the rest of the dask array API.

## Feedback

We can get feedback from a few things:

-   Dask-array test suite, which helps to ensure correctness of our rust work with the `to_dask_graph` function.
-   Frisky roundtrip.  We can make sure that a simple workload with frisky computes properly.  We can use the Frisky in-process LocalCluster for this:  `frisky.LocalCluster(processes=False, dashboard_address="127.0.0.1:0")`.
-   Code simplicity compared to Dask implementation.  They should be similar.

# Detail and current state

## Architecture (as built)

A Frisky **layer** is a Rust object, one per expression kind (`BlockwiseLayer`,
`CreationLayer` in `crates/dask-array-python/src/`). It builds that expression's
subgraph in Rust and exposes two generic converters:

- `to_dask_graph` → `{key: dask Task}` (builds `Task`/`TaskRef`). The
  correctness/legacy path; the dask-array test suite is its oracle.
- `to_task_records` → a flat list of per-task `(key, func, args, kwargs, deps)`
  — a Rust-built mirror of the dask Task. `key`/`deps` are Rust strings;
  `func`/`kwargs` are the shared Python objects; `args` is a Python tuple whose
  dependency positions hold a dask `TaskRef(dep_key)`.

Hand-off: `Array.__frisky_task_records__()` (duck-typed; Frisky never imports
dask_array) returns the records; Frisky's `Client.submit_tasks` (client.rs)
serializes them exactly as `submit`/`map` do — function serialized once per
distinct `(func, kwargs)` (cached), `kwargs` bound via `functools.partial`,
`TaskRef`/`Future` refs replaced with worker placeholders (`lower_dep_refs`),
then the existing run_spec + `UpdateGraph`. **Scheduler and worker are
untouched.** `dask.compute`/`dask.persist` are patched (from
`install_dask_defaults`) so a bare `dask.compute(x)` takes this path
transparently and anything unsupported falls back to stock dask.

Routing per expression: `_layer()` = `try self._frisky_layer().to_dask_graph()
except NotImplementedError: <legacy dask>`. `_frisky_layer()` validates +
normalizes the operands and **raises `NotImplementedError` for any shape it
doesn't fully handle** — the safety valve: when unsure, fall back to the correct
legacy path.

The neutral form (`common.rs`) now covers all three task structures. A
`NeutralTask` carries its own key (`names[name_idx]` + coord) and a `Compute`
(`Call{func_idx}` into the layer's `funcs`, or `Alias`). `ArgSlot` is
`Literal | Dep{name,coord} | IntTuple | List(Vec<ArgSlot>) | Slices`. So one
layer can emit multiple funcs, free-form intermediate keys, nested/variable-fan
deps, aliases and per-block slices. Single-func/flat layers (blockwise,
creation) are just the common case: one entry in `names`/`funcs`, every task a
`Call{0}`.

## Status — 19 layers; now in the completeness/correctness phase

- **Coverage frontier (`bench/coverage_probe.py`):** 32/35 common composite
  operations take the records path **end-to-end** (fully Rust-generated, matching
  numpy) — all elementwise/ufunc/where/clip/astype, transpose, every reduction
  (sum/mean/std/var/min/prod), matmul/tensordot/dot, slicing compositions, rechunk,
  concatenate/stack/coarsen/broadcast, diag, `eye @ x`, **reshape/ravel**. The 3
  fall-backs (still correct via legacy dask) are `cumsum` (cumulative) and
  `argmin`/`argmax` (arg_reduction) — the next data-driven priorities.

- **Done (committed on `rust-layers`):** blockwise (+ grid-preserving
  `adjust_chunks`), creation, **from_array** (Python data source — see note in the
  landscape), **PartialReduce**, **TasksRechunk** (multi-step), broadcast_to,
  expand_dims, squeeze, concatenate, stack, **basic slicing/getitem**
  (`SliceSlicesIntegers` — slices + integer-drop + negative steps; the intricate
  `_slice_1d` index math stays in Python, Rust does the O(n_tasks) product),
  **blocks** (`x.blocks[...]` — pure block-index alias), **coarsen** (per-block
  reduction, the simplest blockwise shape), the **indexed-creation family**:
  **arange** (introduced the per-task scalar `ArgSlot::Scalar(Num)`), **linspace**
  (same shape), **eye** (2-D; per-block `np.eye`/`np.zeros` choice — the
  multi-func + IntTuple + Scalar combo), and **diag** (`Diag1D` 1-D→2-D matrix +
  `Diag2DSimple` 2-D→1-D extract; `Diag1D` introduced per-task kwargs —
  `Compute::CallKw` — for its off-diagonal `np.zeros_like(meta, shape=…)`).
  Plus **reshape** (`ReshapeLowered` + `ReshapeBlockwise` — per-block
  `M.reshape(in_block, out_shape)`, 1:1 C-order block mapping; one Rust layer
  serves both). ~19 of ~79 layer classes. PROTOCOL_REVISION 17.
- dask-array suite green (**2808 pass**; the 3 masked-array failures are
  pre-existing — numpy-2.2.6 masked `.view('i1')` in dask's tokenize, fail at
  `main` too, pure numpy traceback with no layer code, fail standalone in a fresh
  process. They surface intermittently under xdist tokenize-cache priming, so
  ALWAYS deselect the full set (the complete list, confirmed by a single-process
  `-k "mask or weighted_reduction"` run):
  `dask_array/tests/test_reductions.py::test_weighted_reduction`,
  `dask_array/tests/test_slice_pushdown.py::test_masked_array`,
  `dask_array/tests/test_slicing.py::test_slice_masked_arrays`.)
- Each layer validated **3 ways**: suite (`to_dask_graph` oracle) + `diff_layers.py`
  (distinct-arange records-vs-dask per key — catches mis-assembly) +
  `roundtrip_layers.py` (real in-process Frisky, asserts the records path was
  taken). **Frisky needed zero changes across all 10** — the client stays
  array-agnostic, which is the property to preserve.
- **Verification tools (`bench/`):** `diff_layers.py`, `roundtrip_layers.py`,
  `coverage_probe.py` (which common composite ops take the records path
  end-to-end vs fall back — the data-driven priority map),
  `bench_records.py` (blockwise submission speedup), `e2e_transparent.py`.
- Frisky `array-bulk-client` `9bf5760` (unchanged).

## Phase posture: completeness/correctness now, performance later

The current phase is **completeness + correctness** — cover the dask-array layer
API cleanly and correctly, with *decent* (not maximal) performance. The 5–10×
hand-off target from the preamble is **deferred to a later optimization phase**;
it needs work the current "generic, no-Frisky-changes" approach intentionally
avoids. Don't chase perf now — finish the Rust transformation on the current path.

### Measured performance (recorded so it isn't lost)
Client-side submission, records path vs the materialized
`__dask_graph__`+`translate_graph` path: **~2–4.5×, not 5–10×.** By layer:
blockwise / reduction ~3.8–4.5×, expand_dims ~3.3×, rechunk ~2.0×, concatenate
~2.1× (and **net-negative full-pipeline** — alias routing nodes pay full per-task
cost). That's the honest current number; it's fine for this phase.

### Why 5–10× isn't reached, and the real lever (profiled this round)
`submit_tasks` phase breakdown (ones+1, 12,800 tasks, release):
`dumps_value` (per-task **arg pickle**) **~55%**, `lower_dep_refs` (per-task dep
walk) ~20%, run_spec/TaskSpec ~12%, dispatch ~11%, func (cached) ~2%. And
`submit_tasks` is only ~40% of client cost — `collect_task_records` (Rust gen,
which *builds* the per-task `TaskRef`/args Python objects) is ~60%. (A much larger
wall-clock "submit" seen elsewhere is scheduler round-trip/ingest under
contention — out of scope per the preamble; not client serialization.)

The real lever is **generate serialized task state directly**: dask-array's Rust
writes the run_spec bytes itself instead of building per-task `TaskRef`/args Python
objects that Frisky then walks (`lower_dep_refs`) and pickles (`dumps_value`). That
collapses *both* dominant costs at once — for structurally-simple args (ints,
coords, dep-refs: most layers) Rust serializes directly; only opaque shared
literals need a one-time pickle. Projection: blockwise ~3.8× → ~5–6×. **Crucially
this keeps Frisky array-agnostic:** Frisky defines a *generic* structured-task
format (shared blobs + compact per-task data) and deserializes it with one generic
codepath — it does NOT hold per-layer "templates" (that would re-introduce
array-awareness, the failure mode to avoid). The neutral form (`Expanded`: shared
funcs/kwargs/literals + per-task coord/slots) already provides the shared/per-task
split this needs.

### Deferred optimization backlog (future phase)
- **Serialized-state / generic structured-task format** — the main 5–10× lever
  (above). Needs generic Frisky-side protocol + worker support, no array-awareness.
- **Native Frisky `Alias` term** ("key X resolves to key Y", no task run) — removes
  per-task pickle for routing nodes; fixes the concatenate/stack full-pipeline
  regression. The records path currently fakes alias with `toolz.identity`.
- **`FusedBlockwise` layer** — `x.compute()` fuses before submitting and the fused
  layer has no `_frisky_layer`, so idiomatic compute falls back to dask. Cover it
  (still blockwise-shaped) so the records path fires for real user code.
- **`optimize_graph` / low-level fusion** in the records path (currently skipped).
- "Lower deps in Rust" (build dep placeholders in Rust, drop `lower_dep_refs`) is a
  ~20%-of-`submit` subset of the serialized-state work — fold it into that, not as
  a standalone step.

## Adding a layer — the recipe

Mirror `blockwise.rs`/`creation.rs` and `_frisky/{blockwise,creation}.py`.

1. **Rust layer** `crates/dask-array-python/src/<layer>.rs`: a `#[pyclass]`
   holding the layer's compact params; `#[new]` parses normalized params; a
   private `expand()` builds the neutral form; `to_dask_graph`/`to_task_records`
   delegate to `common::`. Register with `mod` + `add_class` in `lib.rs`.
2. **Python wrapper** `dask_array/_frisky/<layer>.py`: thin `Layer` subclass
   building `self._rust = _rust.<Layer>(...)`; export in `_frisky/__init__.py`.
3. **Routing** on the expr (`_layer` try/except + `_frisky_layer`
   validate/normalize, NotImplementedError → fallback). Bump `PROTOCOL_REVISION`
   (lib.rs + `_frisky/base.py`) if the Rust↔Python surface changes.
4. **Verify both paths.** The suite only exercises `to_dask_graph`; a layer can
   pass it and still have a broken `to_task_records` (the two converters differ —
   e.g. dask wraps nested deps in `_task_spec.List`, the records path uses a plain
   list). So also run the records path with **distinct** data: `bench/diff_layers.py`
   (records-vs-dask per key) is **required** for any layer that moves/slices/
   concatenates blocks — `da.ones` can't catch a mis-assembly. Add a
   `bench/roundtrip_layers.py` case for a real-Frisky check too.

**Two guiding principles (held up well in round 1):**
- **Python plans, Rust expands.** Do a layer's one-time structural/planning math
  in Python (reuse dask's tested code) and pass Rust only the per-block params;
  port just the O(n_tasks) expansion. (Rechunk keeps `plan_rechunk` in Python.)
- **`_frisky_layer` is a conservative safety valve.** Raise `NotImplementedError`
  for anything not fully handled; only relax a guard when an existing invariant
  still catches the bad case (e.g. blockwise now allows `adjust_chunks` because
  the alignment check still rejects count-changing forms). `common.rs`
  `ArgSlot`/`Compute` is the shared vocabulary — extending it (a construct not yet
  modeled, e.g. a dict arg) is a lead escalation, not an independent edit.

## The layer landscape (categorized priorities)

~79 layer-defining classes across ~56 files, grouped by task-structure shape:

- **Blockwise-shaped** — per output block → `func(input blocks) + literals`.
  Fits the current model (some need an `Alias` task and/or per-block `IntTuple`
  args). Elementwise/blockwise ✅, creation ✅, `from_array` ✅, simple transforms
  (squeeze ✅, expand_dims ✅, broadcast_to ✅, reshape ✅), aliasing (concatenate ✅,
  stack ✅, blocks ✅, copy), indexed creation (arange ✅, linspace ✅, eye ✅ — fit
  the per-task scalar `ArgSlot::Scalar(Num)`; diag ✅ — `Diag1D`/`Diag2DSimple`,
  k=0; `Diag1D` introduced per-task kwargs `Compute::CallKw` for its off-diagonal
  `np.zeros_like(meta, shape=…)`; `diagonal` and k≠0 diag still deferred), basic
  slicing/getitem ✅, coarsen ✅, gufunc, random.
- **Variable fan-in** — one output ← a nested, variable-length list of input
  blocks. Needs a **nested/list arg** in the neutral form. PartialReduce
  (tree-aggregate, lol_tuples), concatenate-finalize, overlap (neighbors),
  cumulative scans. (Plain `concatenate`/`stack` turned out to be aliasing/getitem,
  not variable fan-in — handled above.)
- **Multi-stage** — a single layer emits intermediate-keyed tasks with ≥2
  funcs. Needs **per-task func + free-form intermediate keys**. Rechunk
  (slice→concat), all linalg (single-chunk in-core then per-block multiply),
  boolean-index, setitem.
- **Sequence by what unlocks testing, not just frequency.** The records path is
  all-or-nothing: `collect_task_records` falls back if *any* node in the lowered
  tree lacks a `_frisky_layer`. So a layer can't be roundtrip-tested on a real
  cluster until its whole input chain is covered — and **`from_array` gated every
  non-creation workload** (done ✅ — distinct-data workloads now roundtrip on a
  real cluster, not just the `diff_layers.py` local resolver). concatenate ✅ +
  stack ✅ done (parallel batch 2). **basic slicing/getitem ✅ done** (lead-first:
  generalized `ArgSlot::Slices(Vec<(i64,i64)>)` → `ArgSlot::Index(Vec<IndexElem>)`
  carrying `slice(start,stop,step)` with `None` bounds + integer-drop; rechunk
  migrated to the same variant). Both testing-unlock roots are now covered.
  **blocks ✅ + coarsen ✅ done (parallel batch 3** — agents wrote disjoint files +
  routing only; lead did lib.rs/__init__/protocol + diff/roundtrip cases + the
  single integrated build. Confirmed: two agents cannot build the shared checkout
  concurrently — a build compiles everyone's in-progress `.rs` — so agents write,
  lead builds). **arange ✅ done (lead-first)** — added the per-task scalar
  `ArgSlot::Scalar(Num{Int|Float})` to `common.rs` and settled it on arange
  (blockstart/stop computed in Python, mirroring the legacy `_layer`; step/dtype
  shared literals). **`linspace` ✅ + `eye` ✅ done (parallel batch 4)** — both fit
  the Scalar slot; eye exercised the 2-func + IntTuple + Scalar combo (per-block
  `np.eye`/`np.zeros` choice, all int arithmetic in Rust). **`diag` ✅ done
  (lead-first)** — `Diag1D` (1-D→2-D) + `Diag2DSimple` (2-D→1-D), k=0 only.
  `Diag1D` resolved the per-task-kwarg question: owner chose general per-task
  kwargs, implemented as an additive `Compute::CallKw { func_idx, kwargs:
  Vec<(String, ArgSlot)> }` variant (merged over shared kwargs; each value a full
  `ArgSlot`) — chosen over a field-on-every-`NeutralTask` to avoid churning the 18
  existing construction sites; per-task kwargs are rare (only diag so far).
  **`reshape` ✅ done (autopilot, agent-built)** — `coverage_probe.py` flagged it
  as the top fall-back (reshape/ravel hit 3× in common workloads); one Rust
  `ReshapeLayer` (1:1 C-order block map + per-block `IntTuple` shape) serves both
  `ReshapeLowered` and `ReshapeBlockwise`, no `common.rs` change. Next data-driven
  candidates (from the probe): `cumulative` (cumsum — intricate: chunk + sequential
  correction-chain + an inline `getitem` + an `apply`+kwargs `full_like`; also a
  `CumReductionBlelloch` variant — needs care, possibly a lead-first port) and
  `arg_reduction` (argmin/argmax — entangled with the reduction combine; the ravel
  case needs a nested-tuple offset). Linalg (qr/svd/lu — the big multi-stage lift),
  map_blocks, vindex, bool-index, setitem, random, `diagonal`, k≠0 diag, overlap
  (delegates to dask's `ArrayOverlapLayer`) are lower-frequency or large — defer.
- **Data-source layers are a Python seam, not Rust.** `from_array` (and other I/O
  sources) have no per-task computation to accelerate — each block is a numpy
  slice — so they build records directly in Python (`_frisky/from_array.py`: a
  plain object, not a Rust layer; data nodes wrapped as `toolz.identity` tasks).
  Only *computed* layers go through Rust + `common.rs`.

## First round: PartialReduce + Rechunk (DONE — model de-risked)

Goal was to extend the neutral form to the two structures the rest of the API
needs and lock the vocabulary before parallelizing. Both landed:

- **PartialReduce** (`reduction.rs`, `_frisky/reduction.py`; routing in
  `reductions/_reduction.py`). Reproduces dask's `lol_tuples` nesting as one
  `ArgSlot::List` per task (reduced axes nest, kept axes fix a coord). The
  reduction's chunk step is a `Blockwise` with `adjust_chunks` — `_blockwise.py`
  now accepts grid-preserving `adjust_chunks` (block *counts* unchanged; the
  alignment check still rejects count-changing forms), so a whole `da.x.sum()`
  takes the records path.
- **Rechunk** (`rechunk.rs`, `_frisky/rechunk.py`; routing in `_rechunk.py`).
  The intricate planning (`plan_rechunk`) stays in tested Python and runs once;
  Rust does the O(n_blocks) work — the 1-D chunk intersection
  (`intersect_1d`/`old_to_new`) and the per-block split (`getitem`) / merge
  (`concatenate3` nested list, or `Alias`) expansion, across multiple steps.
  Exercises the full vocabulary: two `funcs`, `Compute::Alias`, free-form split
  keys, `ArgSlot::List`, `ArgSlot::Index`.

Validated: dask-array suite (2809) + distinct-data differential (`diff_layers.py`,
records vs dask per key, incl. multi-step/3-D/transpose-like) + real-Frisky
roundtrip (`roundtrip_layers.py`). Independent review: 0 critical/medium across
~400 differential configs. The neutral form + `ArgSlot` vocabulary now cover
per-task func, nested/variable deps, intermediate keys, aliases and slices — the
union most remaining layers need.

## Then: parallel-agent workflow

Once the recipe is mechanical and the vocabulary is settled:

- **Sequence by testing-unlock first:** land `from_array` + basic slicing/getitem
  before fanning out (serial or a first wave). They're the roots that let every
  downstream layer roundtrip on a real cluster; until then downstream layers can
  only be checked via the `diff_layers.py` local resolver.
- One agent per layer (or small group); per-layer files keep contention low.
  Shared files: `lib.rs` + `_frisky/__init__.py` are append-only; `common.rs`
  `ArgSlot`/`Compute` is settled in round 1 (an agent needing to extend it
  escalates to the lead).
- **Editable-install trap (confirmed in batch 1):** the venv resolves
  `dask_array` / `dask_array._rust` to the *main checkout*. Spawning agents with
  `isolation: worktree` did NOT isolate — all three agents' edits landed in the
  main checkout and built/tested there. It worked only because the edits were
  *additive* (distinct new files + distinct routing files) and the shared-file
  appends (`lib.rs`, `__init__.py`, `diff_layers.py`) didn't lose updates this
  time. Do not rely on that. Safer patterns: (a) lead owns ALL shared-file edits
  (`lib.rs` mod/add_class, `__init__.py`, `base.py` protocol) — agents only write
  their own new files + routing and *report* the one-liners to add; or (b) give
  each agent a real separate clone+venv. Keep the batch small. Never `git stash`
  in an agent.
- **Protocol bump is the lead's, once per batch.** Agents leave
  `PROTOCOL_REVISION` alone (concurrent bumps collide); the lead bumps it once
  after integrating the batch.
- **Per-agent template:** "Implement `<X>` per `plans/frisky-rust-task-gen.md`
  § Adding a layer; mirror blockwise/creation; your files
  `crates/.../<x>.rs`, `dask_array/_frisky/<x>.py`, routing in `<expr file>`;
  append-only edits to `lib.rs`/`__init__.py`; validate conservatively
  (NotImplementedError → fallback); verify the suite **and** the records path
  with distinct data (`diff_layers.py`) + a Frisky roundtrip; report the diff +
  test output. Deselect the two pre-existing masked-array flakes."

## Verification (the inner loop)

- **Correctness oracle:** `cd ~/workspace/dask-array && .venv/bin/python -m
  pytest dask_array/tests/ -q -n auto` with `_frisky_layer` active. Deselect the
  3 pre-existing numpy-masked-tokenize flakes (full file::test paths — the bare
  `test_slice_pushdown::…` form silently matches nothing):
  `--deselect dask_array/tests/test_reductions.py::test_weighted_reduction
  --deselect dask_array/tests/test_slice_pushdown.py::test_masked_array
  --deselect dask_array/tests/test_slicing.py::test_slice_masked_arrays`.
  Expect **2808 passed**. (They fail intermittently under xdist; the complete set
  is confirmed by `pytest -k "mask or weighted_reduction"` in one process.)
  Build: `.venv/bin/maturin develop` (debug, ~1s rebuild; `--release` for perf).
- **Records path, distinct data:** `bench/diff_layers.py` — cluster-free, runs
  each layer's `to_task_records` through a worker-style resolver and compares to
  the dask path *per key* with `arange` data. This is the check that catches
  mis-sliced / mis-ordered assembly (the suite + `da.ones` roundtrips can't), and
  it's layer-agnostic — extend `cases()` for each new layer.
- **Frisky roundtrip:** in-process cluster
  `frisky.LocalCluster(processes=False, dashboard_address="127.0.0.1:0")` +
  `Client`; `dask.compute(x)` (the free function takes the records path) matches
  numpy (`bench/roundtrip_layers.py`). Benches: `bench/e2e_transparent.py`,
  `bench/bench_records.py` (run with Frisky's venv,
  `PYTHONPATH=~/workspace/dask-array`, `MATURIN_IMPORT_HOOK_ENABLED=0`).
- **Simplicity:** the Rust layer should be comparable in size/clarity to the
  dask Python `_layer` it replaces.

## Open questions / model decisions

Settled in round 1:
- **Nested / variable-length dep args** → `ArgSlot::List(Vec<ArgSlot>)` (recursive
  `ArgSlot` nesting). Both converters build it; the dask path wraps it in
  `dask._task_spec.List` (so embedded `TaskRef`s register as deps), the records
  path a plain Python list (Frisky's worker `resolve_futures` recurses lists).
- **Multi-func + intermediate keys** → `Expanded` carries `names` (keys) +
  `funcs`; each `NeutralTask` picks a `Compute` (`Call{func_idx}` or `Alias`).
  Single-func/flat layers are just one `names`/`funcs` entry — no fast-path fork.
- **Alias** → dask path emits `dask._task_spec.Alias`; records path a
  `toolz.identity` task (Frisky has no alias node). Deps tracked either way.

Still open:
- `from_array`: keep the backing array out of per-task args (per-block getter).
- `dask.order` priorities: the records path uses insertion order; revisit if
  scheduling quality matters on real workloads.
