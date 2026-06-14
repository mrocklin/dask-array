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

The neutral form today (`common.rs` `Expanded` / `ArgSlot`) assumes **one shared
func per layer** and **flat args** (`Literal | Dep{name,coord} | IntTuple`).
That fits blockwise + creation. The next two layers generalize it (below).

## Status

- **Done:** blockwise (same-grid / broadcast elementwise) + creation
  (ones/zeros/empty/full). dask-array suite green (2810 pass; the 2 masked-array
  flakes also fail on `main`); Frisky roundtrip matches numpy; ~4.2–4.7× faster
  client-side submission than the materialized (`__dask_graph__` +
  `translate_graph`) path.
- **Committed:** dask-array `rust-layers` `47dc790`; Frisky `array-bulk-client`
  `9bf5760`.

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
4. **Verify**: suite green with `_frisky_layer` active + a Frisky roundtrip case
   matching numpy.

## The layer landscape (categorized priorities)

~79 layer-defining classes across ~56 files, grouped by task-structure shape:

- **Blockwise-shaped** — per output block → `func(input blocks) + literals`.
  Fits the current model (some need an `Alias` task and/or per-block `IntTuple`
  args). Elementwise/blockwise ✅, creation ✅, simple transforms (squeeze,
  reshape, expand_dims, broadcast_to), aliasing (blocks, concatenate, copy),
  indexed creation (arange, linspace, eye, diag), `from_array` (per-block slice),
  basic slicing/getitem, coarsen, gufunc, random.
- **Variable fan-in** — one output ← a nested, variable-length list of input
  blocks. Needs a **nested/list arg** in the neutral form. PartialReduce
  (tree-aggregate, lol_tuples), stack, concatenate-finalize, overlap (neighbors),
  cumulative scans.
- **Multi-stage** — a single layer emits intermediate-keyed tasks with ≥2
  funcs. Needs **per-task func + free-form intermediate keys**. Rechunk
  (slice→concat), all linalg (single-chunk in-core then per-block multiply),
  boolean-index, setitem.
- **High-value handful** a typical workload hits (do first after the model
  generalizes): reductions, slicing, rechunk, from_array, concatenate/stack.
  Linalg, map_blocks, vindex are lower-frequency — defer.

## First round: PartialReduce + Rechunk (de-risk the model)

One Opus agent, serial. Goal: extend the neutral form to cover the two
structures the rest of the API needs, and lock the shared vocabulary **before**
parallelizing.

- **PartialReduce** (`reductions/_reduction.py`; `Reduction` lowers to
  `Blockwise` (chunk, already handled) + `PartialReduce` (aggregate)). Forces
  **nested / variable-length dependency args** (the aggregate's lol_tuples).
  Likely add `ArgSlot::List(Vec<ArgSlot>)` so a task's args can be arbitrary
  nested structures of `Dep`/`Literal`; both converters build the nested Python
  list (`lower_dep_refs` already recurses). Func = the aggregate (kwargs:
  axis/keepdims).
- **Rechunk** (`TasksRechunk`, `_rechunk.py`). Forces **multiple funcs +
  intermediate keys within one layer** (split = `getitem(old_block, slices)`;
  merge = `concatenate3(nested list)`). The current `Expanded` assumes one shared
  func and `(name, coord)` keys; generalize so a layer emits tasks that each
  carry their own func and an arbitrary key (intermediates aren't the layer's
  outputs). This converges the neutral form toward the per-task
  `(key, func, args, kwargs, deps)` hand-off described above, with single-func
  `Expanded` kept as a fast path for blockwise/creation.

Both are validated by the suite (`to_dask_graph`) + a Frisky roundtrip. When
done, the neutral form + `ArgSlot` vocabulary cover per-task func, nested /
variable deps, and intermediate keys — the union most remaining layers need.

## Then: parallel-agent workflow

Once the recipe is mechanical and the vocabulary is settled:

- One agent per layer (or small group); per-layer files keep contention low.
  Shared files: `lib.rs` + `_frisky/__init__.py` are append-only; `common.rs`
  `ArgSlot` is settled in round 1 (an agent needing to extend it escalates to the
  lead).
- **Editable-install trap:** the venv resolves `dask_array` / `dask_array._rust`
  to the *main checkout*, so an agent in a bare worktree can't import/test its
  changes. Give each parallel agent its own worktree + venv + `maturin develop`,
  or have agents return diffs the lead builds/tests serially. Never edit the main
  checkout concurrently; never `git stash` in an agent.
- **Per-agent template:** "Implement `<X>` per `plans/frisky-rust-task-gen.md`
  § Adding a layer; mirror blockwise/creation; your files
  `crates/.../<x>.rs`, `dask_array/_frisky/<x>.py`, routing in `<expr file>`;
  append-only edits to `lib.rs`/`__init__.py`; validate conservatively; verify
  suite + a Frisky roundtrip matching numpy; report the diff + test output."

## Verification (the inner loop)

- **Correctness oracle:** `cd ~/workspace/dask-array && .venv/bin/python -m
  pytest dask_array/tests/ -q -n auto` with `_frisky_layer` active (pre-existing
  flakes: `test_weighted_reduction`, `test_slice_pushdown::test_masked_array`).
  Build: `.venv/bin/maturin develop` (debug, ~1s rebuild; `--release` for perf).
- **Frisky roundtrip:** in-process cluster
  `frisky.LocalCluster(processes=False, dashboard_address="127.0.0.1:0")` +
  `Client`; `dask.compute(x)` (the free function takes the records path) matches
  numpy. Benches: `bench/e2e_transparent.py`, `bench/bench_records.py` (run with
  Frisky's venv, `PYTHONPATH=~/workspace/dask-array`,
  `MATURIN_IMPORT_HOOK_ENABLED=0`).
- **Simplicity:** the Rust layer should be comparable in size/clarity to the
  dask Python `_layer` it replaces.

## Open questions / model decisions (settle in round 1)

- Nested / variable-length dep args: `ArgSlot::List(Vec<ArgSlot>)` vs a more
  general "args are an arbitrary nested structure with embedded deps".
- Multi-func + intermediate keys: generalize `Expanded` toward per-task func +
  free-form keys (the records model), keeping single-func as a fast path?
- `from_array`: keep the backing array out of per-task args (per-block getter).
- `dask.order` priorities: the records path uses insertion order; revisit if
  scheduling quality matters on real workloads.
