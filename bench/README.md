# bench/

Scripts that answer "does the Frisky path work, and is it fast?" — parity
harnesses, engagement probes, and the benchmarks behind tuning decisions.
None of this runs in CI; the living harnesses are the manual correctness /
engagement net for the records + expression work (the pytest suite covers
graph generation — layers, records encoding, inventory — but not live
submission/engagement; these harnesses add that).

Two invocation patterns, both from the repo root:

```bash
# cluster-free (this repo's venv; first run syncs it, ~1 min)
uv run --extra test python bench/<script>.py

# frisky-venv (live Frisky cluster or frisky imports; PYTHONPATH pins THIS checkout)
PYTHONPATH=$PWD MATURIN_IMPORT_HOOK_ENABLED=0 \
  /Users/mrocklin/workspace/frisky/.venv/bin/python bench/<script>.py
```

For truthful tier/coverage numbers build the native extension first
(`uv run --extra test maturin develop`, seconds when warm) — the probes warn
loudly when it is missing or stale, since everything then misreads as
`adapter`.

## Living parity harnesses (live Frisky cluster, frisky-venv)

All seven spin an in-process `LocalCluster` and spy BOTH Frisky submission
paths via `_spy.py` — scheduler-side expression submission (the preferred
path, ~all computes today) and client-side task records — so per-case tags
(`expr` / `rec ` / `dask`) and engagement counts are truthful. Add cases as
you port layers.

| script | one line |
|---|---|
| `diff_records.py` | broad differential: transparent Frisky path vs forced stock-dask synchronous, ~70 diverse ops |
| `diff_adversarial.py` | same differential, loaded with adapter-hostile cases (float coords, odd dtypes, empty blocks, deep tail compositions) |
| `diff_review.py` | strictest: exact dtype+value equality, and a silent fallback is reported per-op, not counted as a pass |
| `adapter_stress.py` | GraphRecordsLayer adapter compositions (tail op feeding covered op and vice versa) vs numpy |
| `coverage_probe.py` | which common ops Frisky handles at all vs falls back to stock dask, vs numpy |
| `roundtrip_layers.py` | native-layer roundtrips (reductions, rechunk, slicing, creation, cumulative...) vs numpy, engagement asserted per case |
| `e2e_transparent.py` | smoke: plain `dask.compute`/`dask.persist`/`x.compute()` engage Frisky and match numpy (fastest, 4 cases) |

## Cluster-free parity + probes (<10 s)

| script | kind | one line |
|---|---|---|
| `diff_layers.py` | parity | per-key differential of the actual task records (`collect_task_records`) vs the dask path, distinct data, worker-style local resolver — the ~0.5 s correctness net for a new binary layer |
| `tier_probe.py` | probe | task-weighted build-tier split (binary / native_tuples / adapter / fallback) over a realistic-block-count corpus; the classifier is production `dask_array/_frisky/inventory.py` |
| `tail_probe.py` | probe | which expr classes still lack a `_frisky_layer` across specialized ops — the coverage burn-down list (node-weighted, not task-weighted) |

## Decision-record benchmarks

Benchmarks kept because their numbers justified a default; docstrings carry
the recorded results. Re-run when revisiting the decision.

| script | one line | runs on |
|---|---|---|
| `bench_unify_policy.py` | chunk-unification policy (`coarse`/`refine`/`auto`) across merge-inflation and shatter cases; motivates `auto` | cluster-free (subprocess per case; `--no-compute` for static only) |
| `bench_rechunk_insertion.py` | status-quo optimizer plans vs hand-inserted "oracle" rechunks on roll/IO alignment cases | cluster-free threads by default; `--frisky` needs frisky-venv |
| `rechunk_threshold.py` | rechunk planner copies-vs-tasks crossover; motivates `threshold=32` | frisky-venv (live cluster) |

## Frisky-venv benchmarks / profiling (client-side cost)

| script | one line |
|---|---|
| `bench_paths.py` | Rust task generation vs legacy Python `_layer()` loops, per stage, no cluster |
| `bench_records.py` | submission-to-futures cost: records path vs materialized `translate_graph` path, live cluster |
| `bench_submission.py` | legacy-path baseline (materialize + `translate_graph`): phase times + wire-size breakdown, no cluster |
| `profile_pipeline.py` | phase profile of the records submission pipeline (build/lower/optimize/records/submit) on a worker-less cluster |

## Workload generators

| script | one line |
|---|---|
| `synthetic_quantity_expression.py` | standalone stand-in for the research quantity DAG (`synthetic_quantity_array(complexity)`); builds the expression, no compute |

## Support

| file | one line |
|---|---|
| `_spy.py` | shared engagement spy: wraps frisky's four compute/persist helpers (expression + records), arity-tolerant |
