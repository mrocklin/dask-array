"""Client->scheduler submission benchmark for dask-array graphs.

This is the feedback loop for the Rust-layer task-generation work. It measures
the cost of turning a dask-array collection into Frisky tasks and the size of
the resulting wire payload, broken down by component. As the new
``__frisky_tasks__()`` route comes online it will compare old vs new here.

Run it with Frisky's venv (it imports ``frisky``); ``dask_array`` resolves from
this checkout via the editable install:

    FRISKY_PY=/Users/mrocklin/workspace/frisky/.venv/bin/python
    $FRISKY_PY bench/bench_submission.py                  # ~10k tasks, fast loop
    $FRISKY_PY bench/bench_submission.py --blocks 1000    # ~1M tasks, headline
    $FRISKY_PY bench/bench_submission.py --graph from_array

Phase times come from the spans ``translate_graph`` records, so they match what
a dashboard would show for the same graph.
"""

import argparse
import time
from collections import defaultdict

import numpy as np

import frisky
from frisky.dask import translate_graph


def make_graph(name, blocks):
    """Return a dask_array collection with roughly ``blocks**2`` output tasks.

    Shapes chosen to exercise distinct wire profiles:
      add        - elemwise binary; one add-task per block with real per-block
                   deps and a function shared across all blocks.
      ones1      - fused source; ``ones`` and ``+1`` fuse into one task per
                   block with NO deps and byte-identical run_specs (the extreme
                   function-duplication case).
      from_array - getter tasks over a numpy array plus an elemwise; real
                   deps, keys, and per-block slice args.
    """
    import dask_array as da

    n = blocks
    if name == "add":
        a = da.ones((n, n), chunks=(1, 1))
        b = da.ones((n, n), chunks=(1, 1))
        return a + b
    if name == "ones1":
        return da.ones((n, n), chunks=(1, 1)) + 1
    if name == "from_array":
        x = np.ones((2 * n, 2 * n))
        return da.from_array(x, chunks=(2, 2)) + 1
    raise SystemExit(f"unknown graph {name!r}")


def wire_breakdown(tasks):
    """Sum wire bytes per component across ``(key, run_spec, deps, prio)`` tuples.

    run_spec layout is ``[4-byte func_len][func][args]``; deps are the separate
    edge-key strings the scheduler ships alongside.
    """
    func_b = args_b = key_b = dep_b = 0
    for key_str, run_spec, deps, _prio in tasks:
        flen = int.from_bytes(run_spec[:4], "little")
        func_b += 4 + flen
        args_b += len(run_spec) - 4 - flen
        key_b += len(key_str)
        dep_b += sum(len(d) for d in deps)
    return dict(func=func_b, args=args_b, keys=key_b, deps=dep_b)


def measure_old(collection):
    """Legacy path: Python ``_layer()`` materialization + ``translate_graph``."""
    t0 = time.perf_counter()
    expr = collection.optimize()
    t1 = time.perf_counter()
    dsk = dict(expr.__dask_graph__())
    keys = list(expr.__dask_keys__())
    t2 = time.perf_counter()

    frisky.get_spans()  # clear span buffer
    tasks = translate_graph(dsk, keys)
    t3 = time.perf_counter()

    phases = defaultdict(float)
    for s in frisky.get_spans():
        if s["name"].startswith("client."):
            phases[s["name"]] += (s["end_ns"] - s["start_ns"]) / 1e9
    return dict(
        n=len(tasks),
        optimize_ms=1e3 * (t1 - t0),
        gen_ms=1e3 * (t2 - t1),
        translate_ms=1e3 * (t3 - t2),
        phases={k: 1e3 * v for k, v in phases.items()},
        wire=wire_breakdown(tasks),
    )


def report(label, r):
    n = r["n"]
    print(f"\n{label}: {n:,} tasks")
    print(f"  optimize {r['optimize_ms']:6.0f} ms   gen {r['gen_ms']:7.0f} ms   translate {r['translate_ms']:7.0f} ms")
    for p, ms in sorted(r["phases"].items()):
        print(f"      {p:28s} {ms:8.0f} ms")
    w = r["wire"]
    tot = sum(w.values())
    print(
        f"  wire {tot / 1e6:6.1f} MB = func {w['func'] / 1e6:5.1f}  "
        f"args {w['args'] / 1e6:5.1f}  keys {w['keys'] / 1e6:5.1f}  "
        f"deps {w['deps'] / 1e6:5.1f}   ({tot / n:.0f} B/task)"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default="add", choices=["add", "ones1", "from_array"])
    ap.add_argument("--blocks", type=int, default=100, help="per-dim block count; tasks ~ blocks^2")
    args = ap.parse_args()

    frisky.enable_tracing(1_000_000)
    collection = make_graph(args.graph, args.blocks)
    report(f"OLD {args.graph} blocks={args.blocks}", measure_old(collection))


if __name__ == "__main__":
    main()
