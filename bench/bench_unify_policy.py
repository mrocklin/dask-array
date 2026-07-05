"""Chunk-unification policy: merge up to the coarser operand vs refine.

When elemwise operands disagree on chunking, ``unify_chunks_expr`` must pick a
common layout.  The old default (``array.unify-chunks-policy: coarse``) merges
nested chunkings up to the coarsest operand — fewer tasks, but one coarse
operand can inflate everything downstream (a 2-chunk time vector once turned
per-day 46 MB chunks into 3 GB chunks across a whole trading-model DAG).
Stock dask refines to the finest common boundaries — splits only, no data
movement, but per-element-chunk operands (xarray groupby patterns) shatter
their partners into thousands of tasks.  The cost-aware default
(``auto``) picks the direction per dimension: merge as before, unless the
bytes the merge would actually move exceed a small multiple of the bytes
already at the coarse layout — then refine that dimension instead.
``array.unify-chunks-limit`` (default 512 MiB) stays as a hard backstop on
manufactured chunk size under any merging policy.

Cases:
  nested_merge      the inflation bug shape: day-chunked 2D × 2-chunk 1D
                    (merge moves the heavy panel; auto must refine)
  shatter_guard     the case coarse exists for: coarse 3D − per-element indexed
                    (refine shatters; auto must merge)
  comparable_merge  equally heavy operands, nested layouts (the rolling-window
                    halo regime); auto must keep merging like coarse
  macro             the synthetic quantity DAG (bench/synthetic_quantity_expression.py)
                    with a coarse time vector multiplied in

Each (case, policy) runs in a subprocess: expressions are cached singletons,
so layouts computed under one config must not leak into the next.

    uv run python bench/bench_unify_policy.py            # metadata + threaded compute
    uv run python bench/bench_unify_policy.py --no-compute

Representative results (2026-07-03, 8-core dev box, threads scheduler, float64):

  == nested_merge (90k x 200 day-chunked 2D x nested 2-chunk 1D; .sum()) ==
    policy     build   tasks  max chunk           est transfer    wall
    auto       0.01s     306 468.75 kiB   712.0 B-35.71 MiB   0.13 s
    coarse     0.01s      99  91.55 MiB 134.28 MiB-276.03 MiB  0.20 s
    capped     0.01s     306 468.75 kiB   712.0 B-35.71 MiB   0.13 s
    refine     0.01s     306 468.75 kiB   712.0 B-35.71 MiB   0.12 s
  == shatter_guard (12000 x 20 x 30, 10 chunks vs per-element indexed) ==
    auto       0.03s   12401   5.49 MiB 106.69 MiB-222.58 MiB  2.55 s
    coarse     0.03s   12401   5.49 MiB 106.69 MiB-222.58 MiB  2.46 s
    capped     0.04s   12401   5.49 MiB 106.69 MiB-222.58 MiB  2.48 s
    refine     0.07s   60719   5.49 MiB 506.24 kiB-3.93 GiB   28.56 s
  == comparable_merge (90k x 200, 360 fine blocks x 10 coarse nested blocks) ==
    auto       0.01s     384  13.73 MiB 133.51 MiB-274.66 MiB  0.16 s
    coarse     0.01s     384  13.73 MiB 133.51 MiB-274.66 MiB  0.17 s
    capped     0.01s     384  13.73 MiB 133.51 MiB-274.66 MiB  0.26 s
    refine     0.01s    1212  13.73 MiB  2.80 kiB-4.96 GiB    0.36 s
  == macro (synthetic quantity DAG, complexity=1, x aligned coarse time vector) ==
    auto       5.34s   75237  90.00 kiB 62.94 MiB-1.32 GiB   20.16 s
    coarse     5.08s   75237  90.00 kiB 62.94 MiB-1.32 GiB   18.74 s
    capped     5.59s   75237  90.00 kiB 62.94 MiB-1.32 GiB   19.80 s
    refine     6.20s  695958  90.00 kiB 65.09 MiB-2.30 GiB   101.36 s

Reading: neither fixed direction is right -- merging is a pure loss on the
inflation shape (~50% slower wall, ~8x the transfer, unboundedly larger
chunks) but a pure win against shattering (11x) and on the macro DAG (5x,
9x the tasks).  The cost-aware auto policy picks the winner in every case:
it refines the inflation shape (the 2-chunk vector is ~200x lighter than
the panel it would inflate), merges the shatter and comparable shapes (the
moved bytes are backed by an anchor of equal weight), and on the macro DAG
produces a graph identical to coarse (its internal nested merges are all
either equal-weight or near-free fragment healing; macro wall differences
between auto/coarse/capped are same-graph noise, ~±1.5 s across runs).

Update (2026-07-05, realign unification + lower-time IO pushdown, see
bench_rechunk_insertion.py): auto still wins or ties every case, and its
macro graph dropped to ~48k tasks (was 75k, now below coarse's 50k) --
interleaved dims realign to an operand's grid instead of refining.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import dask
import numpy as np
from dask.utils import format_bytes

import dask_array as da
from dask_array._diagnostics import _max_chunk_bytes
from dask_array._expr import ArrayExpr

POLICIES = {
    # cost-aware default: merge unless the merge moves too many bytes
    "auto": {"array.unify-chunks-policy": "auto", "array.unify-chunks-limit": "512 MiB"},
    # the old fixed direction, unguarded
    "coarse": {"array.unify-chunks-policy": "coarse", "array.unify-chunks-limit": 0},
    # coarse with the size guard at a bench-visible threshold
    "capped": {"array.unify-chunks-policy": "coarse", "array.unify-chunks-limit": "16 MiB"},
    # stock-dask refinement
    "refine": {"array.unify-chunks-policy": "refine", "array.unify-chunks-limit": 0},
}

CASES = ["nested_merge", "shatter_guard", "comparable_merge", "macro"]


def build(case):
    """Return the array to compute for a case."""
    if case == "nested_merge":
        days, per_day, assets = 90, 1000, 200
        x = da.random.random((days * per_day, assets), chunks=(per_day, assets))
        t = da.random.random(days * per_day, chunks=((60 * per_day, 30 * per_day),))
        y = x * t[:, None]
        return ((y + x) * 2.0).sum()
    if case == "shatter_guard":
        n, groups, chunk = 12000, 4, 1200
        arr = da.random.random((n, 20, 30), chunks=(chunk, 20, 30))
        mean_arr = da.random.random((groups, 20, 30), chunks=(1, 20, 30))
        indexed = mean_arr[np.tile(np.arange(groups), n // groups), ...]
        return (arr - indexed).sum()
    if case == "comparable_merge":
        # two equally heavy operands, nested layouts (the rolling-window-halo
        # regime): merging stays worthwhile because the moved bytes are backed
        # by an operand of the same weight; refining shatters the coarse one
        n, assets = 90000, 200
        a = da.random.random((n, assets), chunks=(250, assets))  # 360 blocks
        b = da.random.random((n, assets), chunks=(9000, assets))  # 10 blocks, nested
        return (a * b + a).sum()
    if case == "macro":
        import os

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from synthetic_quantity_expression import synthetic_quantity_array

        stack = synthetic_quantity_array(complexity=1)  # (quantity, time, asset)
        n_time = stack.shape[1]
        # coarse time vector whose single boundary nests in the stack's time grid,
        # so the coarse policy is allowed to merge (misaligned would refine anyway)
        tchunks = stack.chunks[1]
        first = sum(tchunks[: max(1, 2 * len(tchunks) // 3)])
        t = da.random.random(n_time, chunks=((first, n_time - first),))
        return (stack * t[None, :, None]).mean()
    raise ValueError(case)


def measure(case, compute):
    t0 = time.perf_counter()
    out = build(case)
    expr = out.expr
    expr.chunks  # unification happens lazily; force it under this config
    opt = expr.optimize()
    build_s = time.perf_counter() - t0

    n_tasks = len(opt.__dask_graph__())
    max_chunk, lo, hi = 0, 0.0, 0.0
    for node in opt.walk():
        if not isinstance(node, ArrayExpr):
            continue
        try:
            size = _max_chunk_bytes(node)
        except Exception:
            size = None
        if size:
            max_chunk = max(max_chunk, size)
        t = node.transfer_bytes
        lo, hi = lo + t.min, hi + t.max

    wall = None
    if compute:
        t0 = time.perf_counter()
        dask.compute(out, scheduler="threads")
        wall = time.perf_counter() - t0
    return dict(build_s=build_s, n_tasks=n_tasks, max_chunk=max_chunk, transfer_lo=lo, transfer_hi=hi, wall=wall)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=CASES)
    p.add_argument("--policy", choices=list(POLICIES))
    p.add_argument("--no-compute", action="store_true")
    args = p.parse_args()

    if args.case:  # child mode: one (case, policy), JSON to stdout
        with dask.config.set(POLICIES[args.policy]):
            print(json.dumps(measure(args.case, compute=not args.no_compute)))
        return

    for case in CASES:
        print(f"\n== {case} ==")
        print(f"  {'policy':8s} {'build':>7s} {'tasks':>7s} {'max chunk':>10s} {'est transfer':>22s} {'wall':>7s}")
        for policy in POLICIES:
            cmd = [sys.executable, __file__, "--case", case, "--policy", policy]
            if args.no_compute:
                cmd.append("--no-compute")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode:
                sys.stderr.write(proc.stderr)
                proc.check_returncode()
            r = json.loads(proc.stdout)
            wall = f"{r['wall']:.2f} s" if r["wall"] is not None else "-"
            print(
                f"  {policy:8s} {r['build_s']:6.2f}s {r['n_tasks']:7d} "
                f"{format_bytes(r['max_chunk']):>10s} "
                f"{format_bytes(r['transfer_lo']):>9s}-{format_bytes(r['transfer_hi']):<10s} "
                f"{wall:>7s}"
            )


if __name__ == "__main__":
    main()
