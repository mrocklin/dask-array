"""Chunk-unification policy: merge up to the coarser operand vs refine.

When elemwise operands disagree on chunking, ``unify_chunks_expr`` must pick a
common layout.  Today's default (``array.unify-chunks-policy: coarse``) merges
nested chunkings up to the coarsest operand — fewer tasks, but one coarse
operand can inflate everything downstream (a 2-chunk time vector once turned
per-day 46 MB chunks into 3 GB chunks across a whole trading-model DAG).
Stock dask refines to the finest common boundaries — splits only, no data
movement, but per-element-chunk operands (xarray groupby patterns) shatter
their partners into thousands of tasks.  ``array.unify-chunks-limit`` (default
512 MiB) now caps the merge direction; this benchmark measures the whole
tradeoff to inform whether the *default direction* should change.

Cases:
  nested_merge   the inflation bug shape: day-chunked 2D × 2-chunk 1D
  shatter_guard  the case coarse exists for: coarse 3D − per-element indexed
  macro          the synthetic quantity DAG (bench/synthetic_quantity_expression.py)
                 with a coarse time vector multiplied in

Each (case, policy) runs in a subprocess: expressions are cached singletons,
so layouts computed under one config must not leak into the next.

    uv run python bench/bench_unify_policy.py            # metadata + threaded compute
    uv run python bench/bench_unify_policy.py --no-compute

Representative results (2026-07-03, 8-core dev box, threads scheduler, float64):

  == nested_merge (90k x 200 day-chunked 2D x nested 2-chunk 1D; .sum()) ==
    policy     build   tasks  max chunk           est transfer    wall
    coarse     0.01s      99  91.55 MiB 134.28 MiB-276.03 MiB  0.22 s
    capped     0.01s     306 468.75 kiB   712.0 B-35.71 MiB   0.13 s
    refine     0.01s     306 468.75 kiB   712.0 B-35.71 MiB   0.13 s
  == shatter_guard (12000 x 20 x 30, 10 chunks vs per-element indexed) ==
    coarse     0.03s   12401   5.49 MiB 106.69 MiB-222.58 MiB  2.39 s
    capped     0.03s   12401   5.49 MiB 106.69 MiB-222.58 MiB  2.39 s
    refine     0.07s   60719   5.49 MiB 506.24 kiB-3.93 GiB    9.09 s
  == macro (synthetic quantity DAG, complexity=1, x aligned coarse time vector) ==
    coarse     5.38s   75236  90.00 kiB 62.94 MiB-1.32 GiB   18.24 s
    capped     5.25s   75236  90.00 kiB 62.94 MiB-1.32 GiB   18.38 s
    refine     6.46s  695957  90.00 kiB 65.09 MiB-2.30 GiB   97.90 s

Reading: the merge direction is a pure loss on the inflation shape (40% slower
wall, ~8x the transfer, and unboundedly larger chunks) but a pure win against
shattering (4x) and on the macro DAG (5x, 9x the tasks).  Neither fixed
direction is right; a byte-capped coarse default keeps both pathologies out.
The interesting #2 follow-up is a cost-aware rule (rechunk the lighter operand
toward the heavier operand's layout, still under the cap), which picks the
winner in all three cases above by construction.
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
    # today's behavior, unguarded
    "coarse": {"array.unify-chunks-policy": "coarse", "array.unify-chunks-limit": 0},
    # coarse with the size guard at a bench-visible threshold
    "capped": {"array.unify-chunks-policy": "coarse", "array.unify-chunks-limit": "16 MiB"},
    # stock-dask refinement
    "refine": {"array.unify-chunks-policy": "refine", "array.unify-chunks-limit": 0},
}


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
    p.add_argument("--case", choices=["nested_merge", "shatter_guard", "macro"])
    p.add_argument("--policy", choices=list(POLICIES))
    p.add_argument("--no-compute", action="store_true")
    args = p.parse_args()

    if args.case:  # child mode: one (case, policy), JSON to stdout
        with dask.config.set(POLICIES[args.policy]):
            print(json.dumps(measure(args.case, compute=not args.no_compute)))
        return

    for case in ["nested_merge", "shatter_guard", "macro"]:
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
