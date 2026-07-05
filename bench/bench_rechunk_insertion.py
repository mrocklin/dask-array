"""Where should the optimizer insert rechunks it wasn't asked for?

Ops like ``roll`` produce chunk layouts offset from their input's (boundaries
shifted by ``shift % chunksize``).  When that output meets a regularly-chunked
sibling in an elemwise op, unification today can only refine — splitting both
operands at every boundary, doubling the block count with slivers that every
downstream op inherits.  The alternative is inserting a true rechunk back to
the sibling's layout: it moves ``min(s, c - s) / c`` of the array's bytes
(s = shift % chunksize, c = chunksize) but keeps the block count and, for
IO-backed operands, can push into the read and cost nothing.

Each case compares the status-quo plan against "oracle" plans with the
rechunk inserted by hand — the plans an insertion policy should discover:

  roll_sliver       x + roll(x, 1):     realign moves ~0.2% of the array;
                    refinement makes 1-row sliver chunks.  Realign should win.
  roll_half_chunk   x + roll(x, c/2):   realign moves half the array; refine
                    splits every block in two.  The genuinely contested case.
  roll_io           same as half_chunk but IO-backed: the roll's slices push
                    into the reads; the realign rechunk currently survives on
                    the concat axis (concat-axis pushdown would zero it).
  cross_io          IO a(1000x100) + IO b(100x1000): unify merges both and
                    pays real movement; any pre-inserted rechunk pushes into
                    the reads for free.  Oracle is strictly better everywhere.
  cross_random      same shape fight between two persisted (concrete) arrays:
                    movement is real for every plan; the ranking depends on
                    the scheduler, which is what makes it worth measuring.
  negative_control  roll(x, 1).sum(): nothing downstream needs alignment, so
                    inserting a rechunk is pure cost.  status_quo must win —
                    any policy that "always rechunks after roll" fails here.

Each (case, plan) runs in a subprocess: expressions are cached singletons, so
layouts computed under one config must not leak into the next.

    uv run python bench/bench_rechunk_insertion.py                # threads wall
    uv run python bench/bench_rechunk_insertion.py --no-compute   # static only
    uv run python bench/bench_rechunk_insertion.py --frisky       # Frisky LocalCluster

Static columns come from the optimized expression: task count, surviving
rechunk nodes, max chunk size, and summed ``transfer_bytes`` (min, max).
Note that the min model can never justify inserting a rechunk on its own —
refinement splits at the source and is always min-free — so expect the
interesting rankings to show up in tasks, the max model, and wall clock.

Representative results (2026-07-05, 8-core dev box).  Before the realign
unification rule and the lower-time IO pushdown, status_quo refined: the
roll cases ran 3363 tasks (137-150 ms threads), cross_io moved 879 MiB
through two TasksRechunks (117 ms).  After, status_quo discovers the oracle
plans by itself — identical structure on every roll case, and better than
either hand-written oracle on cross_io:

  case             plan        tasks rechunks  est transfer (min)  threads  frisky
  roll_sliver      status_quo   1809     1          0.98 MiB         94 ms  179 ms
  roll_half_chunk  status_quo   1809     1        244.14 MiB         99 ms  223 ms
  cross_io         status_quo    197     0           504.0 B         45 ms 1060 ms
  cross_random     status_quo   1477     2        878.91 MiB         93 ms  144 ms
  negative_control status_quo   1095     0          2.12 kiB         24 ms   81 ms

cross_random keeps its nested merge (fewest tasks, fastest on LocalCluster —
both hand-inserted alternatives lose 2-4x).  The realign row of
negative_control stays the strawman: inserting a rechunk with no alignment
demand downstream is a pure 2x loss.  The roll_io frisky walls (~2 s) are
dominated by from_array literal handling under Frisky, not by plan shape.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import dask
import numpy as np
from dask.utils import format_bytes

import dask_array as da
from dask_array._diagnostics import _max_chunk_bytes
from dask_array._expr import ArrayExpr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRISKY_PY = os.path.expanduser("~/workspace/frisky/.venv/bin/python")

N = 8000  # 8000 x 8000 float64 = 512 MB
CH = 500  # 16 x 16 blocks of ~2 MB

CASES = {
    "roll_sliver": ["status_quo", "realign"],
    "roll_half_chunk": ["status_quo", "realign"],
    "roll_io": ["status_quo", "realign"],
    "cross_io": ["status_quo", "rechunk_one", "rechunk_square"],
    "cross_random": ["status_quo", "rechunk_one", "rechunk_square"],
    "negative_control": ["status_quo", "realign"],
}


def build(case, plan):
    """Return the array to compute. Sources are persisted / in-memory so wall
    clock measures the operation, not data generation."""
    if plan not in CASES[case]:
        raise ValueError(f"{plan!r} is not a plan of {case!r}: {CASES[case]}")
    if case in ("roll_sliver", "roll_half_chunk", "negative_control"):
        shift = 1 if case != "roll_half_chunk" else CH // 2
        x = da.random.random((N, N), chunks=(CH, CH)).persist()
        r = da.roll(x, shift, axis=0)
        if plan == "realign":
            r = r.rechunk(x.chunks)
        if case == "negative_control":
            return r.sum()
        return (x + r).sum()

    if case == "roll_io":
        data = np.random.standard_normal((N, N))
        y = da.from_array(data, chunks=(CH, CH))
        r = da.roll(y, CH // 2, axis=0)
        if plan == "realign":
            r = r.rechunk(y.chunks)
        return (y + r).sum()

    if case == "cross_io":
        data = np.random.standard_normal((N, N))
        a = da.from_array(data, chunks=(1000, 100))
        b = da.from_array(data, chunks=(100, 1000))
    elif case == "cross_random":
        a = da.random.random((N, N), chunks=(1000, 100)).persist()
        b = da.random.random((N, N), chunks=(100, 1000)).persist()
    else:
        raise ValueError(case)
    if plan == "rechunk_one":
        a = a.rechunk(b.chunks)
    elif plan == "rechunk_square":
        a = a.rechunk((CH, CH))
        b = b.rechunk((CH, CH))
    return (a + b).sum()


def measure(case, plan, compute):
    out = build(case, plan)
    expr = out.expr
    t0 = time.perf_counter()
    expr.chunks  # unification happens lazily; force it before optimize
    opt = expr.optimize()
    build_s = time.perf_counter() - t0

    n_tasks = len(opt.__dask_graph__())
    max_chunk, lo, hi, n_rechunks = 0, 0.0, 0.0, 0
    for node in opt.walk():
        if not isinstance(node, ArrayExpr):
            continue
        if "Rechunk" in type(node).__name__:
            n_rechunks += 1
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
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            out.compute()
            times.append(time.perf_counter() - t0)
        wall = min(times)
    return dict(
        build_s=build_s,
        n_tasks=n_tasks,
        n_rechunks=n_rechunks,
        max_chunk=max_chunk,
        transfer_lo=lo,
        transfer_hi=hi,
        wall=wall,
    )


def child(args):
    if args.scheduler == "frisky":
        import frisky

        with frisky.LocalCluster(n_workers=4, processes=False, dashboard_address="127.0.0.1:0") as cluster:
            with frisky.Client(cluster.scheduler) as client:
                with dask.config.set({"scheduler": client}):
                    print(json.dumps(measure(args.case, args.plan, compute=not args.no_compute)))
    else:
        print(json.dumps(measure(args.case, args.plan, compute=not args.no_compute)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=list(CASES))
    p.add_argument("--plan")
    p.add_argument("--no-compute", action="store_true")
    p.add_argument("--frisky", action="store_true")
    p.add_argument("--scheduler", choices=["threads", "frisky"], default="threads")
    args = p.parse_args()

    if args.case:  # child mode: one (case, plan), JSON to stdout
        child(args)
        return

    interpreter = sys.executable
    env = None
    if args.frisky:
        if not os.path.exists(FRISKY_PY):
            sys.exit(f"--frisky needs {FRISKY_PY}")
        interpreter = FRISKY_PY
        env = dict(os.environ, PYTHONPATH=REPO)

    for case, plans in CASES.items():
        print(f"\n== {case} ==")
        print(
            f"  {'plan':16s} {'build':>7s} {'tasks':>7s} {'rechunks':>8s} "
            f"{'max chunk':>10s} {'est transfer':>22s} {'wall':>8s}"
        )
        for plan in plans:
            cmd = [interpreter, os.path.abspath(__file__), "--case", case, "--plan", plan]
            if args.no_compute:
                cmd.append("--no-compute")
            if args.frisky:
                cmd += ["--scheduler", "frisky"]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if proc.returncode:
                sys.stderr.write(proc.stderr)
                proc.check_returncode()
            r = json.loads(proc.stdout.strip().splitlines()[-1])
            wall = f"{r['wall'] * 1000:6.0f} ms" if r["wall"] is not None else "-"
            print(
                f"  {plan:16s} {r['build_s']:6.2f}s {r['n_tasks']:7d} {r['n_rechunks']:8d} "
                f"{format_bytes(r['max_chunk']):>10s} "
                f"{format_bytes(r['transfer_lo']):>9s}-{format_bytes(r['transfer_hi']):<10s} "
                f"{wall:>8s}"
            )


if __name__ == "__main__":
    main()
