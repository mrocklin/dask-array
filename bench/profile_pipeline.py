"""Profile the dask-array -> Frisky *submission* pipeline (NOT execution) for
large graphs, to find where client-side time goes before any task runs.

Phases timed (per graph):
  build    - construct the lazy collection (expr tree); should be ~free.
  lower    - expr.lower_completely(): dask-array's own expr lowering/simplify.
  optimize - expr.optimize(): adds blockwise FUSION on top of lower (measured
             separately so we can see fusion's cost and reach).
  records  - collect_task_records(): walk the lowered tree, build the flat
             (key, func, args, kwargs, deps) records. Rust for native layers
             (blockwise/rechunk/reduction/...), Python adapter for the tail and
             the random data source.
  submit   - client.submit_tasks(records, keys): Rust serialize + send +
             scheduler graph-ingest, on a WORKER-LESS cluster (n_workers=0), so
             nothing executes. We never gather.

Run with Frisky's venv (dask_array resolves via the editable install):
    FRISKY_PY=/Users/mrocklin/workspace/frisky/.venv/bin/python
    PYTHONPATH=/Users/mrocklin/workspace/dask-array MATURIN_IMPORT_HOOK_ENABLED=0 \
      $FRISKY_PY bench/profile_pipeline.py --blocks 200 400        # ~ up to 1M tasks
      $FRISKY_PY bench/profile_pipeline.py --blocks 400 --graph mixed --cprofile records
"""

import argparse
import cProfile
import pstats
import time
from io import StringIO

from dask.core import flatten

from dask_array._frisky import collect_task_records
from frisky import Client, LocalCluster


def make_graph(name, blocks):
    """Build a lazy collection whose lowered graph has ~ a few * blocks**2 tasks.

    `blocks` is the per-axis block count; the array is blocks*chunk on a side but
    is NEVER executed, so the element count is irrelevant — only block count is.
    """
    import dask_array as da

    chunk = 50
    side = blocks * chunk

    if name == "elementwise":
        # random source + a chain of blockwise + a reduction
        x = da.random.random((side, side), chunks=(chunk, chunk))
        y = da.sqrt(x * 2.0 + 1.0) - x
        y = da.where(y > 0, y, 0.0)
        return (y + 1.0).sum()
    if name == "rechunk":
        # random -> blockwise -> rechunk (re-grids) -> blockwise -> reduction
        x = da.random.random((side, side), chunks=(chunk, chunk))
        y = x * 2.0 + 1.0
        z = y.rechunk((chunk * 2, chunk * 2))
        return (da.sqrt(z) - 1.0).mean(axis=0)
    if name == "mixed":
        # the works: two random sources, blockwise binary, rechunk, reduction
        a = da.random.random((side, side), chunks=(chunk, chunk))
        b = da.random.random((side, side), chunks=(chunk, chunk))
        c = a * b + 1.0
        c = da.sqrt(c).rechunk((chunk * 2, chunk * 2))
        d = c - a.rechunk((chunk * 2, chunk * 2))
        return (d + 1.0).sum(axis=1)
    raise SystemExit(f"unknown graph {name!r}")


def time_phases(name, blocks, client):
    out = {}
    t = time.perf_counter()
    make_graph(name, blocks)
    out["build"] = (time.perf_counter() - t) * 1e3

    # lower (fresh expr each phase so caching doesn't hide work)
    c2 = make_graph(name, blocks)
    t = time.perf_counter()
    c2.expr.lower_completely()
    out["lower"] = (time.perf_counter() - t) * 1e3

    # optimize (fusion) — may or may not keep the records path representable
    c3 = make_graph(name, blocks)
    t = time.perf_counter()
    try:
        c3.optimize()
        out["optimize"] = (time.perf_counter() - t) * 1e3
    except Exception:
        out["optimize"] = float("nan")

    # records
    c4 = make_graph(name, blocks)
    t = time.perf_counter()
    records = collect_task_records(c4)
    out["records"] = (time.perf_counter() - t) * 1e3
    out["ntasks"] = len(records)

    # submit (worker-less; no gather)
    c5 = make_graph(name, blocks)
    recs5 = collect_task_records(c5)
    keys = [str(k) for k in flatten(c5.__dask_keys__())]
    t = time.perf_counter()
    client.submit_tasks(recs5, keys)
    out["submit"] = (time.perf_counter() - t) * 1e3
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, nargs="+", default=[100, 200, 400])
    ap.add_argument("--graph", default="mixed", choices=["elementwise", "rechunk", "mixed"])
    ap.add_argument(
        "--cprofile",
        default=None,
        choices=["records", "lower"],
        help="cProfile this phase at the largest --blocks and print top funcs",
    )
    args = ap.parse_args()

    print(f"graph={args.graph}")
    hdr = f"{'tasks':>10} | {'build':>7} {'lower':>7} {'optimize':>8} {'records':>8} {'submit':>8} | {'rec µs/task':>11} {'sub µs/task':>11}"
    print(hdr)
    print("-" * len(hdr))
    with LocalCluster(n_workers=0) as cluster:
        with Client(cluster.scheduler) as client:
            for blocks in args.blocks:
                p = time_phases(args.graph, blocks, client)
                n = p["ntasks"]
                print(
                    f"{n:>10,} | {p['build']:>6.0f}m {p['lower']:>6.0f}m {p['optimize']:>7.0f}m "
                    f"{p['records']:>7.0f}m {p['submit']:>7.0f}m | "
                    f"{p['records'] / n * 1e3:>10.2f} {p['submit'] / n * 1e3:>10.2f}"
                )

    if args.cprofile:
        blocks = args.blocks[-1]
        print(f"\n=== cProfile of '{args.cprofile}' phase @ blocks={blocks} (graph={args.graph}) ===")
        c = make_graph(args.graph, blocks)
        pr = cProfile.Profile()
        if args.cprofile == "records":
            pr.enable()
            collect_task_records(c)
            pr.disable()
        else:
            pr.enable()
            c.expr.lower_completely()
            pr.disable()
        s = StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
        print(s.getvalue())


if __name__ == "__main__":
    main()
