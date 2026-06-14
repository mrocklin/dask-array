"""Client-side submission cost: the task-records path vs the materialized
(``__dask_graph__`` + ``translate_graph`` via ``frisky.dask.get``) path, on the
same unfused graph. Times *submission to futures* (gen + serialize + send) — the
client-side cost — not execution; gathers a handful of keys per round to confirm
both paths agree. Run with Frisky's venv:

    PYTHONPATH=/Users/mrocklin/workspace/dask-array \
      /Users/mrocklin/workspace/frisky/.venv/bin/python bench/bench_records.py
"""

import argparse
import time

import numpy as np
from dask.core import flatten

import frisky.dask as fdask
from frisky import Client, LocalCluster

from dask_array._frisky import collect_task_records


def build(blocks):
    import dask_array as da

    return lambda: da.ones((blocks, blocks), chunks=(1, 1)) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, nargs="+", default=[100, 200, 300])
    args = ap.parse_args()

    print(f"{'tasks':>9} | {'OLD ms':>8} {'NEW ms':>8} {'speedup':>8}")
    with LocalCluster(n_workers=4) as cluster:
        with Client(cluster.scheduler) as client:
            for blocks in args.blocks:
                make = build(blocks)
                n = 2 * blocks * blocks

                # OLD: materialize the per-task dict + translate + submit
                x = make()
                keys = [str(k) for k in flatten(x.__dask_keys__())]
                t = time.perf_counter()
                dsk = dict(x.__dask_graph__())
                old_futs = fdask.get(client, dsk, keys, sync=False)
                old_ms = (time.perf_counter() - t) * 1e3
                old_sample = client.gather(old_futs[:4])

                # NEW: generate task records in Rust + submit_tasks
                x = make()
                t = time.perf_counter()
                records = collect_task_records(x)
                new_futs = client.submit_tasks(records, keys)
                new_ms = (time.perf_counter() - t) * 1e3
                new_sample = client.gather(new_futs[:4])

                assert all(np.allclose(a, b) for a, b in zip(old_sample, new_sample))
                print(f"{n:>9,} | {old_ms:>7.0f}m {new_ms:>7.0f}m {old_ms / new_ms:>6.1f}x")


if __name__ == "__main__":
    main()
