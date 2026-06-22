"""Rechunk's ``threshold`` trades data copies against task count: each
intermediate step the planner inserts avoids tasks at the cost of an extra full
copy of the array. Frisky makes task overhead cheap, so a single direct rechunk
(one copy, more tasks) usually beats a multi-step plan. This benchmark measures
that tradeoff and locates the crossover, which is what motivates the default of
32 (see ``dask_array/__init__.py``).

    PYTHONPATH=/Users/mrocklin/workspace/dask-array \
      /Users/mrocklin/workspace/frisky/.venv/bin/python bench/rechunk_threshold.py

Representative results (Frisky LocalCluster, 4 threaded workers, float64):

  transpose 4000x4000 @100  (128 MB)   2 copies -> 705 ms   1 copy -> 585 ms  (-17%)
  3D swap  400x400x400 @50  (512 MB)    2 copies -> 2831 ms  1 copy -> 1800 ms (-36%)

Crossover (128 MB transpose, M chunks/axis; planner picks 2 copies at M>=2*threshold):
  M=20  (400 tasks)    1-copy 530 ms  vs 2-copy 714 ms   -> 1-copy +26%
  M=40  (1600 tasks)   1-copy 687 ms  vs 2-copy 705 ms   -> tie
  M=80  (6400 tasks)   1-copy 952 ms  vs 2-copy 728 ms   -> 2-copy wins
  M=160 (25600 tasks)  1-copy 2295 ms vs 2-copy 1162 ms  -> 2-copy 2x
The crossover (~a few thousand single-copy tasks) holds at 512 MB too, so the
copy savings do not scale up enough to justify a larger threshold. 32 lands the
switch right at the crossover.
"""

from __future__ import annotations

import statistics
import time

import dask
import numpy as np

import dask_array as da
from dask_array._rechunk import estimate_graph_size, plan_rechunk


def total_tasks(old, steps):
    cur, t = old, 0
    for s in steps:
        t += estimate_graph_size(cur, s)
        cur = s
    return t


def time_it(base, tgt, thr, repeat=5):
    times = []
    for _ in range(repeat):
        r = base.rechunk(tgt, threshold=thr)
        start = time.perf_counter()
        r.compute()
        times.append(time.perf_counter() - start)
    return min(times), statistics.median(times)


def main():
    import frisky

    # (shape, source_chunks, target_chunks) -- transpose-style rechunks, where
    # threshold actually changes the plan.
    scenarios = {
        "transpose 2000x2000 @100": ((2000, 2000), (100, 2000), (2000, 100)),
        "transpose 4000x4000 @100": ((4000, 4000), (100, 4000), (4000, 100)),
        "3D swap 400x400x400 @50": ((400, 400, 400), (50, 400, 50), (400, 50, 50)),
    }

    with frisky.LocalCluster(n_workers=4, processes=False, dashboard_address="127.0.0.1:0") as cluster:
        with frisky.Client(cluster.scheduler) as client:
            with dask.config.set({"scheduler": client}):
                for name, (shape, src, tgt) in scenarios.items():
                    mb = np.prod(shape) * 8 / 1e6
                    print(f"\n== {name}  (float64, {mb:.0f} MB)")
                    # Persist source into cluster memory so we only time the rechunk.
                    base = da.random.random(shape, chunks=src).persist()
                    base.compute()
                    tgt_chunks = base.rechunk(tgt).chunks
                    for thr in [4, 16, 64, 256]:
                        steps = plan_rechunk(base.chunks, tgt_chunks, 8, threshold=thr)
                        mn, med = time_it(base, tgt, thr)
                        print(
                            f"   thr={thr:4d}  copies={len(steps)}  "
                            f"tasks={total_tasks(base.chunks, steps):6d}  "
                            f"min={mn * 1000:7.1f} ms  median={med * 1000:7.1f} ms"
                        )


if __name__ == "__main__":
    main()
