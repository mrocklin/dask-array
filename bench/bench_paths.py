"""Rust task generation vs the legacy pure-Python path, per stage, on large graphs.

For a given graph and size it times the client preamble in stages:

  * gen (Rust)   - ``x.__dask_graph__()`` with the Rust layers (the default).
  * gen (Python) - the same, with ``_frisky_layer`` forced to fall back to the
                   legacy ``core_blockwise`` / ``BroadcastTrick`` Python loops.
  * translate    - ``frisky.dask.translate_graph`` on the resulting graph; this
                   is shared by both paths (same dask Tasks) and is the per-task
                   pickle the future compact-template work would remove.

Graphs are built unfused (``__dask_graph__`` doesn't fuse), so the blockwise +
creation layers are actually exercised rather than collapsing to FusedBlockwise.

Run with Frisky's venv; ``dask_array`` resolves from this checkout:

    PYTHONPATH=/Users/mrocklin/workspace/dask-array \
      /Users/mrocklin/workspace/frisky/.venv/bin/python bench/bench_paths.py
"""

import argparse
import time

from dask.core import flatten

import dask_array._blockwise as _bw
import dask_array.creation._ones_zeros as _oz
from frisky.dask import translate_graph

# Capture the real Rust-routing methods so we can toggle the legacy fallback.
_ORIG_BW = _bw.Blockwise._frisky_layer
_ORIG_OZ = _oz.BroadcastTrick._frisky_layer


def _force_legacy(on):
    def _ni(self):
        raise NotImplementedError

    _bw.Blockwise._frisky_layer = _ni if on else _ORIG_BW
    _oz.BroadcastTrick._frisky_layer = _ni if on else _ORIG_OZ


def make_graph(name, blocks):
    import dask_array as da

    n = blocks
    if name == "ones1":  # creation + elementwise add (per-block dep)
        return lambda: da.ones((n, n), chunks=(1, 1)) + 1
    if name == "ab":  # two creations + binary elementwise
        return lambda: da.ones((n, n), chunks=(1, 1)) + da.ones((n, n), chunks=(1, 1))
    raise SystemExit(f"unknown graph {name!r}")


def time_gen(build):
    # Fresh collection each time so cached _layer results don't hide the cost.
    x = build()
    t0 = time.perf_counter()
    dsk = dict(x.__dask_graph__())
    dt = time.perf_counter() - t0
    return dsk, x, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="ones1", choices=["ones1", "ab"])
    ap.add_argument("--blocks", type=int, nargs="+", default=[100, 300, 600])
    args = ap.parse_args()

    print(f"graph={args.graph}  (tasks ~ 2 x blocks^2, unfused)")
    print(f"{'tasks':>10} {'gen Rust':>10} {'gen Python':>11} {'speedup':>8} {'translate':>10}")
    for blocks in args.blocks:
        build = make_graph(args.graph, blocks)

        _force_legacy(False)
        dsk, x, t_rust = time_gen(build)
        _force_legacy(True)
        _, _, t_legacy = time_gen(build)
        _force_legacy(False)

        keys = list(flatten(x.__dask_keys__()))
        t0 = time.perf_counter()
        translate_graph(dsk, keys)
        t_tr = time.perf_counter() - t0

        speedup = t_legacy / t_rust if t_rust else float("nan")
        print(f"{len(dsk):>10,} {t_rust * 1e3:>9.0f}m {t_legacy * 1e3:>10.0f}m {speedup:>7.1f}x {t_tr * 1e3:>9.0f}m")


if __name__ == "__main__":
    main()
