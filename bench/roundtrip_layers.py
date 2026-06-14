"""Frisky roundtrip for the PartialReduce + Rechunk layers.

Builds each collection on a `da.ones`/`da.full` base (all covered layers), runs
it through the transparent records path on an in-process Frisky cluster
(`dask.compute(x)` -> `Client.submit_tasks`), and checks the result against
numpy. Asserts the records path was actually taken (not a silent dask fallback).

    PYTHONPATH=/Users/mrocklin/workspace/dask-array \
      MATURIN_IMPORT_HOOK_ENABLED=0 \
      /Users/mrocklin/workspace/frisky/.venv/bin/python bench/roundtrip_layers.py
"""

import numpy as np

import dask
import dask_array as da
import frisky.dask as fdask
from frisky import Client, LocalCluster


def cases():
    # (label, build dask-array collection, numpy expected)
    o = lambda shp, c: da.ones(shp, chunks=c)
    yield ("sum all (2x2 grid)", lambda: o((6, 4), c=(3, 2)).sum(), np.ones((6, 4)).sum())
    yield ("sum axis0", lambda: o((6, 4), c=(3, 2)).sum(axis=0), np.ones((6, 4)).sum(0))
    yield ("sum axis1", lambda: o((6, 4), c=(3, 2)).sum(axis=1), np.ones((6, 4)).sum(1))
    yield (
        "sum keepdims",
        lambda: o((6, 4), c=(3, 2)).sum(axis=1, keepdims=True),
        np.ones((6, 4)).sum(1, keepdims=True),
    )
    yield ("mean axis0", lambda: o((9, 4), c=(2, 2)).mean(axis=0), np.ones((9, 4)).mean(0))
    yield ("sum 3d axis (0,2)", lambda: o((4, 3, 5), c=(2, 3, 2)).sum(axis=(0, 2)), np.ones((4, 3, 5)).sum((0, 2)))
    yield ("min full", lambda: (o((8, 8), c=(3, 3)) * 2).min(), np.full((8, 8), 2.0).min())
    yield ("max axis1 irregular", lambda: (o((7, 5), c=(3, 2)) + 1).max(axis=1), np.full((7, 5), 2.0).max(1))
    # rechunk
    yield ("rechunk regular", lambda: o((12, 12), c=(4, 4)).rechunk((3, 6)), np.ones((12, 12)))
    yield ("rechunk irregular", lambda: o((10, 10), c=(3, 3)).rechunk((4, 5)), np.ones((10, 10)))
    yield ("rechunk split+merge", lambda: o((16, 50), c=(16, 1)).rechunk((3, 10)), np.ones((16, 50)))
    yield ("rechunk 1d", lambda: o((20,), c=(7,)).rechunk((5,)), np.ones((20,)))
    yield ("rechunk then sum", lambda: o((12, 12), c=(4, 4)).rechunk((6, 3)).sum(axis=0), np.ones((12, 12)).sum(0))
    # from_array — DISTINCT data on a real cluster: validates serialization +
    # spatial assembly (which the da.ones cases above can't).
    b = np.arange(48, dtype="f8").reshape(6, 8)
    fa = lambda c: da.from_array(np.arange(48, dtype="f8").reshape(6, 8), chunks=c)
    yield ("from_array", lambda: fa((2, 4)), b)
    yield ("from_array + 1", lambda: fa((2, 4)) + 1, b + 1)
    yield ("from_array rechunk", lambda: fa((2, 4)).rechunk((3, 2)), b)
    yield ("from_array sum axis0", lambda: fa((2, 4)).sum(axis=0), b.sum(0))
    yield ("from_array rechunk sum", lambda: fa((3, 2)).rechunk((2, 4)).sum(axis=1), b.sum(1))


def main():
    calls = {"n": 0}
    orig = fdask._frisky_compute_collections

    def spy(client, collections):
        out = orig(client, collections)
        calls["n"] += out is not None
        return out

    fdask._frisky_compute_collections = spy

    failures = 0
    with LocalCluster(n_workers=2) as cluster:
        with Client(cluster.scheduler):
            for label, build, expected in cases():
                before = calls["n"]
                try:
                    (got,) = dask.compute(build())
                    used = calls["n"] > before
                    ok = np.allclose(np.asarray(got), expected) and np.shape(got) == np.shape(expected)
                except Exception as e:  # noqa: BLE001
                    ok, used = False, False
                    got = f"{type(e).__name__}: {str(e)[:60]}"
                bad = not (ok and used)
                failures += bad
                print(f"  {'BAD' if bad else 'OK '} {label:<26} match={ok} records={used}")

    fdask._frisky_compute_collections = orig
    print("\nroundtrip:", "all good" if not failures else f"{failures} FAILURES")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
