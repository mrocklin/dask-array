"""Frisky roundtrip for native dask-array layers.

Builds each collection on a `da.ones`/`da.full` base (all covered layers), runs
it through the transparent Frisky path on an in-process Frisky cluster, and
checks the result against numpy. Asserts Frisky was actually used (not a silent
dask fallback), spying both paths (see ``_spy.py``): current Frisky prefers
scheduler-side expression submission, falling back to client-side records.

    PYTHONPATH=$PWD MATURIN_IMPORT_HOOK_ENABLED=0 \
      /Users/mrocklin/workspace/frisky/.venv/bin/python bench/roundtrip_layers.py
"""

import numpy as np

import dask
import dask_array as da
from _spy import frisky_spy
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
    # manipulation layers (distinct data, real cluster -> serialization + assembly)
    r = np.arange(12, dtype="f8").reshape(3, 4)
    r1 = np.arange(6, dtype="f8").reshape(1, 6)
    yield (
        "broadcast_to",
        lambda: da.broadcast_to(da.from_array(r1, chunks=(1, 3)), (4, 6)),
        np.broadcast_to(r1, (4, 6)),
    )
    yield ("expand_dims", lambda: da.expand_dims(da.from_array(r, chunks=(3, 2)), axis=1), np.expand_dims(r, 1))
    yield ("squeeze", lambda: da.from_array(r.reshape(1, 3, 4), chunks=(1, 3, 2)).squeeze(axis=0), r)
    r2 = np.arange(12, 24, dtype="f8").reshape(3, 4)
    yield (
        "concatenate",
        lambda: da.concatenate([da.from_array(r, chunks=(3, 2)), da.from_array(r2, chunks=(3, 2))], axis=0),
        np.concatenate([r, r2], axis=0),
    )
    yield (
        "stack",
        lambda: da.stack([da.from_array(r, chunks=(3, 2)), da.from_array(r2, chunks=(3, 2))], axis=0),
        np.stack([r, r2], axis=0),
    )
    # basic slicing / getitem (distinct data, real cluster -> serialization + assembly)
    s = np.arange(60, dtype="f8").reshape(6, 10)
    fs = lambda c=(3, 5): da.from_array(s, chunks=c)
    yield ("slice 2d", lambda: fs()[1:5, 2:9], s[1:5, 2:9])
    yield ("slice 2d step", lambda: fs()[::2, 1::3], s[::2, 1::3])
    yield ("slice 2d neg step", lambda: fs()[::-1, 2:9], s[::-1, 2:9])
    yield ("integer row", lambda: fs()[4], s[4])
    yield ("integer + slice", lambda: fs()[3, 2:9], s[3, 2:9])
    yield ("slice + integer", lambda: fs()[1:5, 7], s[1:5, 7])
    yield ("scalar index", lambda: fs()[3, 4], s[3, 4])
    yield ("slice then slice", lambda: fs()[1:6, 1:9][1:4, 2:6], s[1:6, 1:9][1:4, 2:6])
    # blocks (distinct data, real cluster)
    yield ("blocks slice", lambda: fs().blocks[0:1, 1:], s[0:3, 5:])
    yield (
        "blocks reorder 1d",
        lambda: da.from_array(np.arange(20, dtype="f8"), chunks=5).blocks[[3, 1, 0, 2]],
        np.arange(20, dtype="f8").reshape(4, 5)[[3, 1, 0, 2]].ravel(),
    )
    # shuffle (distinct data, real cluster -> source takes + output-order restore)
    sh = np.arange(32, dtype="f8").reshape(4, 8)
    yield (
        "shuffle axis1",
        lambda: da.shuffle(da.from_array(sh, chunks=(2, 4)), [[6, 5, 2], [4, 1], [3, 0, 7]], axis=1),
        sh[:, [6, 5, 2, 4, 1, 3, 0, 7]],
    )
    yield (
        "shuffle axis0",
        lambda: da.shuffle(da.from_array(sh, chunks=(2, 4)), [[3, 1], [0, 2]], axis=0),
        sh[[3, 1, 0, 2], :],
    )
    # coarsen (distinct data, real cluster -> serialization + per-block reduction)
    cx = np.arange(48, dtype="f8").reshape(6, 8)
    yield (
        "coarsen 2d",
        lambda: da.coarsen(np.sum, da.from_array(cx, chunks=(3, 4)), {0: 2, 1: 2}),
        cx.reshape(3, 2, 4, 2).sum(axis=(1, 3)),
    )
    yield (
        "coarsen 1d",
        lambda: da.coarsen(np.max, da.from_array(np.arange(12, dtype="f8"), chunks=(6,)), {0: 3}),
        np.arange(12, dtype="f8").reshape(4, 3).max(axis=1),
    )
    # arange (1-D indexed creation, distinct values inherent)
    yield ("arange basic", lambda: da.arange(20, chunks=6), np.arange(20))
    yield ("arange int step", lambda: da.arange(5, 50, 3, chunks=7), np.arange(5, 50, 3))
    yield ("arange float step", lambda: da.arange(0, 10, 0.5, chunks=4), np.arange(0, 10, 0.5))
    # linspace (1-D indexed creation)
    yield ("linspace endpoint", lambda: da.linspace(0, 10, 20, chunks=6), np.linspace(0, 10, 20))
    yield (
        "linspace no endpoint",
        lambda: da.linspace(0, 10, 20, endpoint=False, chunks=7),
        np.linspace(0, 10, 20, endpoint=False),
    )
    # eye (2-D indexed creation)
    yield ("eye main diag", lambda: da.eye(8, chunks=3), np.eye(8))
    yield ("eye pos k off-square", lambda: da.eye(6, chunks=2, M=10, k=2), np.eye(6, 10, k=2))
    # diag (1-D -> 2-D matrix; off-diagonal uses per-task-kwargs zeros_like)
    dd = np.arange(8, dtype="f8")
    yield ("diag 1d->2d", lambda: da.diag(da.from_array(dd, chunks=3)), np.diag(dd))
    dd2 = np.arange(36, dtype="f8").reshape(6, 6)
    yield ("diag 2d->1d", lambda: da.diag(da.from_array(dd2, chunks=(2, 2))), np.diag(dd2))
    # reshape (distinct data, real cluster -> serialization + per-block reshape)
    rr = np.arange(48, dtype="f8").reshape(6, 8)
    yield ("reshape merge", lambda: da.from_array(rr, chunks=(3, 4)).reshape(48), rr.reshape(48))
    yield (
        "reshape split",
        lambda: da.from_array(np.arange(24, dtype="f8"), chunks=6).reshape(4, 6),
        np.arange(24, dtype="f8").reshape(4, 6),
    )
    # arg-reduction (argmin/argmax axis-wise; chunk = ArgChunk, combine = PartialReduce)
    ag = (np.arange(54, dtype="f8").reshape(9, 6) * 7) % 11  # distinct-ish, non-monotone
    yield ("argmin axis0", lambda: da.from_array(ag, chunks=(2, 2)).argmin(axis=0), ag.argmin(0))
    yield ("argmax axis1", lambda: da.from_array(ag, chunks=(2, 2)).argmax(axis=1), ag.argmax(1))
    yield ("argmin ravel", lambda: da.from_array(ag, chunks=(2, 2)).argmin(), np.array(ag.argmin()))
    # cumulative (sequential carry, real cluster -> serialization + cross-block carry)
    cu = np.arange(54, dtype="f8").reshape(9, 6)
    yield ("cumsum axis0", lambda: da.from_array(cu, chunks=(2, 2)).cumsum(axis=0), cu.cumsum(0))
    yield ("cumsum axis1", lambda: da.from_array(cu, chunks=(2, 2)).cumsum(axis=1), cu.cumsum(1))
    yield ("cumprod axis0", lambda: da.from_array(cu % 3 + 1, chunks=(2, 2)).cumprod(axis=0), (cu % 3 + 1).cumprod(0))


def main():
    failures = 0
    with frisky_spy() as spy:
        with LocalCluster(n_workers=2) as cluster:
            with Client(cluster.scheduler):
                for label, build, expected in cases():
                    before = spy.snapshot()
                    try:
                        (got,) = dask.compute(build())
                        path = spy.path_since(before)
                        ok = np.allclose(np.asarray(got), expected) and np.shape(got) == np.shape(expected)
                    except Exception as e:  # noqa: BLE001
                        ok, path = False, "none"
                        got = f"{type(e).__name__}: {str(e)[:60]}"
                    bad = not (ok and path != "none")
                    failures += bad
                    print(f"  {'BAD' if bad else 'OK '} {label:<26} match={ok} frisky={path}")

    print(f"\nengaged: {spy.counts['expression']} expression, {spy.counts['records']} records")
    print("roundtrip:", "all good" if not failures else f"{failures} FAILURES")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
