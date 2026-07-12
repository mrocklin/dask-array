"""Stress the generic GraphRecordsLayer adapter with *compositions*: a tail op
(adapter, reusing legacy _layer) feeding a Rust-covered op and vice versa. The
riskiest property of the adapter is cross-layer key consistency — its records'
keys/deps must match the canonical (plain-int) form the Rust/from_array layers
emit. Each case computes through a real Frisky cluster and checks numpy.
Engagement is spied on both Frisky paths (see ``_spy.py``); note the preferred
expression path expands scheduler-side, so most cases exercise the adapter
there rather than through client-side records.

    PYTHONPATH=$PWD MATURIN_IMPORT_HOOK_ENABLED=0 \
      /Users/mrocklin/workspace/frisky/.venv/bin/python bench/adapter_stress.py
"""

import numpy as np

import dask
import dask_array as da
from _spy import TAGS, frisky_spy
from frisky import Client, LocalCluster


def cases():
    a = np.arange(48, dtype="f8").reshape(6, 8)
    b = np.arange(48, 96, dtype="f8").reshape(6, 8)
    sq = np.arange(36, dtype="f8").reshape(6, 6)
    v = np.arange(20, dtype="f8")
    fa = lambda c=(3, 4): da.from_array(a, chunks=c)
    fb = lambda c=(3, 4): da.from_array(b, chunks=c)
    fsq = lambda c=(3, 3): da.from_array(sq, chunks=c)
    fv = lambda c=5: da.from_array(v, chunks=c)

    # tail-op -> Rust-covered op (adapter keys consumed downstream)
    yield ("take then add", lambda: fa()[[0, 2, 4]] + 1.0, a[[0, 2, 4]] + 1.0)
    yield ("take then sum", lambda: fa()[[0, 2, 4]].sum(), a[[0, 2, 4]].sum())
    yield ("take+take add", lambda: fa()[[0, 2, 4]] + fb()[[0, 2, 4]], a[[0, 2, 4]] + b[[0, 2, 4]])
    yield ("diagonal then *2", lambda: da.diagonal(fsq()) * 2.0, np.diagonal(sq) * 2.0)
    yield ("diagonal then sum", lambda: da.diagonal(fsq()).sum(), np.diagonal(sq).sum())
    yield ("bool mask then sum", lambda: fv()[fv() > 5].sum(), v[v > 5].sum())
    yield ("bincount then sum", lambda: da.bincount(fv().astype("i8")).sum(), np.bincount(v.astype("i8")).sum())
    yield ("unique then add", lambda: da.unique(fv()) + 1.0, np.unique(v) + 1.0)
    yield ("setitem then *3", lambda: _set(fa(), (0,), 0.0) * 3.0, _npset(a, (0,), 0.0) * 3.0)

    # Rust-covered op -> tail-op (downstream adapter consumes Rust keys)
    yield ("add then take", lambda: (fa() + 1.0)[[1, 3, 5]], (a + 1.0)[[1, 3, 5]])
    yield ("rechunk then take", lambda: fa().rechunk((2, 8))[[0, 2]], a[[0, 2]])
    yield ("transpose then diagonal", lambda: da.diagonal(fsq().T), np.diagonal(sq.T))
    yield ("slice then bool mask", lambda: (lambda y: y[y > 10])(fv()[2:18]), (lambda y: y[y > 10])(v[2:18]))

    # tail-op -> tail-op
    yield (
        "take then diagonal",
        lambda: da.diagonal(fa()[:, :6][[0, 1, 2, 3, 4, 5]]),
        np.diagonal(a[:, :6][[0, 1, 2, 3, 4, 5]]),
    )
    yield ("unique then take", lambda: da.unique(fv())[[0, 2]], np.unique(v)[[0, 2]])

    # irregular chunks
    yield ("take irregular chunks", lambda: da.from_array(a, chunks=(2, 5))[[0, 3, 5]], a[[0, 3, 5]])
    yield ("diagonal irregular", lambda: da.diagonal(da.from_array(sq, chunks=(2, 4))), np.diagonal(sq))

    # overlap variants (old-style HLG, cross-layer key refs — the fragile case)
    yield (
        "overlap reflect+sum",
        lambda: fa().map_overlap(lambda b: b * 2, depth=1, boundary="reflect").sum(),
        (a * 2).sum(),
    )
    yield ("overlap depth2 none", lambda: fa().map_overlap(lambda b: b + 1, depth=2, boundary="none"), a + 1)
    yield ("overlap 1d periodic", lambda: fv().map_overlap(lambda b: b + 1, depth=2, boundary="periodic"), v + 1)
    yield (
        "overlap then take",
        lambda: fa().map_overlap(lambda b: b + 1, depth=1, boundary="none")[[0, 2]],
        (a + 1)[[0, 2]],
    )
    yield ("overlap reflect identity", lambda: fa().map_overlap(lambda b: b, depth=1, boundary="reflect"), a)
    # single chunk so dask's percentile is exact (multi-chunk dask is approximate)
    yield (
        "percentile then *2",
        lambda: da.percentile(fv(-1), [25, 50, 75]) * 2.0,
        np.percentile(v, [25, 50, 75]) * 2.0,
    )
    yield (
        "apply_along then sum",
        lambda: da.apply_along_axis(np.cumsum, 1, fa()).sum(),
        np.apply_along_axis(np.cumsum, 1, a).sum(),
    )


def _set(x, idx, val):
    x[idx] = val
    return x


def _npset(arr, idx, val):
    out = arr.copy()
    out[idx] = val
    return out


def main():
    ok_count = bad = 0
    with frisky_spy() as spy:
        with LocalCluster(n_workers=2) as cluster:
            with Client(cluster.scheduler):
                for label, build, expected in cases():
                    before = spy.snapshot()
                    try:
                        (got,) = dask.compute(build())
                        path = spy.path_since(before)
                        match = np.allclose(np.asarray(got), expected) and np.shape(got) == np.shape(expected)
                    except Exception as e:  # noqa: BLE001
                        print(f"  ERR  {label:<24} {type(e).__name__}: {str(e)[:55]}")
                        bad += 1
                        continue
                    flag = "OK " if match else "BAD"
                    ok_count += match
                    bad += not match
                    print(f"  {flag} [{TAGS[path]}] {label:<24} match={match}")

    print(
        f"\n{ok_count} correct, {bad} wrong/errored; "
        f"engaged: {spy.counts['expression']} expression, {spy.counts['records']} records"
    )


if __name__ == "__main__":
    main()
