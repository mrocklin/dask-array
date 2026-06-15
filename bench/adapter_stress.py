"""Stress the generic GraphRecordsLayer adapter with *compositions*: a tail op
(adapter, reusing legacy _layer) feeding a Rust-covered op and vice versa. The
riskiest property of the adapter is cross-layer key consistency — its records'
keys/deps must match the canonical (plain-int) form the Rust/from_array layers
emit. Each case computes through a real Frisky cluster and checks numpy.

    PYTHONPATH=/Users/mrocklin/workspace/dask-array MATURIN_IMPORT_HOOK_ENABLED=0 \
      /Users/mrocklin/workspace/frisky/.venv/bin/python bench/adapter_stress.py
"""

import numpy as np

import dask
import dask_array as da
import frisky.dask as fdask
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


def _set(x, idx, val):
    x[idx] = val
    return x


def _npset(arr, idx, val):
    out = arr.copy()
    out[idx] = val
    return out


def main():
    calls = {"n": 0}
    orig = fdask._frisky_compute_collections

    def spy(client, collections):
        out = orig(client, collections)
        calls["n"] += out is not None
        return out

    fdask._frisky_compute_collections = spy
    ok_count = bad = 0
    with LocalCluster(n_workers=2) as cluster:
        with Client(cluster.scheduler):
            for label, build, expected in cases():
                before = calls["n"]
                try:
                    (got,) = dask.compute(build())
                    used = calls["n"] > before
                    match = np.allclose(np.asarray(got), expected) and np.shape(got) == np.shape(expected)
                except Exception as e:  # noqa: BLE001
                    print(f"  ERR  {label:<24} {type(e).__name__}: {str(e)[:55]}")
                    bad += 1
                    continue
                tag = "REC" if used else "dask"
                flag = "OK " if match else "BAD"
                ok_count += match
                bad += not match
                print(f"  {flag} [{tag:>4}] {label:<24} match={match}")

    fdask._frisky_compute_collections = orig
    print(f"\n{ok_count} correct, {bad} wrong/errored")


if __name__ == "__main__":
    main()
