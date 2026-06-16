"""Coverage probe: which common dask-array operations take the Frisky records
path end-to-end (vs. fall back to legacy dask because some node in the lowered
tree lacks a `_frisky_layer`)?

The records path is all-or-nothing: `dask.compute(op)` engages it only if EVERY
expr in the lowered tree is covered. So `records=True` here means the whole
operation — including all the layers it lowers to — is Rust-generated, and the
result still matches numpy. `records=False` pinpoints an operation that hits an
uncovered layer: the data-driven priority list for the remaining tail.

    PYTHONPATH=/Users/mrocklin/workspace/dask-array \
      MATURIN_IMPORT_HOOK_ENABLED=0 \
      /Users/mrocklin/workspace/frisky/.venv/bin/python bench/coverage_probe.py
"""

import numpy as np

import dask
import dask_array as da
import frisky.dask as fdask
from frisky import Client, LocalCluster


def _setitem(x, idx, val):
    x[idx] = val
    return x


def _np_setitem(arr, idx, val):
    out = arr.copy()
    out[idx] = val
    return out


def cases():
    # (label, build dask-array collection, numpy expected)
    a = np.arange(48, dtype="f8").reshape(6, 8)
    b = np.arange(48, 96, dtype="f8").reshape(6, 8)
    fa = lambda c=(3, 4): da.from_array(a, chunks=c)
    fb = lambda c=(3, 4): da.from_array(b, chunks=c)

    # --- elementwise / blockwise family ---
    yield ("add", lambda: fa() + fb(), a + b)
    yield ("scalar mul", lambda: fa() * 2.0, a * 2.0)
    yield ("astype", lambda: fa().astype("f4"), a.astype("f4"))
    yield ("clip", lambda: da.clip(fa(), 10, 30), np.clip(a, 10, 30))
    yield ("where", lambda: da.where(fa() > 20, fa(), 0.0), np.where(a > 20, a, 0.0))
    yield ("ufunc sqrt", lambda: da.sqrt(fa()), np.sqrt(a))
    yield ("negative", lambda: -fa(), -a)
    yield ("maximum", lambda: da.maximum(fa(), fb() - 50), np.maximum(a, b - 50))

    # --- transpose / reshape ---
    yield ("transpose .T", lambda: fa().T, a.T)
    yield ("transpose axes", lambda: fa().transpose(1, 0), a.transpose(1, 0))
    yield ("reshape flat", lambda: fa().reshape(48), a.reshape(48))
    yield ("reshape split", lambda: fa().reshape(2, 3, 8), a.reshape(2, 3, 8))
    yield ("ravel", lambda: fa().ravel(), a.ravel())

    # --- reductions ---
    yield ("sum axis0", lambda: fa().sum(axis=0), a.sum(0))
    yield ("mean", lambda: fa().mean(), a.mean())
    yield ("std", lambda: fa().std(), a.std())
    yield ("var axis1", lambda: fa().var(axis=1), a.var(axis=1))
    yield ("min", lambda: fa().min(), a.min())
    yield ("prod axis0", lambda: (fa() / 10).prod(axis=0), (a / 10).prod(0))

    # --- linear algebra-ish (blockwise + reduction) ---
    yield ("matmul", lambda: fa() @ fb().T, a @ b.T)
    yield ("tensordot", lambda: da.tensordot(fa(), fb(), axes=([1], [1])), np.tensordot(a, b, axes=([1], [1])))
    yield ("dot 1d", lambda: da.from_array(a[0], chunks=4) @ da.from_array(b[0], chunks=4), a[0] @ b[0])

    # --- slicing + compute compositions ---
    # pure slice pushed into the source ndarray as a FromArray _region (native path)
    yield ("fromarray region", lambda: fa()[1:5, 2:7], a[1:5, 2:7])
    yield ("fromarray region 1block", lambda: da.from_array(a, chunks=-1)[2:5, 3:7], a[2:5, 3:7])
    yield ("slice + add", lambda: fa()[1:5, 2:7] + 1, a[1:5, 2:7] + 1)
    yield ("slice neg step", lambda: fa()[::-1] * 2, a[::-1] * 2)
    yield ("rechunk + sum", lambda: fa().rechunk((2, 8)).sum(axis=1), a.sum(1))
    yield (
        "concatenate + mean",
        lambda: da.concatenate([fa(), fb()], axis=0).mean(axis=0),
        np.concatenate([a, b], 0).mean(0),
    )
    yield ("stack + sum", lambda: da.stack([fa(), fb()]).sum(axis=0), np.stack([a, b]).sum(0))
    yield ("coarsen + add", lambda: da.coarsen(np.sum, fa(), {0: 2, 1: 2}) + 1, a.reshape(3, 2, 4, 2).sum((1, 3)) + 1)
    yield ("broadcast + mul", lambda: (fa() * da.from_array(a[0], chunks=4)), a * a[0])
    sq = np.arange(36, dtype="f8").reshape(6, 6)
    yield ("diag of 2d", lambda: da.diag(da.from_array(sq, chunks=(3, 3))), np.diag(sq))

    # --- indexed creation composed ---
    yield ("arange + reshape", lambda: da.arange(24, chunks=6).reshape(4, 6), np.arange(24).reshape(4, 6))
    yield ("eye @ x", lambda: da.eye(6, chunks=3) @ fa(), np.eye(6) @ a)

    # --- random (data-source layer; check records-path + finite, not exact) ---
    yield ("random+rechunk+sum", lambda: da.random.random((40, 40), chunks=(1, -1)).rechunk((-1, 1)).sum(), None)
    yield ("normal mean", lambda: da.random.normal(0, 1, (30, 20), chunks=(10, 10)).mean(), None)

    # --- structural ops with inline constant arrays (FromArray broadened) ---
    yield ("pad", lambda: da.pad(fa(), 1, mode="constant"), np.pad(a, 1, mode="constant"))
    yield ("triu", lambda: da.triu(da.from_array(sq, chunks=(3, 3))), np.triu(sq))
    yield ("tril", lambda: da.tril(da.from_array(sq, chunks=(3, 3))), np.tril(sq))
    yield ("isin", lambda: da.isin(fa(), np.array([1.0, 5.0, 9.0])), np.isin(a, np.array([1.0, 5.0, 9.0])))

    # --- unknown-chunk ops (ChunksOverride alias; nan-chunk metadata) ---
    v = np.arange(12, dtype="f8")
    fv = lambda c=4: da.from_array(v, chunks=c)
    yield ("nonzero", lambda: fv().nonzero()[0], v.nonzero()[0])
    yield ("argwhere", lambda: da.argwhere(fv() > 3), np.argwhere(v > 3))
    yield ("topk", lambda: da.topk(fv(), 3), np.array([11.0, 10.0, 9.0]))

    # --- specialized tail, now covered via GraphRecordsLayer (reuse _layer) ---
    sq = np.arange(36, dtype="f8").reshape(6, 6)
    fsq = lambda c=(3, 3): da.from_array(sq, chunks=c)
    yield ("take list-index", lambda: fa()[[0, 2, 4]], a[[0, 2, 4]])
    yield ("take axis1", lambda: fa()[:, [1, 3, 5]], a[:, [1, 3, 5]])
    yield ("bool mask 1d", lambda: fv()[fv() > 5], v[v > 5])
    yield ("vindex", lambda: fa().vindex[[0, 2], [1, 3]], a[[0, 2], [1, 3]])
    yield ("setitem scalar", lambda: _setitem(fa(), (0, 0), 99.0), _np_setitem(a, (0, 0), 99.0))
    yield ("setitem slice", lambda: _setitem(fa(), (slice(0, 2),), 0.0), _np_setitem(a, (slice(0, 2),), 0.0))
    yield ("diagonal", lambda: da.diagonal(fsq()), np.diagonal(sq))
    yield ("map_overlap +1", lambda: fa().map_overlap(lambda b: b + 1, depth=1, boundary="none"), a + 1)
    yield ("apply_along_axis", lambda: da.apply_along_axis(np.cumsum, 0, fa()), np.apply_along_axis(np.cumsum, 0, a))
    yield ("histogram", lambda: da.histogram(fv(), bins=4, range=(0, 12))[0], np.histogram(v, bins=4, range=(0, 12))[0])
    yield ("unique", lambda: da.unique(fv()), np.unique(v))
    yield ("bincount", lambda: da.bincount(fv().astype("i8")), np.bincount(v.astype("i8")))
    # single chunk: dask's multi-chunk percentile is approximate (≠ np.percentile)
    yield ("percentile", lambda: da.percentile(fv(-1), [50]), np.percentile(v, [50]))

    # --- FusedBlockwise: an op-chain fused into one task/block ---
    # `.optimize()` performs the blockwise fusion that `x.compute()` does, so the
    # records path sees a FusedBlockwise node (covered natively). Without the fix
    # these fell back to legacy dask; now they take the records path end-to-end.
    yield ("fused chain", lambda: da.sqrt(fa() * 2 + 1).optimize(), np.sqrt(a * 2 + 1))
    yield ("fused two-input", lambda: (fa() * fb() + 1).optimize(), a * b + 1)
    yield ("fused -> reduction", lambda: da.sqrt(fa() * 2 + 1).sum().optimize(), np.sqrt(a * 2 + 1).sum())
    yield ("fused transpose chain", lambda: (fa().T * 2 + 1).optimize(), a.T * 2 + 1)

    # --- known-uncovered (expect records=False, fall back) ---
    yield ("cumsum [tail]", lambda: fa().cumsum(axis=0), a.cumsum(0))
    yield ("argmin [tail]", lambda: fa().argmin(axis=0), a.argmin(0))
    yield ("argmax ravel [tail]", lambda: fa().argmax(), np.array(a.argmax()))


def main():
    calls = {"n": 0}
    orig = fdask._frisky_compute_collections

    def spy(client, collections):
        out = orig(client, collections)
        calls["n"] += out is not None
        return out

    fdask._frisky_compute_collections = spy

    covered, fellback, failed = [], [], []
    with LocalCluster(n_workers=2) as cluster:
        with Client(cluster.scheduler):
            for label, build, expected in cases():
                before = calls["n"]
                try:
                    (got,) = dask.compute(build())
                    used = calls["n"] > before
                    if expected is None:
                        # random: no exact reference — just require finite output.
                        ok = bool(np.all(np.isfinite(np.asarray(got))))
                    else:
                        ok = np.allclose(np.asarray(got), expected) and np.shape(got) == np.shape(expected)
                except Exception as e:  # noqa: BLE001
                    failed.append((label, f"{type(e).__name__}: {str(e)[:70]}"))
                    print(f"  ERR  {label:<22} {type(e).__name__}: {str(e)[:60]}")
                    continue
                tag = "REC" if used else "dask"
                (covered if used else fellback).append(label)
                flag = "OK " if ok else "BAD"
                print(f"  {flag} [{tag:>4}] {label:<22} match={ok}")

    fdask._frisky_compute_collections = orig
    print(f"\nrecords-path (fully Rust-generated): {len(covered)}/{len(covered) + len(fellback) + len(failed)}")
    print("  fell back to dask:", ", ".join(fellback) or "(none)")
    if failed:
        print("  ERRORED:", ", ".join(l for l, _ in failed))


if __name__ == "__main__":
    main()
