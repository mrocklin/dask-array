"""Adversarial differential probe for the GraphRecordsLayer adapter.

Same structure as diff_records.py (records path vs forced stock-dask
synchronous), but loaded with cases chosen to break the adapter's specific
assumptions:

  - float block coords surviving cross-layer (overlap depth>1, multi-chunk)
  - single-block from_array (copy path) + getitem/asarray operands
  - object dtype, datetime dtype, structured dtype (string-in-value)
  - dict-NestedContainer producing ops
  - 3D / higher-rank fancy + overlap
  - deeply nested compositions tail<->tail<->covered
  - inline DataNode carrying ndarray (large/mutable value identity)
  - empty results / zero-size blocks
  - tuple-returning ops (gradient, nonzero) - skipped (non-array)

    PYTHONPATH=/Users/mrocklin/workspace/dask-array MATURIN_IMPORT_HOOK_ENABLED=0 \
      /Users/mrocklin/workspace/frisky/.venv/bin/python bench/diff_adversarial.py
"""

import numpy as np

import dask
import dask_array as da
import frisky.dask as fdask
from frisky import Client, LocalCluster


def ops():
    a = np.arange(60, dtype="f8").reshape(6, 10)
    a3 = np.arange(2 * 4 * 6, dtype="f8").reshape(2, 4, 6)
    sq = np.arange(64, dtype="f8").reshape(8, 8) + 1.0
    v = np.arange(24, dtype="f8") - 8.0
    fa = lambda c=(3, 5): da.from_array(a, chunks=c)
    fa3 = lambda c=(1, 2, 3): da.from_array(a3, chunks=c)
    fsq = lambda c=(3, 4): da.from_array(sq, chunks=c)
    fv = lambda c=5: da.from_array(v, chunks=c)

    # ---- float coords cross-layer: overlap depth>=2 multi-chunk, then a covered op
    yield (
        "overlap d2 + slice + sum",
        lambda: fa().map_overlap(lambda b: b + 1, depth=2, boundary="reflect")[1:5, 2:8].sum(),
    )
    yield "overlap d2 1d + add", lambda: fv().map_overlap(lambda b: b * 2, depth=2, boundary="periodic") + 100.0
    yield "overlap 3d + mean", lambda: fa3().map_overlap(lambda b: b + 1, depth=1, boundary="none").mean()
    yield (
        "overlap then take then add",
        lambda: fa().map_overlap(lambda b: b + 1, depth=2, boundary="none")[[0, 2, 4]] + 5.0,
    )

    # ---- single-block from_array (copy path) feeding tail ops
    yield "single-block diagonal", lambda: da.diagonal(da.from_array(sq, chunks=(8, 8)))
    yield (
        "single-block overlap",
        lambda: da.from_array(a, chunks=(6, 10)).map_overlap(lambda b: b + 1, depth=1, boundary="reflect"),
    )
    yield "single-block take", lambda: da.from_array(a, chunks=(6, 10))[[5, 0, 3]]

    # ---- 3D fancy + structural
    yield "3d take axis0", lambda: fa3()[[1, 0, 1]]
    yield "3d diagonal", lambda: da.diagonal(fa3()[0])
    yield "3d flip+roll", lambda: da.roll(da.flip(fa3(), axis=2), 1, axis=0)
    yield "3d transpose+take", lambda: da.transpose(fa3(), (2, 0, 1))[[0, 5]]
    yield "3d pad", lambda: da.pad(fa3(), 1, mode="constant")
    yield "3d moveaxis", lambda: da.moveaxis(fa3(), 0, -1)

    # ---- object / datetime / structured dtype (string-in-value, non-numeric)
    yield "datetime sort", lambda: _dt_sorted()
    yield "object take", lambda: _obj_take()

    # ---- ops likely producing dict-NestedContainers / kwargs-heavy
    yield "histogram bins", lambda: da.histogram(fv(), bins=8, range=(-8, 16))[0]
    yield "histogramdd", lambda: _histdd(fa())
    yield "digitize", lambda: da.digitize(fv(), bins=[-4, 0, 4, 8])
    yield "unique counts", lambda: da.unique((fv() + 8).astype("i8") % 4, return_counts=True)[1]

    # ---- empty / zero-size results
    yield "bool mask empty", lambda: fv()[fv() > 1000]
    yield "take empty", lambda: fa()[[]]
    yield "argwhere empty", lambda: da.argwhere(fa() > 1e9)

    # ---- deep tail<->tail<->covered compositions
    yield "diag(overlap)", lambda: da.diagonal(fsq().map_overlap(lambda b: b + 1, depth=1, boundary="reflect"))
    yield "overlap(take)", lambda: fa()[[0, 1, 2, 3]].map_overlap(lambda b: b + 1, depth=1, boundary="none")
    yield "pad(take)+sum", lambda: da.pad(fa()[[0, 2, 4]], 1, mode="reflect").sum()
    yield "unique(setitem)", lambda: da.unique(_set(fa(), fa() > 30, 0.0).astype("i8"))
    yield "roll(diagonal)", lambda: da.roll(da.diagonal(fsq()), 2)
    yield "cov(take)", lambda: da.cov(fa()[[0, 2, 4]])

    # ---- repeated/tiled (value duplication)
    yield "tile 2d", lambda: da.tile(fv(), (2, 3))
    yield "repeat axis", lambda: da.repeat(fa(), 2, axis=0)

    # ---- negative-step + fancy combos
    yield "negstep then overlap", lambda: fa()[::-1].map_overlap(lambda b: b + 1, depth=1, boundary="none")
    yield "double fancy", lambda: fa()[[0, 2, 4]][:, [9, 0, 3]]

    # ---- block_id / blockwise meta (map_blocks with block_info)
    yield "map_blocks block_info", lambda: _mb_blockinfo(fa())


def _set(x, idx, val):
    x[idx] = val
    return x


def _dt_sorted():
    d = (np.arange(12) * 86400_000_000_000).astype("datetime64[ns]")[::-1]
    x = da.from_array(d, chunks=4)
    return x[da.argsort(x)] if hasattr(da, "argsort") else x[[11, 0, 5, 3]]


def _obj_take():
    o = np.array([{"a": i} for i in range(8)], dtype=object)
    return da.from_array(o, chunks=3)[[7, 0, 4, 4]]


def _histdd(fa):
    # histogramdd needs sample columns; build a 2-col sample
    x = fa[:, 0]
    y = fa[:, 1]
    s = da.stack([x.flatten(), y.flatten()], axis=1)
    return da.histogramdd(s, bins=4)[0]


def _mb_blockinfo(x):
    def f(block, block_info=None):
        # add the block's first global index so a mismatched block_info shows up
        loc = block_info[0]["array-location"][0][0]
        return block + loc

    return x.map_blocks(f, dtype=x.dtype)


def main():
    orig = fdask._frisky_compute_collections
    used = {"n": 0}

    def spy(client, collections):
        out = orig(client, collections)
        used["n"] += out is not None
        return out

    same = diff = err = norec = 0
    diffs = []
    with LocalCluster(n_workers=2) as cluster:
        with Client(cluster.scheduler):
            for label, build in ops():
                print(f"  ... {label}", flush=True)
                try:
                    fdask._frisky_compute_collections = spy
                    before = used["n"]
                    (rec,) = dask.compute(build())
                    took_records = used["n"] > before
                    (ref,) = dask.compute(build(), scheduler="synchronous")
                except Exception as e:  # noqa: BLE001
                    print(f"  ERR  {label:<28} {type(e).__name__}: {str(e)[:60]}")
                    err += 1
                    continue
                finally:
                    fdask._frisky_compute_collections = orig
                rec_a, ref_a = np.asarray(rec), np.asarray(ref)
                try:
                    if (
                        rec_a.dtype.kind in "OUS"
                        or ref_a.dtype.kind in "OUS"
                        or np.issubdtype(rec_a.dtype, np.datetime64)
                    ):
                        match = rec_a.shape == ref_a.shape and bool(np.all(rec_a == ref_a))
                    else:
                        match = rec_a.shape == ref_a.shape and np.allclose(rec_a, ref_a, equal_nan=True)
                except Exception as e:  # noqa: BLE001
                    match = False
                    print(f"       compare-fail {type(e).__name__}: {e}")
                tag = "REC" if took_records else "dask"
                if not took_records:
                    norec += 1
                if match:
                    same += 1
                else:
                    diff += 1
                    diffs.append((label, tag, rec_a.shape, ref_a.shape))
                print(f"  {'OK ' if match else 'DIFF'} [{tag:>4}] {label:<28} match={match}")

    fdask._frisky_compute_collections = orig
    print(f"\n{same} faithful, {diff} divergent, {err} errored; {norec} did not take records path")
    for label, tag, rs, fs in diffs:
        print(f"  DIFF {label} [{tag}] rec_shape={rs} ref_shape={fs}")


if __name__ == "__main__":
    main()
