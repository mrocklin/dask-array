"""Differential correctness: for a large, diverse batch of ops, compute each via
the Frisky records path AND via forced stock-dask (records path disabled), and
assert the two results are identical.

This is the strongest signal for the GraphRecordsLayer auto-fallback: it checks
the records path is *faithful to dask itself*, not to numpy — so approximate ops
(percentile, multi-chunk reductions) don't produce false mismatches, and many
diverse ops can be thrown at it to flush out any silent divergence.

    PYTHONPATH=/Users/mrocklin/workspace/dask-array MATURIN_IMPORT_HOOK_ENABLED=0 \
      /Users/mrocklin/workspace/frisky/.venv/bin/python bench/diff_records.py
"""

import numpy as np

import dask
import dask_array as da
import frisky.dask as fdask
from frisky import Client, LocalCluster


def ops():
    """Each is a 0-arg builder, evaluated fresh for both paths (distinct names)."""
    a = np.arange(60, dtype="f8").reshape(6, 10)
    sq = np.arange(49, dtype="f8").reshape(7, 7) + 1.0
    v = np.arange(24, dtype="f8") - 8.0
    fa = lambda c=(3, 5): da.from_array(a, chunks=c)
    fsq = lambda c=(3, 4): da.from_array(sq, chunks=c)
    fv = lambda c=5: da.from_array(v, chunks=c)

    # indexing — fancy / boolean / mixed / negative / repeated
    yield "take", lambda: fa()[[0, 2, 4, 1]]
    yield "take axis1", lambda: fa()[:, [9, 0, 5, 5]]
    yield "take neg", lambda: fa()[[-1, -2, 0]]
    yield "take 2d both", lambda: fa()[[0, 2], :][:, [1, 3]]
    yield "bool rows", lambda: fa()[fa()[:, 0] > 10]
    yield "bool full mask", lambda: fv()[fv() > 0]
    yield "vindex", lambda: fa().vindex[[0, 3, 5], [1, 9, 4]]
    yield "slice+fancy", lambda: fa()[1:5][[0, 2]]
    yield "fancy then reduce", lambda: fa()[[0, 2, 4]].mean(axis=1)
    yield "neg step + fancy", lambda: fa()[::-1][[0, 1]]

    # setitem
    yield "setitem scalar", lambda: _set(fa(), (1, 2), -5.0)
    yield "setitem row", lambda: _set(fa(), (0,), 0.0)
    yield "setitem slice", lambda: _set(fa(), (slice(2, 4), slice(0, 3)), 7.0)
    yield "setitem bool", lambda: _set(fa(), fa() > 30, 0.0)

    # concatenate / stack on unknown (nan) chunks — pure block-index alias; the
    # layer needs only block counts, which are known even when sizes are NaN.
    # `_unk` keeps every element but marks chunk sizes unknown (boolean index).
    _unk = lambda x: x[x.reshape(-1)[: x.shape[0]] >= -1e18] if x.ndim == 1 else x[x[:, 0] >= -1e18]
    yield "concat nan-chunks ax0", lambda: da.concatenate([_unk(fa()), _unk(fa())], axis=0)
    yield "concat nan-chunks 3way", lambda: da.concatenate([_unk(fa()), _unk(fa()), _unk(fa())], axis=0)
    yield (
        "concat nan-chunks ax1",
        lambda: da.concatenate([_unk(fa()), _unk(fa())], axis=1, allow_unknown_chunksizes=True),
    )
    yield "stack nan-chunks", lambda: da.stack([_unk(fv()), _unk(fv())])

    # from_array with a pushed-in _region (deferred slice into the source ndarray)
    yield "fromarray region 2d", lambda: fa()[1:5, 2:8]
    yield "fromarray region single", lambda: da.from_array(a, chunks=-1)[2:5, 3:9]
    yield "fromarray region int", lambda: fa()[1:5, 3]
    yield "fromarray region+reduce", lambda: fa()[1:5, 2:8].sum()
    yield "fromarray region+add", lambda: fa()[1:5, 2:8] + 1.0

    # blockwise with new_axes (map_blocks adding a dim) — output dim absent from
    # every input; native BlockwiseLayer iterates it, inputs ignore its coord.
    yield (
        "map_blocks new_axis",
        lambda: fv().map_blocks(
            lambda b: b[:, None] * np.ones(3), new_axis=1, chunks=(fv().chunks[0], (3,)), dtype="f8"
        ),
    )
    yield (
        "blockwise new_axes multi",
        lambda: da.blockwise(
            lambda b: np.broadcast_to(b[:, None], (b.shape[0], 10)).copy(),
            "az",
            fv(),
            "a",
            new_axes={"z": (5, 5)},
            dtype="f8",
        ),
    )

    # creation-ish / structural
    yield "diagonal", lambda: da.diagonal(fsq())
    yield "diag offset", lambda: da.diagonal(fsq(), offset=1)
    yield "pad", lambda: da.pad(fa(), 2, mode="constant")
    yield "pad reflect", lambda: da.pad(fa(), 1, mode="reflect")
    yield "triu", lambda: da.triu(fsq())
    yield "tril k1", lambda: da.tril(fsq(), k=1)
    yield "flip", lambda: da.flip(fa(), axis=1)
    yield "roll", lambda: da.roll(fa(), 3, axis=1)
    yield "repeat", lambda: da.repeat(fv(), 3)
    yield "tile", lambda: da.tile(fv(), 2)

    # counting / set
    yield "unique", lambda: da.unique(fv())
    yield "bincount", lambda: da.bincount((fv() + 8).astype("i8"))
    yield "histogram", lambda: da.histogram(fv(), bins=5, range=(-8, 16))[0]
    yield "isin", lambda: da.isin(fa(), np.array([1.0, 7.0, 42.0]))
    yield "nonzero", lambda: fv().nonzero()[0]
    yield "argwhere", lambda: da.argwhere(fa() > 25)
    yield "topk", lambda: da.topk(fv(), 4)
    yield "digitize", lambda: da.digitize(fv(), bins=[-4, 0, 4, 8])

    # overlap / stencil
    yield "map_overlap +1", lambda: fa().map_overlap(lambda b: b + 1, depth=1, boundary="none")
    yield "map_overlap reflect", lambda: fa().map_overlap(lambda b: b * 2, depth=1, boundary="reflect")
    yield "map_overlap 1d", lambda: fv().map_overlap(lambda b: b - 1, depth=2, boundary="periodic")

    # statistical
    yield "percentile", lambda: da.percentile(fv(), [10, 50, 90])
    yield "apply_along cumsum", lambda: da.apply_along_axis(np.cumsum, 0, fa())
    yield "cov", lambda: da.cov(fa())
    yield "corrcoef", lambda: da.corrcoef(fa())
    yield "gradient", lambda: da.gradient(fv())

    # cumulative / arg
    yield "cumsum", lambda: fa().cumsum(axis=0)
    yield "cumprod", lambda: (fa() / 10 + 1).cumprod(axis=1)
    yield "argmin axis", lambda: fa().argmin(axis=0)
    yield "argmax ravel", lambda: fa().argmax()

    # compositions tail<->covered
    yield "take+rechunk+sum", lambda: fa()[[0, 2, 4]].rechunk((1, 10)).sum()
    yield "diagonal+sqrt", lambda: da.sqrt(da.diagonal(fsq()))
    yield "unique+add+sum", lambda: (da.unique(fv()) + 1).sum()
    yield "overlap+slice+mean", lambda: fa().map_overlap(lambda b: b + 1, depth=1, boundary="none")[1:5].mean()

    # FusedBlockwise: an op-chain fused into one task/block (what x.compute() and
    # x.optimize() produce). `.optimize()` forces the fusion so the records path
    # sees the FusedBlockwise node; the synchronous reference uses its legacy
    # `to_dask_graph` form. The embedded subgraph (a dict of fused-away Tasks) must
    # ship and run on the worker byte-faithfully, with only the source blocks as
    # real deps.
    fb_ = lambda c=(3, 5): da.from_array(a + 100, chunks=c)
    yield "fused elemwise chain", lambda: da.sqrt(fa() * 2 + 1).optimize()
    yield "fused two-input", lambda: (fa() * fb_() + 1).optimize()
    yield "fused deep", lambda: da.exp(da.sqrt(fa() * 2 + 1) - fa() / 3).optimize()
    yield "fused 1d chain", lambda: (da.sqrt(da.abs(fv()) + 1) * 2).optimize()
    yield "fused irregular chunks", lambda: da.sqrt(da.from_array(a, chunks=((2, 1, 3), (4, 6))) * 2 + 1).optimize()
    yield "fused transpose chain", lambda: (fa().T * 2 + 1).optimize()
    yield "fused where", lambda: (da.where(fa() > 20, fa() * 2, 0.0) + 1).optimize()
    yield "fused chain -> reduction", lambda: da.sqrt(fa() * 2 + 1).sum().optimize()
    yield "fused slice -> chain", lambda: da.sqrt(fa()[1:5, 2:9] * 2 + 1).optimize()
    yield "fused chain -> rechunk", lambda: ((fa() * 2 + 1).rechunk((2, 5))).optimize()


def _set(x, idx, val):
    x[idx] = val
    return x


def main():
    orig = fdask._frisky_compute_collections
    used = {"n": 0}

    def spy(client, collections):
        out = orig(client, collections)
        used["n"] += out is not None
        return out

    same = diff = err = norec = 0
    with LocalCluster(n_workers=2) as cluster:
        with Client(cluster.scheduler):
            for label, build in ops():
                print(f"  ... {label}", flush=True)
                try:
                    fdask._frisky_compute_collections = spy
                    before = used["n"]
                    (rec,) = dask.compute(build())
                    took_records = used["n"] > before
                    # Reference: plain dask synchronous (no Frisky) — the true dask
                    # semantics, bypassing the patch and Frisky's legacy get() path.
                    (ref,) = dask.compute(build(), scheduler="synchronous")
                except Exception as e:  # noqa: BLE001
                    print(f"  ERR  {label:<22} {type(e).__name__}: {str(e)[:50]}")
                    err += 1
                    continue
                finally:
                    fdask._frisky_compute_collections = orig
                rec_a, ref_a = np.asarray(rec), np.asarray(ref)
                match = rec_a.shape == ref_a.shape and np.allclose(rec_a, ref_a, equal_nan=True)
                tag = "REC" if took_records else "dask"
                if not took_records:
                    norec += 1
                if match:
                    same += 1
                else:
                    diff += 1
                print(f"  {'OK ' if match else 'DIFF'} [{tag:>4}] {label:<22} match={match}")

    fdask._frisky_compute_collections = orig
    print(f"\n{same} faithful, {diff} divergent, {err} errored; {norec} did not take records path")


if __name__ == "__main__":
    main()
