"""Reviewer adversarial probe — stricter than diff_records/diff_adversarial.

Differences from the existing sweeps:
  - exact dtype + exact value equality (not np.allclose), so a dtype/precision
    divergence in the Frisky path can't hide behind a tolerance;
  - asserts a Frisky path (expression or records — both spied, see ``_spy.py``)
    *actually fired* per op (a silent fallback to dask is reported as FELL,
    not counted as a pass) — a pass that fell back is not evidence the Frisky
    path is correct;
  - targets edge cases a final review must rule out: object/structured/datetime/
    bool/complex dtypes, masked arrays, empty/zero-size pieces, key-collision
    bait, dict-container nodes, large-fan reductions, persist round-trips.

Run:
  PYTHONPATH=$PWD MATURIN_IMPORT_HOOK_ENABLED=0 \
    /Users/mrocklin/workspace/frisky/.venv/bin/python bench/diff_review.py
"""

import numpy as np

import dask
import dask_array as da
from _spy import TAGS, frisky_spy
from frisky import Client, LocalCluster


def ops():
    a = np.arange(60, dtype="f8").reshape(6, 10)
    ai = np.arange(60, dtype="i8").reshape(6, 10)
    v = np.arange(24, dtype="f8") - 8.0
    fa = lambda c=(3, 5): da.from_array(a, chunks=c)
    fai = lambda c=(3, 5): da.from_array(ai, chunks=c)
    fv = lambda c=5: da.from_array(v, chunks=c)

    # --- dtype faithfulness: allclose would mask an int->float drift ---
    yield "int sum dtype", lambda: fai().sum(axis=0)
    yield "int cumsum dtype", lambda: fai().cumsum(axis=1)
    yield "int prod", lambda: fai()[:, :3].prod(axis=1)
    yield "bool any", lambda: (fai() > 30).any(axis=0)
    yield "bool all", lambda: (fai() > -1).all(axis=1)
    yield "argmin dtype", lambda: fai().argmin(axis=0)
    yield "astype f4", lambda: fa().astype("f4").sum(axis=0)
    yield "mean keeps f8", lambda: fai().mean(axis=0)

    # --- complex dtype ---
    yield "complex add", lambda: (da.from_array(a.astype("c16"), chunks=(3, 5)) + 1j).sum(axis=0)

    # --- datetime / timedelta ---
    dt = np.arange("2020-01-01", "2020-02-12", dtype="datetime64[D]")
    yield "datetime take", lambda: da.from_array(dt, chunks=10)[[5, 0, 30, 12]]
    yield "datetime max", lambda: da.from_array(dt, chunks=10).max()

    # --- object dtype through from_array + indexing (no copy semantics) ---
    o = np.array([{"i": i} for i in range(12)], dtype=object)
    yield "object take", lambda: da.from_array(o, chunks=4)[[11, 0, 7, 7, 3]]
    yield "object single block", lambda: da.from_array(o, chunks=12)[[2, 2]]

    # --- structured dtype ---
    st = np.zeros(10, dtype=[("x", "i8"), ("y", "f8")])
    st["x"] = np.arange(10)
    st["y"] = np.arange(10) * 1.5
    yield "structured take", lambda: da.from_array(st, chunks=4)[[9, 0, 5]]

    # --- masked array through from_array (the _frisky_layer accepts MaskedArray) ---
    m = np.ma.masked_array(np.arange(20, dtype="f8"), mask=[0, 1] * 10)
    yield "masked take", lambda: da.from_array(m, chunks=7)[[19, 0, 11]]
    yield "masked single block", lambda: da.from_array(m, chunks=20)[[1, 3]]

    # --- empty / zero-size ---
    yield "empty slice", lambda: fa()[2:2]
    yield "empty bool mask", lambda: fv()[fv() > 1e9]
    yield "zero-len take", lambda: fa()[[]]

    # --- single-element / scalar reductions ---
    yield "full reduce sum", lambda: fai().sum()
    yield "full reduce max", lambda: fa().max()
    yield "nanmax with nan", lambda: da.nanmax(da.from_array(_with_nan(a), chunks=(3, 5)))

    # --- key-collision bait: a name that could clash with "<parent>-subN" ---
    # diagonal/concatenate3 synthesize "<key>-subN"; hammer nested ops to provoke
    yield (
        "nested overlap concat",
        lambda: (
            fa().map_overlap(lambda b: b + 1, depth=1, boundary="reflect").rechunk((2, 5))
            + fa().map_overlap(lambda b: b * 2, depth=1, boundary="none")
        ),
    )

    # --- bincount (no weights: weights+minlength is a pre-existing dask-array
    #     bug that fails identically in plain dask, so not a records-path probe) ---
    yield "bincount", lambda: da.bincount((fv() + 8).astype("i8"))

    # --- large fan-in reduction (tree aggregate, split_every) ---
    big = np.arange(1000, dtype="f8")
    yield "tree sum split2", lambda: da.from_array(big, chunks=20).sum(split_every=2)
    yield "tree min split3", lambda: da.from_array(big, chunks=10).min(split_every=3)

    # --- broadcasting / blockwise with two distinct inputs ---
    yield "two-input add", lambda: fa() + fa()[::-1]
    yield "outer broadcast", lambda: fv()[:, None] * fv()[None, :]

    # --- transpose / reshape combos ---
    yield "transpose sum", lambda: fa().T.sum(axis=0)
    # reshape must merge/split evenly (dask limitation): ravel then split back
    yield "reshape roundtrip", lambda: fa().reshape(60).reshape(6, 10).sum()
    yield "ravel", lambda: fa().ravel()


def _with_nan(a):
    a = a.copy()
    a[0, 0] = np.nan
    return a


def _exact(rec, ref):
    """Return (ok, why). Exact: same dtype-kind-or-better, same shape, same values."""
    rec_a, ref_a = np.asarray(rec), np.asarray(ref)
    if rec_a.shape != ref_a.shape:
        return False, f"shape {rec_a.shape} != {ref_a.shape}"
    if rec_a.dtype != ref_a.dtype:
        return False, f"dtype {rec_a.dtype} != {ref_a.dtype}"
    # masked: compare filled + mask
    if np.ma.isMaskedArray(ref_a) or np.ma.isMaskedArray(rec_a):
        rm = np.ma.getmaskarray(np.ma.asarray(rec_a))
        fm = np.ma.getmaskarray(np.ma.asarray(ref_a))
        if not np.array_equal(rm, fm):
            return False, "mask differs"
        return (bool(np.all(np.ma.asarray(rec_a).filled(0) == np.ma.asarray(ref_a).filled(0))), "values")
    if (
        rec_a.dtype.kind in "OUSV"
        or np.issubdtype(rec_a.dtype, np.datetime64)
        or np.issubdtype(rec_a.dtype, np.timedelta64)
    ):
        return bool(rec_a.shape == ref_a.shape and np.all(rec_a == ref_a)), "values(obj)"
    if rec_a.dtype.kind in "fc":
        return bool(np.array_equal(rec_a, ref_a, equal_nan=True)), "values(exact-float)"
    return bool(np.array_equal(rec_a, ref_a)), "values"


def main():
    ok = bad = err = fellback = 0
    problems = []
    with frisky_spy() as spy:
        with LocalCluster(n_workers=2) as cluster:
            with Client(cluster.scheduler):
                for label, build in ops():
                    try:
                        before = spy.snapshot()
                        (rec,) = dask.compute(build())
                        path = spy.path_since(before)
                        (ref,) = dask.compute(build(), scheduler="synchronous")
                    except Exception as e:  # noqa: BLE001
                        print(f"  ERR  {label:<24} {type(e).__name__}: {str(e)[:70]}")
                        err += 1
                        problems.append((label, "ERR", str(e)[:120]))
                        continue
                    match, why = _exact(rec, ref)
                    tag = TAGS[path] if path != "none" else "FELL"
                    if path == "none":
                        fellback += 1
                        problems.append((label, "FELL", "fell back to dask (Frisky path NOT exercised)"))
                    if match:
                        ok += 1
                    else:
                        bad += 1
                        problems.append((label, tag, why))
                    flag = "OK  " if match else "BAD "
                    print(f"  {flag} [{tag}] {label:<24} {why if not match else ''}")

    print(
        f"\n{ok} faithful, {bad} divergent, {err} errored; "
        f"engaged: {spy.counts['expression']} expression, {spy.counts['records']} records; "
        f"{fellback} fell back to dask (Frisky path NOT exercised)"
    )
    for label, tag, why in problems:
        print(f"  >> {tag:<4} {label}: {why}")


if __name__ == "__main__":
    main()
