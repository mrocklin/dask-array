"""Tail probe: for a broad batch of *specialized* dask-array ops, walk the
lowered expr tree and report which expr classes still lack a `_frisky_layer`
(or whose `_frisky_layer()` raises). This is the data-driven priority list for
the remaining tail — cluster-free and fast (it only lowers + walks; it does not
compute).

    uv run --extra test python bench/tail_probe.py
"""

from collections import Counter

import numpy as np

import dask_array as da


def cases():
    a = np.arange(48, dtype="f8").reshape(6, 8)
    sq = np.arange(36, dtype="f8").reshape(6, 6)
    v = np.arange(12, dtype="f8")
    fa = lambda c=(3, 4): da.from_array(a, chunks=c)
    fsq = lambda c=(3, 3): da.from_array(sq, chunks=c)
    fv = lambda c=4: da.from_array(v, chunks=c)

    # --- fancy / boolean indexing ---
    yield ("take list-index", lambda: fa()[[0, 2, 4]])
    yield ("take np-index axis1", lambda: fa()[:, np.array([1, 3, 5])])
    yield ("bool mask 1d", lambda: fv()[fv() > 5])
    yield ("vindex", lambda: fa().vindex[[0, 2], [1, 3]])
    yield ("setitem scalar", lambda: _setitem(fa(), (0, 0), 99.0))
    yield ("setitem slice", lambda: _setitem(fa(), (slice(0, 2),), 0.0))

    # --- structural rearrange ---
    yield ("pad", lambda: da.pad(fa(), 1, mode="constant"))
    yield ("roll", lambda: da.roll(fa(), 2, axis=0))
    yield ("flip", lambda: da.flip(fa(), axis=0))
    yield ("repeat", lambda: da.repeat(fv(), 2))
    yield ("tile", lambda: da.tile(fv(), 2))
    yield ("insert", lambda: da.insert(fv(), 1, 99.0))
    yield ("roll 2d both", lambda: da.roll(fa(), (1, 2), axis=(0, 1)))

    # --- overlap family ---
    yield ("map_overlap", lambda: fa().map_overlap(lambda x: x, depth=1, boundary="none"))

    # --- counting / set ops ---
    yield ("histogram", lambda: da.histogram(fv(), bins=4, range=(0, 12))[0])
    yield ("unique", lambda: da.unique(fv()))
    yield ("bincount", lambda: da.bincount(fv().astype("i8")))
    yield ("isin", lambda: da.isin(fv(), np.array([1.0, 2.0])))
    yield ("nonzero", lambda: fv().nonzero()[0])
    yield ("argwhere", lambda: da.argwhere(fv() > 3))

    # --- diagonal variants ---
    yield ("diagonal", lambda: da.diagonal(fsq()))
    yield ("diag k=1", lambda: da.diag(fv()))

    # --- linalg (multi-stage) ---
    yield ("qr", lambda: da.linalg.qr(fa())[0])
    yield ("svd", lambda: da.linalg.svd(fa())[0])
    yield ("cholesky", lambda: da.linalg.cholesky(fsq() @ fsq().T))
    yield ("solve", lambda: da.linalg.solve(fsq(), fv()[:6]))
    yield ("lstsq", lambda: da.linalg.lstsq(fa(), fv()[:6])[0])
    yield ("inv", lambda: da.linalg.inv(fsq()))
    yield ("norm", lambda: da.linalg.norm(fa()))

    # --- stats ---
    yield ("nansum", lambda: da.nansum(fa(), axis=0))
    yield ("nanmean", lambda: da.nanmean(fa()))
    yield ("percentile", lambda: da.percentile(fv(), [50]))
    yield ("quantile axis", lambda: da.quantile(fa(), 0.5, axis=0))
    yield ("cov", lambda: da.cov(fa()))
    yield ("corrcoef", lambda: da.corrcoef(fa()))
    yield ("digitize", lambda: da.digitize(fv(), bins=[2, 4, 6]))
    yield ("searchsorted", lambda: da.searchsorted(fv(), np.array([3.0])))
    yield ("gradient", lambda: da.gradient(fv()))
    yield ("ptp", lambda: fa().ptp(axis=0))

    # --- combination / building ---
    yield ("block", lambda: da.block([[fa(), fa()]]))
    yield ("outer", lambda: da.outer(fv(), fv()))
    yield ("einsum", lambda: da.einsum("ij->i", fa()))
    yield ("apply_along_axis", lambda: da.apply_along_axis(np.cumsum, 0, fa()))
    yield ("around", lambda: da.around(fa(), 1))
    yield ("sort", lambda: da.sort(fv()))
    yield ("topk", lambda: da.topk(fv(), 3))
    yield ("isclose", lambda: da.isclose(fa(), fa()))
    yield ("triu", lambda: da.triu(fsq()))
    yield ("tril", lambda: da.tril(fsq()))


def _setitem(x, idx, val):
    x[idx] = val
    return x


def walk(expr):
    """Return (uncovered_classes, total_nodes). uncovered = list of (cls, reason)."""
    try:
        e = expr.lower_completely()
    except Exception as ex:  # noqa: BLE001
        return [("<lower failed>", f"{type(ex).__name__}: {str(ex)[:50]}")], 0
    seen, stack = set(), [e]
    uncovered, total = [], 0
    while stack:
        n = stack.pop()
        name = getattr(n, "_name", id(n))
        if name in seen:
            continue
        seen.add(name)
        total += 1
        cls = type(n).__name__
        if not hasattr(n, "_frisky_layer"):
            uncovered.append((cls, "no _frisky_layer"))
        else:
            try:
                n._frisky_layer()
            except Exception as ex:  # noqa: BLE001
                uncovered.append((cls, f"raises {type(ex).__name__}: {str(ex)[:40]}"))
        try:
            stack.extend(n.dependencies())
        except Exception:  # noqa: BLE001
            pass
    return uncovered, total


def warn_if_native_build_stale():
    """When ``dask_array._rust`` is missing or its build generation mismatches
    the source, every ``_frisky_layer()`` raises ImportError and every op below
    reads as GAP — a coverage collapse that is really just a stale build.
    Surface it up front."""
    try:
        import dask_array._frisky.base  # noqa: F401  (import runs the generation check)
    except ImportError as exc:
        print("!" * 76)
        print("! WARNING: native extension unavailable — every op below will read as GAP")
        print("! (`raises ImportError`); coverage numbers are MEANINGLESS until you rebuild:")
        print(f"!   {exc}")
        print("!   fix: uv run --extra test maturin develop")
        print("!" * 76)


def main():
    warn_if_native_build_stale()
    fully, partial, errored = [], [], []
    culprit_classes = Counter()
    for label, build in cases():
        try:
            coll = build()
            # multi-output ops may return a tuple
            if isinstance(coll, tuple):
                coll = coll[0]
            uncovered, total = walk(coll.expr)
        except Exception as ex:  # noqa: BLE001
            errored.append((label, f"{type(ex).__name__}: {str(ex)[:55]}"))
            print(f"  ERR   {label:<22} {type(ex).__name__}: {str(ex)[:50]}")
            continue
        if not uncovered:
            fully.append(label)
            print(f"  COVERED  {label:<22} ({total} nodes)")
        else:
            partial.append((label, uncovered))
            for c, _ in uncovered:
                culprit_classes[c] += 1
            detail = "; ".join(f"{c} [{r}]" for c, r in sorted(set(uncovered)))
            print(f"  GAP   {label:<22} {detail}")

    print(f"\nfully covered (lowered tree): {len(fully)}/{len(fully) + len(partial) + len(errored)}")
    print(f"errored at construction: {len(errored)}")
    print("\nMost common uncovered expr classes (port these first):")
    for cls, n in culprit_classes.most_common():
        print(f"  {n:>3}  {cls}")


if __name__ == "__main__":
    main()
