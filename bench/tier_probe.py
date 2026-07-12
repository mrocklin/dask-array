"""Tier probe: task-weighted breakdown of how a collection's graph is BUILT.

For graph-build cost at scale the question isn't "which op lacks a Rust layer"
(that's `tail_probe.py`), it's "how many *tasks* get emitted the cheap way?".
The classifier is the production one — ``classify`` in
``dask_array/_frisky/inventory.py`` (mirrors ``_walk_record_chunks``, covered
by ``dask_array/tests/test_frisky_inventory.py``) — which sorts every lowered
layer into four tiers of ascending client-side cost:

  - ``binary``        Rust ``to_records_chunk`` -> one ``bytes`` blob for the whole
                      layer. O(1) Python objects regardless of block count. The goal.
  - ``native_tuples`` Rust layer exists but declines the binary chunk, so
                      ``to_task_records`` builds N Python tuples in Rust. O(tasks).
  - ``adapter``       no native layer -> ``GraphRecordsLayer`` runs the expr's
                      Python ``_layer()`` and translates it. O(tasks), most work.
  - ``fallback``      ``_check_frisky_supported`` rejects the whole graph -> stock
                      dask builds every task in Python. Worst case.

Cluster-free and fast: it lowers, walks, and attempts record *generation* (it
never computes). Weighs each layer by its block count, so a single 1M-block
Blockwise on ``native_tuples`` outranks 40 one-block tail nodes on ``adapter``.
Drop your own expressions into ``corpus()``.

    uv run --extra test python bench/tier_probe.py
"""

from collections import Counter

import numpy as np

import dask_array as da
from dask_array._frisky.inventory import BINARY, TIERS, classify


def corpus():
    """Representative ops at realistic block counts (walk is O(nodes), so big
    grids are free). Chunk counts, not sizes, drive the task weights."""

    def A(shape, chunks):
        return da.ones(shape, chunks=chunks)

    def F(shape, chunks):
        base = np.arange(int(np.prod(shape)), dtype="f8").reshape(shape)
        return da.from_array(base, chunks=chunks)

    big = (2000, 2000)
    bc = (20, 20)  # 100x100 = 10_000 blocks

    # --- everyday, high-block-count (where build cost actually hurts) ---
    yield "elemwise chain (x+y*2-1)/3", lambda: (A(big, bc) + A(big, bc) * 2 - 1) / 3
    yield "ufunc chain exp(log(x+1))", lambda: da.exp(da.log(A(big, bc) + 1))
    yield "where/clip", lambda: da.where(A(big, bc) > 0.5, A(big, bc), 0).clip(0, 1)
    yield "fused elemwise + reduce", lambda: ((A(big, bc) + A(big, bc)) * 3).sum(axis=0)
    yield "sum axis0", lambda: A(big, bc).sum(axis=0)
    yield "mean all", lambda: A(big, bc).mean()
    yield "std", lambda: A(big, bc).std()
    yield "argmax axis1", lambda: A(big, bc).argmax(axis=1)
    yield "cumsum axis0", lambda: A(big, bc).cumsum(axis=0)
    yield "matmul", lambda: A(big, bc) @ A(big, bc)
    yield "tensordot", lambda: da.tensordot(A(big, bc), A(big, bc), axes=1)
    yield "transpose+add", lambda: A(big, bc).T + A(big, bc)
    yield "rechunk", lambda: A(big, bc).rechunk((40, 10))
    yield "rechunk then sum", lambda: A(big, bc).rechunk((40, 10)).sum(axis=1)
    yield "slice", lambda: A(big, bc)[100:1900, ::2]
    yield "stack 4", lambda: da.stack([A(big, bc)] * 4)
    yield "concatenate 3", lambda: da.concatenate([F(big, bc), F(big, bc), F(big, bc)])
    yield "reshape", lambda: A(big, bc).reshape(4_000_000)
    yield "broadcast add row", lambda: A(big, bc) + A((2000,), (20,))
    yield "roll", lambda: da.roll(A(big, bc), 3, axis=0)
    yield "pad", lambda: da.pad(A(big, bc), 2, mode="constant")
    yield "map_overlap", lambda: A(big, bc).map_overlap(lambda b: b + 1, depth=1, boundary="none")
    yield "map_blocks", lambda: A(big, bc).map_blocks(lambda b: b + 1)
    yield "coarsen", lambda: da.coarsen(np.sum, A(big, bc), {0: 4, 1: 4})

    # --- known-shape tail (adapter today) ---
    yield "vindex", lambda: F(big, bc).vindex[[0, 5, 9], [1, 3, 7]]
    yield "setitem", lambda: _setitem(F(big, bc))
    yield "histogram", lambda: da.histogram(F(big, bc).ravel(), bins=10, range=(0, 1))[0]
    yield "diagonal", lambda: da.diagonal(F(big, bc))
    yield "take", lambda: F(big, bc)[np.arange(0, 2000, 3)]

    # --- unknown-shape (stock-dask fallback today) ---
    yield "bool mask", lambda: _flat(F(big, bc))[_flat(F(big, bc)) > 0.5]
    yield "unique", lambda: da.unique(F(big, bc))


def _setitem(x):
    x[0:100] = 0.0
    return x


def _flat(x):
    return x.ravel()


def warn_if_native_build_stale():
    """When ``dask_array._rust`` is missing or its build generation mismatches
    the source, every ``_frisky_layer()`` raises ImportError and the whole
    corpus silently classifies as ``adapter`` — a 0% binary readout that looks
    like a regression but is just a stale build. Surface it up front."""
    try:
        import dask_array._frisky.base  # noqa: F401  (import runs the generation check)
    except ImportError as exc:
        print("!" * 76)
        print("! WARNING: native extension unavailable — every layer below will read as")
        print("! `adapter`; the binary% numbers are MEANINGLESS until you rebuild:")
        print(f"!   {exc}")
        print("!   fix: uv run --extra test maturin develop")
        print("!" * 76)


def main():
    warn_if_native_build_stale()
    grand = Counter()
    grand_culprits = Counter()
    print(f"  {'op':<28} {'tasks':>9}  binary  ntuple  adapt  fallbk   status")
    for label, build in corpus():
        try:
            coll = build()
            coll = coll[0] if isinstance(coll, tuple) else coll
            r = classify(coll)
        except Exception as ex:  # noqa: BLE001
            print(f"  {label:<28} ERR {type(ex).__name__}: {str(ex)[:40]}")
            continue
        t = r["tiers"]
        total = sum(t.values()) or 1
        grand.update(t)
        grand_culprits.update(r["culprits"])
        pct = lambda k: f"{100 * t.get(k, 0) / total:5.0f}%"
        rejected = next(iter(r["rejected"]), None)
        status = rejected or ("binary" if t.get(BINARY, 0) == total else "mixed")
        print(f"  {label:<28} {total:>9,}  {'  '.join(pct(k) for k in TIERS)}   {status[:40]}")

    gtotal = sum(grand.values()) or 1
    print(f"\n  === corpus rollup: {gtotal:,} tasks ===")
    for tier in TIERS:
        n = grand.get(tier, 0)
        print(f"    {tier:<14} {n:>12,}  {100 * n / gtotal:5.1f}%")
    print("\n  top non-binary layers by TASK volume (port these to move the needle):")
    for (tier, cls, reason), n in grand_culprits.most_common(15):
        print(f"    {n:>12,}  {tier:<13} {cls:<24} {reason}")


if __name__ == "__main__":
    main()
