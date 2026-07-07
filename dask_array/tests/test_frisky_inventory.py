from __future__ import annotations

import numpy as np

import dask_array as da
from dask_array._frisky.inventory import (
    BINARY,
    TIERS,
    classify,
    classify_node,
    python_groups,
)


def test_classify_conserves_tasks_and_valid_tiers():
    # Machinery contract (independent of which ops are binary *today* — that is
    # the moving target the tool measures): every tier is a known tier, and the
    # per-tier task counts sum to the graph's total block count.
    o = da.ones((8, 4), chunks=(4, 4))
    coll = (o * 2 + 1).optimize()
    result = classify(coll)

    assert set(result["tiers"]) <= set(TIERS)
    total = sum(result["tiers"].values())
    assert total > 0
    # culprit task counts never exceed the non-binary total.
    culprit_tasks = sum(result["culprits"].values())
    assert culprit_tasks == total - result["tiers"].get(BINARY, 0)

    # classify_node returns (tier, reason) for a concrete lowered node.
    node_tier, reason = classify_node(coll._lowered_expr)
    assert node_tier in TIERS
    assert isinstance(reason, str)


def test_classify_task_weighted_and_named():
    # A concat of from_delayed blocks with DISTINCT ndarray args lowers to one
    # FromMap that declines the binary chunk (the arg varies per block and isn't
    # binary-expressible, so it can be neither baked nor slotted), so classify
    # names it and weighs it by block count.
    from dask import delayed

    def load(a):
        return a

    leaf = da.concatenate(
        [da.from_delayed(delayed(load)(np.full((8, 4), i, dtype="f8")), shape=(8, 4), dtype="f8") for i in range(2)]
    )
    result = classify(leaf + 1.0)

    assert set(result["tiers"]) <= set(TIERS)
    assert result["nodes"] >= 1
    nonbinary = {cls for (_tier, cls, _reason) in result["culprits"]}
    assert "FromMap" in nonbinary
    assert all(n > 0 for n in result["tiers"].values())


def test_classify_shares_seen_across_collections():
    # One shared leaf across two collections is walked once (dedup by _name),
    # the way a real dask.compute(x, y) submission dedups its subgraph.
    x = da.from_array(np.ones((8, 4)), chunks=(4, 4))
    a = (x + 1).sum(axis=1)
    b = (x + 1).mean(axis=0)
    shared = classify([a, b])["nodes"]
    separate = classify([a])["nodes"] + classify([b])["nodes"]
    assert shared < separate


def test_python_groups_ranked_descending():
    from dask import delayed

    def load(a):
        return a

    leaf = da.concatenate(
        [da.from_delayed(delayed(load)(np.full((16, 4), i, dtype="f8")), shape=(16, 4), dtype="f8") for i in range(2)]
    )
    rows = python_groups([(leaf + 1).rechunk((8, 4)).sum(axis=1)])
    assert rows  # at least the FromMap leaf
    tasks = [n for _cls, _reason, n in rows]
    assert tasks == sorted(tasks, reverse=True)
    assert all(BINARY not in reason for _cls, reason, _n in rows)
