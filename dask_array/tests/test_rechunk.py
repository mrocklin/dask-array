"""Tests for rechunk graph structure: per-block degree (fan-in and fan-out)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

import dask

import dask_array as da
from dask_array._test_utils import assert_eq


def degrees(arr):
    """(max fan-in, max fan-out) over all tasks in the optimized graph.

    Fan-in is a task's dependency count; fan-out is how many tasks depend on it.
    """
    graph = arr.optimize().__dask_graph__()
    out = defaultdict(int)
    max_in = 0
    for value in graph.values():
        deps = getattr(value, "dependencies", ())
        max_in = max(max_in, len(deps))
        for dep in deps:
            out[dep] += 1
    return max_in, max(out.values(), default=0)


def test_rechunk_bounds_degree_transpose():
    """A transpose-style rechunk is symmetric: it has huge fan-in AND fan-out.

    Every output column needs every input row (fan-in) and every input row feeds
    every output column (fan-out); both must be bounded.
    """
    x = da.random.random((20000, 20000), chunks=(1, -1))
    max_in, max_out = degrees(x.rechunk((-1, 1)))
    assert max_in <= 100
    assert max_out <= 100


def test_rechunk_bounds_fanin_pure_merge():
    """Many tiny chunks -> one chunk is the classic high-fan-in hub node."""
    max_in, _ = degrees(da.ones(20000, chunks=1).rechunk(-1))
    assert max_in <= 100


def test_rechunk_bounds_fanout_pure_split():
    """One chunk -> many tiny chunks is the dual: a single high-fan-out source."""
    _, max_out = degrees(da.ones(20000, chunks=-1).rechunk(1))
    assert max_out <= 100


def test_rechunk_bounds_degree_scales_with_chunk_size():
    """A larger block-size config must not reintroduce huge fan-in or fan-out."""
    x = da.random.random((20000, 20000), chunks=(1, -1))
    with dask.config.set({"array.chunk-size": "512MiB"}):
        max_in, max_out = degrees(x.rechunk((-1, 1)))
        assert max_in <= 100
        assert max_out <= 100


@pytest.mark.parametrize(
    "old_chunks,new_chunks",
    [
        ((1, -1), (-1, 1)),  # transpose-style, both axes flip
        (1, -1),  # 1-D full merge
        (-1, 1),  # 1-D full split
        ((2, -1), (-1, 2)),
        ((1, -1, -1), (-1, 1, -1)),  # 3-D
    ],
)
def test_rechunk_degree_values_correct(old_chunks, new_chunks):
    """Subdividing high-degree steps must not change the result."""
    shape = {1: (600,), 2: (120, 80), 3: (60, 40, 8)}[len(old_chunks) if isinstance(old_chunks, tuple) else 1]
    a = np.arange(int(np.prod(shape))).reshape(shape).astype(float)
    x = da.from_array(a, chunks=old_chunks).rechunk(new_chunks)
    assert_eq(x, a)


def test_rechunk_degree_limit_configurable():
    """A tighter degree-limit yields lower fan-in and fan-out."""
    with dask.config.set({"array.rechunk.degree-limit": 30}):
        assert degrees(da.ones(20000, chunks=1).rechunk(-1))[0] <= 30  # fan-in
        assert degrees(da.ones(20000, chunks=-1).rechunk(1))[1] <= 30  # fan-out


def test_rechunk_degree_no_duplicate_steps():
    """Subdivision must never emit a no-op (identity) intermediate step.

    When degree comes from misaligned splits rather than merges, every axis goes
    straight to the target; the pass must collapse that to a single step instead
    of a redundant rechunk layer.
    """
    from dask_array._rechunk import plan_rechunk

    old = ((4, 4),) * 4
    new = ((3, 2, 3),) * 4  # net split on every axis, but boundary-misaligned
    with dask.config.set({"array.rechunk.degree-limit": 10}):
        steps = plan_rechunk(old, new, 8)
    assert all(steps[i] != steps[i + 1] for i in range(len(steps) - 1))
    assert steps[-1] == new


def test_rechunk_below_limit_unchanged():
    """A rechunk whose natural degree is below the limit gets no extra steps."""
    x = da.ones((200, 200), chunks=(20, 20))
    y = x.rechunk((40, 40))  # degree ~4, far below the limit

    keys_default = set(y.optimize().__dask_graph__())
    with dask.config.set({"array.rechunk.degree-limit": 10**9}):
        keys_unbounded = set(y.optimize().__dask_graph__())

    assert keys_default == keys_unbounded


def test_rechunk_balance_preserves_empty_chunk_layouts():
    x = da.from_array(np.empty((0, 0)), chunks=((0,), (0,)))

    assert x.rechunk(((0,), (0,)), balance=True).chunks == ((0,), (0,))

    y = x.rechunk(((0, 0), (0,)), balance=True)
    assert y.chunks == ((0, 0), (0,))
    assert_eq(y, np.empty((0, 0)))

    z = da.from_array(np.arange(1), chunks=((0, 1),)).rechunk(((0, 1),), balance=True)
    assert z.chunks == ((0, 1),)
    assert_eq(z, np.arange(1))


def test_rechunk_split_tasks_copy_small_selections():
    """Split tasks must use chunk.getitem (copy-if-small), not operator.getitem.

    A raw-getitem split is a numpy view that keeps its whole parent block
    refcount-alive after the parent is released; lingering splits on
    rechunk-heavy graphs pin "released" parents as untracked worker RSS.
    """
    from dask_array._chunk import getitem as chunk_getitem

    x = da.ones((40, 40), chunks=(40, 40))
    y = x.rechunk((40, 4))

    # Python-materialized graph path (_compute_rechunk).
    graph = dict(y.optimize().__dask_graph__())
    split_tasks = [task for key, task in graph.items() if "rechunk-split" in str(key) and hasattr(task, "func")]
    assert split_tasks, "expected rechunk-split tasks in the graph"
    parent = np.ones((40, 40))
    for task in split_tasks:
        assert task.func is chunk_getitem
    piece = split_tasks[0].func(parent, (slice(0, 40), slice(0, 4)))
    assert piece.flags.owndata, "small split selection must be a copy"

    # Rust records path (TasksRechunk._frisky_layer -> RechunkLayer).
    import dask_array._frisky.rechunk as rechunk_mod

    captured = {}
    real_layer = rechunk_mod.RechunkLayer

    class RecordingLayer:
        def __init__(self, getitem, concatenate3, steps, dtype):
            captured["getitem"] = getitem
            self._wrapped = real_layer(getitem, concatenate3, steps, dtype)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    rechunk_mod.RechunkLayer = RecordingLayer
    try:
        expr = y.optimize().expr
        rechunk_exprs = [e for e in expr.walk() if type(e).__name__ == "TasksRechunk"]
        assert rechunk_exprs, "expected a TasksRechunk node"
        rechunk_exprs[0]._frisky_layer()
    finally:
        rechunk_mod.RechunkLayer = real_layer
    assert captured["getitem"] is chunk_getitem
