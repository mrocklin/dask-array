"""Tests for rechunk graph structure, especially fan-in (per-node dependency count)."""

from __future__ import annotations

import numpy as np
import pytest

import dask

import dask_array as da
from dask_array._test_utils import assert_eq


def max_fanin(arr):
    """Largest number of dependencies of any single task in the optimized graph."""
    graph = arr.optimize().__dask_graph__()
    return max(
        (len(v.dependencies) for v in graph.values() if hasattr(v, "dependencies")),
        default=0,
    )


def test_rechunk_bounds_fanin_2d():
    """A transpose-style rechunk must not produce nodes with thousands of deps.

    Every output column needs data from every input row, so a naive plan builds
    a few hub nodes each depending on ~all input blocks.
    """
    x = da.random.random((20000, 20000), chunks=(1, -1))
    y = x.rechunk((-1, 1))
    assert max_fanin(y) <= 100


def test_rechunk_bounds_fanin_1d():
    """The 1-D full merge (many tiny chunks -> one chunk) is the classic hub node."""
    x = da.ones(20000, chunks=1)
    assert max_fanin(x.rechunk(-1)) <= 100


def test_rechunk_bounds_fanin_scales_with_chunk_size():
    """A larger block-size config must not reintroduce huge fan-in."""
    x = da.random.random((20000, 20000), chunks=(1, -1))
    with dask.config.set({"array.chunk-size": "512MiB"}):
        y = x.rechunk((-1, 1))
        assert max_fanin(y) <= 100


@pytest.mark.parametrize(
    "old_chunks,new_chunks",
    [
        ((1, -1), (-1, 1)),  # transpose-style, both axes flip
        (1, -1),  # 1-D full merge
        ((2, -1), (-1, 2)),
    ],
)
def test_rechunk_fanin_values_correct(old_chunks, new_chunks):
    """Subdividing high-fan-in steps must not change the result."""
    shape = (120, 80) if isinstance(old_chunks, tuple) else (600,)
    a = np.arange(int(np.prod(shape))).reshape(shape).astype(float)
    x = da.from_array(a, chunks=old_chunks).rechunk(new_chunks)
    assert_eq(x, a)


def test_rechunk_fanin_limit_configurable():
    """A tighter fanin-limit yields lower fan-in."""
    x = da.ones(20000, chunks=1)
    with dask.config.set({"array.rechunk.fanin-limit": 30}):
        assert max_fanin(x.rechunk(-1)) <= 30


def test_rechunk_fanin_no_duplicate_steps():
    """Subdivision must never emit a no-op (identity) intermediate step.

    When fan-in comes from misaligned splits rather than merges, every axis goes
    straight to the target; the pass must collapse that to a single step instead
    of a redundant rechunk layer.
    """
    from dask_array._rechunk import plan_rechunk

    old = ((4, 4),) * 4
    new = ((3, 2, 3),) * 4  # net split on every axis, but boundary-misaligned
    with dask.config.set({"array.rechunk.fanin-limit": 10}):
        steps = plan_rechunk(old, new, 8)
    assert all(steps[i] != steps[i + 1] for i in range(len(steps) - 1))
    assert steps[-1] == new


def test_rechunk_below_limit_unchanged():
    """A rechunk whose natural fan-in is below the limit gets no extra steps."""
    x = da.ones((200, 200), chunks=(20, 20))
    y = x.rechunk((40, 40))  # fan-in ~4, far below the limit

    keys_default = set(y.optimize().__dask_graph__())
    with dask.config.set({"array.rechunk.fanin-limit": 10**9}):
        keys_unbounded = set(y.optimize().__dask_graph__())

    assert keys_default == keys_unbounded
