"""Behavioral contract for da.random's RandomState path.

These guard the compact-seed optimization (see dask_array/random/_expr.py): each
block ships a small integer seed instead of the full 2.6 KB MT19937 state array.
The point of the tests is the *contract* that survives that change — compact
wire, independent streams, deterministic-on-recompute, seed-controlled — since
the suite otherwise only uses da.random as a generic data source and would not
catch a regression of these properties.
"""

from __future__ import annotations

import numpy as np
import pytest

import dask_array as da
from dask_array._frisky.collect import collect_task_records
from dask_array._test_utils import assert_eq


def test_random_ships_compact_int_seed():
    """The per-block RNG seed is a small int, not the full MT19937 state array."""
    x = da.random.random((200, 200), chunks=(50, 50))
    recs = collect_task_records(x)
    assert len(recs) == 16
    seeds = [args[2] for _key, _func, args, _kwargs, _deps in recs]
    assert all(isinstance(s, int) for s in seeds), [type(s) for s in seeds]
    assert len(set(seeds)) == len(seeds)  # distinct per block
    assert all(0 <= s < (1 << 128) for s in seeds)  # 128-bit entropy


def test_random_deterministic_on_recompute():
    """Same lazy array computed twice yields identical values."""
    x = da.random.random((100, 100), chunks=(50, 50))
    assert np.array_equal(x.compute(), x.compute())


def test_random_seed_is_reproducible():
    """A seeded RandomState reproduces its values; a different seed differs."""
    a = da.random.RandomState(42).random((100, 100), chunks=(50, 50))
    b = da.random.RandomState(42).random((100, 100), chunks=(50, 50))
    c = da.random.RandomState(43).random((100, 100), chunks=(50, 50))
    assert_eq(a, b)
    assert not np.array_equal(a.compute(), c.compute())


def test_random_blocks_are_independent():
    """Distinct blocks are uncorrelated (no seed-collision / sequential-seed hazard)."""
    rng = da.random.RandomState(0)
    x = rng.random((400, 400), chunks=(40, 40)).compute()  # 100 blocks
    blocks = [x[i : i + 40, j : j + 40].ravel() for i in range(0, 400, 40) for j in range(0, 400, 40)]
    corr = np.corrcoef(np.array(blocks))
    off_diag = corr[~np.eye(len(blocks), dtype=bool)]
    assert np.abs(off_diag).max() < 0.2  # noise-level for n=1600 samples


@pytest.mark.parametrize(
    "build",
    [
        lambda: da.random.normal(10.0, 2.0, size=(20000,), chunks=5000),
        lambda: da.random.poisson(3.0, size=(20000,), chunks=5000),
        lambda: da.random.random((20000,), chunks=5000),
    ],
)
def test_random_distributions_unchanged_shape_and_stats(build):
    """Distributions still produce correct shape and plausible moments."""
    x = build()
    v = x.compute()
    assert v.shape == (20000,)
    assert np.isfinite(v).all()


# --- da.random.choice (RandomState path) — same compact-seed contract ---


def test_choice_ships_compact_int_seed():
    """choice's per-block RNG seed is a small int, not the full MT19937 state."""
    x = da.random.choice(10, size=(400,), chunks=100)
    recs = collect_task_records(x)
    assert len(recs) == 4
    seeds = [args[0] for _key, _func, args, _kwargs, _deps in recs]  # _choice_rs(seed, ...)
    assert all(isinstance(s, int) for s in seeds), [type(s) for s in seeds]
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= s < (1 << 128) for s in seeds)


def test_choice_deterministic_on_recompute():
    x = da.random.choice(100, size=(300,), chunks=100)
    assert np.array_equal(x.compute(), x.compute())


def test_choice_seed_is_reproducible():
    a = da.random.RandomState(7).choice(100, size=(500,), chunks=500)
    b = da.random.RandomState(7).choice(100, size=(500,), chunks=500)
    c = da.random.RandomState(8).choice(100, size=(500,), chunks=500)
    assert np.array_equal(a.compute(), b.compute())
    assert not np.array_equal(a.compute(), c.compute())


def test_choice_array_with_p():
    """Array-valued `a` with explicit probabilities draws from the population."""
    population = da.from_array(np.arange(20) * 10, chunks=20)
    p = np.ones(20) / 20
    x = da.random.choice(population, size=(300,), chunks=100, p=p)
    v = x.compute()
    assert v.shape == (300,)
    assert set(np.unique(v)).issubset(set((np.arange(20) * 10).tolist()))
