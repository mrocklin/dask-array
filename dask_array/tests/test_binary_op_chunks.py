"""Tests for chunk unification in binary operations.

When combining arrays with different chunk granularities, we prefer coarser
chunks (fewer blocks) when boundaries align — merging cuts task overhead.
The default "auto" policy is cost-aware: merging finer operands up to a
coarse layout moves their bytes (concatenation), so the merge only happens
while those bytes stay in proportion to the operands already at that layout;
otherwise unification refines instead (splits only, no data movement).
"""

from __future__ import annotations

import math
import warnings

import dask
import numpy as np
import pytest

import dask_array as da
from dask_array._test_utils import assert_eq


def total_chunks(arr):
    """Total number of chunks across all dimensions."""
    return math.prod(arr.numblocks)


# =============================================================================
# Coarse chunk preference: prefer coarser chunks when boundaries align.
# =============================================================================


def test_shuffle_indexed_array():
    """Main use case: xarray groupby pattern.

    Binary op between array with nice chunks and shuffle-indexed array
    (which has per-element chunks) should preserve the nice chunks.
    """
    # Original data: 10 chunks of size 12
    arr = da.random.random((120, 20, 30), chunks=(12, 20, 30))

    # Aggregated data indexed to match original shape
    n_groups = 4
    mean_arr = da.random.random((n_groups, 20, 30), chunks=(1, 20, 30))
    indexer = np.tile(np.arange(n_groups), 30)
    indexed_mean = mean_arr[indexer, ...]

    result = arr - indexed_mean

    # Should preserve arr's chunk count, not explode to 120
    assert total_chunks(result) <= total_chunks(arr) * 2
    assert_eq(result, arr.compute() - indexed_mean.compute())


def test_aligned_1d():
    """1D: (20,20) + (10,10,10,10) -> (20,20)"""
    coarse = da.ones(40, chunks=20)
    fine = da.ones(40, chunks=10)

    result = coarse + fine

    assert result.chunks == ((20, 20),)


def test_aligned_2d():
    """2D: coarse chunks preferred in both dimensions."""
    coarse = da.ones((40, 40), chunks=(20, 20))
    fine = da.ones((40, 40), chunks=(10, 10))

    result = coarse + fine

    assert result.chunks == ((20, 20), (20, 20))
    assert total_chunks(result) == 4  # not 16


def test_multiples_align():
    """Chunk sizes that are multiples align: (30,30) + (10,...) -> (30,30)"""
    coarse = da.ones(60, chunks=30)
    fine = da.ones(60, chunks=10)

    result = coarse + fine

    assert result.chunks == ((30, 30),)


# =============================================================================
# Interleaved boundaries: the auto policy realigns to an operand's own layout
# (see the realign-interleaved section below) instead of manufacturing the
# finest common refinement; the refinement remains the fallback when
# realigning would move too many bytes, and under policy="refine".
# =============================================================================


def test_misaligned_boundaries():
    """(15,15) vs (10,20): boundaries interleave; equal weights realign
    to one input's layout (a sliver moves either way) rather than
    splitting both to (10,5,5,10).  A full tie on (blocks, cost, anchored
    bytes) resolves by the lexicographic layout tie-break -- (10,20) is
    the *second* operand, so first-operand-wins would fail here."""
    a = da.ones(30, chunks=(15, 15))
    b = da.ones(30, chunks=(10, 20))

    result = a + b

    assert result.chunks == ((10, 20),)
    assert_eq(result, np.full(30, 2.0))


def test_non_divisible():
    """(12,12) vs (8,8,8): realigns to the fewest-block layout."""
    a = da.ones(24, chunks=12)
    b = da.ones(24, chunks=8)

    result = a + b

    assert result.chunks == ((12, 12),)


def test_classic_uneven():
    """(4,6) vs (6,4): a full tie -- both directions move 2 of 10
    elements -- resolved by the deterministic lexicographic tie-break,
    rather than splitting both to (4,2,4)."""
    a = da.arange(10, chunks=((4, 6),))
    b = da.ones(10, chunks=((6, 4),))

    result = a + b

    assert result.chunks == ((4, 6),)
    assert_eq(result, np.arange(10) + 1.0)


# =============================================================================
# Cost-aware direction: the default "auto" policy chooses the merge direction
# by what it costs.
#
# Merging a finer operand up to a coarse layout moves that operand's bytes;
# refining a coarser operand only splits it.  A coarse operand pulls the
# others up only while the moved bytes stay within a small multiple of the
# bytes already at its layout.  (Shapes here are unique per test: chunks
# resolve lazily at first ``.chunks`` access and expressions are cached
# singletons — see the NOTE in test_refines_instead_of_merging_past_limit.)
# =============================================================================


def test_light_coarse_operand_does_not_inflate():
    """The inflation-bug shape: a small 2-chunk time vector must not merge
    a heavy per-day-chunked panel — the panel keeps its layout and the
    vector splits, with no guard trip and no warning."""
    x2d = da.ones((840, 30), chunks=(70, 30))  # 197 kB, 12 fine row-chunks
    t1d = da.ones(840, chunks=((560, 280),))  # 6.6 kB, nests in x2d's grid

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = x2d * t1d[:, None]
        assert result.chunks[0] == (70,) * 12

    assert_eq(result, np.ones((840, 30)))


def test_light_fine_operand_follows_heavy_coarse():
    """The mirror image: a small finely-chunked vector merges up to a
    heavy coarse partner's layout (moving the vector is cheap)."""
    big = da.ones((6400, 40), chunks=(3200, 40))  # 2 MB, 2 coarse chunks
    idx = da.ones(6400, chunks=800)  # 50 kB, 8 fine chunks

    result = big * idx[:, None]

    assert result.chunks[0] == (3200, 3200)


def test_fragment_healing_merge():
    """A light operand may still propose a coarse layout when the merge
    moves almost nothing — re-coalescing a sliver-fragmented panel (the
    shift/pad pattern) must keep working."""
    frag = da.ones((1442, 30), chunks=((1, 720, 721), 30))  # heavy panel
    lite = da.ones((1442, 1), chunks=((721, 721), 1))  # light, coarse

    result = frag + lite

    assert result.chunks[0] == (721, 721)


def test_comparable_weights_keep_merging():
    """Operands within the cost ratio (here 2x, from mixed dtypes) merge
    as before; only clearly lighter operands lose the right to inflate."""
    fine64 = da.ones(5600, chunks=700)
    coarse32 = da.ones(5600, chunks=1400, dtype="f4")

    result = fine64 + coarse32

    assert result.chunks == ((1400,) * 4,)


def test_cost_ratio_boundary():
    """Pin _MERGE_COST_RATIO = 4: the same merge (moving 75% of a float64
    operand's bytes) is accepted against an int16 anchor (3x its weight)
    and refused against a bool anchor (6x its weight)."""
    mover = da.ones(800, chunks=100)  # merge to (400, 400) moves 4800 B

    under = mover + da.ones(800, chunks=((400, 400),), dtype="i2")  # 4 x 1600 B
    assert under.chunks == ((400, 400),)

    over = mover + da.ones(800, chunks=((400, 400),), dtype=bool)  # 4 x 800 B
    assert over.chunks == ((100,) * 8,)


def test_coarse_policy_always_merges():
    """policy="coarse" restores the unconditional merge direction."""
    x2d = da.ones((880, 20), chunks=(88, 20))
    t1d = da.ones(880, chunks=((616, 264),))

    with dask.config.set({"array.unify-chunks-policy": "coarse"}):
        result = x2d * t1d[:, None]
        assert result.chunks[0] == (616, 264)


# =============================================================================
# The unify-chunks size guard: coarsening must not manufacture chunks past
# the limit.
# =============================================================================


def test_refines_instead_of_merging_past_limit():
    """Merging (12,)*4 up to (24,24) would exceed a tiny limit -> refine + warn.

    NOTE (here and below): chunk unification runs lazily on first ``.chunks``
    access, and expressions are singletons -- so each test touches ``.chunks``
    inside its config context and uses shapes no other test builds.  Lowering
    re-runs unification under the config current at optimize/compute time, so
    a compute outside the context executes that config's layout (values are
    identical either way; these tests assert layout at metadata level).
    """
    coarse = da.ones(48, chunks=24)  # 192 B chunks
    fine = da.ones(48, chunks=12)

    with dask.config.set({"array.unify-chunks-limit": "100 B"}):
        with pytest.warns(da.PerformanceWarning, match="unify-chunks-limit"):
            result = coarse + fine
            assert result.chunks == ((12, 12, 12, 12),)
    assert_eq(result, np.full(48, 2.0))


def test_merges_under_limit():
    """Default limit leaves the coarse preference alone."""
    coarse = da.ones(40, chunks=20)
    fine = da.ones(40, chunks=10)

    result = coarse + fine

    assert result.chunks == ((20, 20),)


def test_guard_scales_with_other_dims():
    """A 1D coarse operand must not inflate a 2D partner past the limit.

    This is the shape of the bug that motivated the guard: a coarsely
    chunked time vector met per-day-chunked (time, asset) quantities and
    merged them into multi-GB chunks.  The auto policy now refuses this
    merge outright (see the cost-aware direction section), so the guard is
    pinned under the always-merge "coarse" policy it was built for.
    """
    x2d = da.ones((900, 50), chunks=(100, 50))  # 40 kB chunks
    t1d = da.ones(900, chunks=((600, 300),))  # boundaries nest in x2d's

    with dask.config.set({"array.unify-chunks-policy": "coarse", "array.unify-chunks-limit": "100 kiB"}):
        with pytest.warns(da.PerformanceWarning, match="unify-chunks-limit"):
            result = x2d * t1d[:, None]
            assert result.chunks[0] == (100,) * 9  # refined, not merged

    # under a permissive limit the same shapes merge (fresh arrays: the
    # refined expression above is a cached singleton)
    y2d = da.ones((960, 50), chunks=(96, 50))
    u1d = da.ones(960, chunks=((672, 288),))
    with dask.config.set({"array.unify-chunks-policy": "coarse", "array.unify-chunks-limit": "10 MiB"}):
        assert (y2d * u1d[:, None]).chunks[0] == (672, 288)


def test_refine_policy():
    """array.unify-chunks-policy="refine" restores stock-dask unification."""
    coarse = da.ones(56, chunks=28)
    fine = da.ones(56, chunks=14)

    with dask.config.set({"array.unify-chunks-policy": "refine"}):
        result = coarse + fine
        assert result.chunks == ((14, 14, 14, 14),)
    assert_eq(result, np.full(56, 2.0))


def test_shrinking_operand_does_not_trip():
    """Only chunks the merge would manufacture count against the limit.

    The single-chunk 2D operand's chunks *shrink* under the merged layout,
    so its (over-limit) size must not force refinement of the others.
    """
    a2d = da.ones((2250, 40), chunks=(2250, 20))  # 360 kB now, 240 kB merged
    b1d = da.ones(2250, chunks=((1500, 750),))
    c1d = da.ones(2250, chunks=750)  # nests in b1d; merge grows it to 12 kB

    with dask.config.set({"array.unify-chunks-limit": "100 kiB"}):
        result = a2d * b1d[:, None] * c1d[:, None]
        assert result.chunks[0] == (1500, 750)  # merged; no false trip


def test_single_chunk_operand_still_defers():
    """Trivial (single-chunk) axes carry no layout opinion; no guard trip."""
    x = da.ones(88, chunks=11)
    whole = da.ones(88, chunks=88)

    with dask.config.set({"array.unify-chunks-limit": "100 B"}):
        result = x + whole
        assert result.chunks == ((11,) * 8,)


# =============================================================================
# chunk_report
# =============================================================================


def test_chunk_report_names_the_offending_layout():
    x2d = da.ones((800, 40), chunks=(80, 40))
    t1d = da.ones(800, chunks=((560, 240),))
    # always-merge policy: the report should name the manufactured layout
    with dask.config.set({"array.unify-chunks-policy": "coarse", "array.unify-chunks-limit": "1 GiB"}):
        y = x2d * t1d[:, None]
        y.chunks  # unify under the permissive limit -> merged layout

    report = da.chunk_report(y)

    assert "2ch(560..240) x 1ch(40)" in report  # the merged product layout
    assert "largest chunks" in report
    assert "layouts" in report


def test_chunk_report_metadata_only():
    """The report never computes; a poisoned graph still reports fine."""

    def boom():
        raise AssertionError("chunk_report must not compute")

    x = da.from_delayed(dask.delayed(boom)(), shape=(10,), dtype=float)
    report = da.chunk_report(x + 1)
    assert "1ch(10)" in report


# =============================================================================
# Realign interleaved (non-nested) layouts — the roll/shift pattern.
#
# When boundaries interleave, neither operand's layout nests in the
# other's, so merging is off the table.  Refining moves no bytes but
# multiplies blocks and manufactures slivers that every downstream op
# inherits; realigning the misaligned operands to an anchor layout that
# already holds enough bytes moves only what crosses boundaries (one
# sliver per block, for a small shift).  The auto policy realigns to the
# fewest-block feasible anchor (then least movement) under the same cost
# ratio as merging.
# (Shapes here are unique per test: chunks resolve lazily at first
# ``.chunks`` access and expressions are cached singletons.)
# =============================================================================


def test_roll_sliver_realigns_to_source_layout():
    """x + roll(x, 1): the rolled operand is offset by one row; realign
    moves one sliver per boundary instead of splitting every block of
    both operands."""
    x = da.random.random((1200, 6), chunks=(150, 6))
    result = x + da.roll(x, 1, axis=0)

    assert result.chunks == x.chunks
    xx = x.compute()
    assert_eq(result, xx + np.roll(xx, 1, axis=0))


def test_half_chunk_shift_still_realigns():
    """Even a half-chunk shift (moves half the rolled operand's bytes)
    stays within the cost ratio for equal-weight operands."""
    x = da.ones((1440, 4), chunks=(160, 4))
    result = x + da.roll(x, 80, axis=0)

    assert result.chunks == x.chunks


def test_light_interleaved_operand_follows_heavy():
    """The lighter operand moves to the heavier operand's grid, never
    the reverse (bool anchor vs complex mover, offset boundaries)."""
    heavy = da.ones(1760, chunks=((110,) + (220,) * 7 + (110,),), dtype="c16")
    light = da.ones(1760, chunks=220, dtype=bool)

    result = heavy + light

    assert result.chunks == heavy.chunks


def test_refine_policy_still_refines():
    """policy="refine" keeps the stock-dask escape hatch: interleaved
    layouts split to the finest common boundaries, no realign."""
    a = da.ones(1520, chunks=190)
    b = da.ones(1520, chunks=((95,) + (190,) * 7 + (95,),))

    with dask.config.set({"array.unify-chunks-policy": "refine"}):
        result = a + b
        assert result.chunks == ((95,) * 16,)


def test_roll_into_reduction_stays_untouched():
    """Negative control: roll feeding straight into a reduction meets no
    alignment demand, so no rechunk may be inserted anywhere."""
    x = da.random.random((1360, 4), chunks=(170, 4))
    result = da.roll(x, 1, axis=0).sum()

    opt = result.expr.optimize()
    names = [type(node).__name__ for node in opt.walk()]
    assert not any("Rechunk" in name for name in names), names
