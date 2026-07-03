"""Tests for coarse_blockdim: preferring larger chunks in binary operations.

When combining arrays with different chunk granularities, we prefer coarser
chunks (fewer blocks) when boundaries align. This reduces task overhead.
"""

from __future__ import annotations

import math

import dask
import numpy as np
import pytest

import dask_array as da
from dask_array._test_utils import assert_eq


def total_chunks(arr):
    """Total number of chunks across all dimensions."""
    return math.prod(arr.numblocks)


class TestCoarseChunkPreference:
    """Tests for preferring coarser chunks when boundaries align."""

    def test_shuffle_indexed_array(self):
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

    def test_aligned_1d(self):
        """1D: (20,20) + (10,10,10,10) -> (20,20)"""
        coarse = da.ones(40, chunks=20)
        fine = da.ones(40, chunks=10)

        result = coarse + fine

        assert result.chunks == ((20, 20),)

    def test_aligned_2d(self):
        """2D: coarse chunks preferred in both dimensions."""
        coarse = da.ones((40, 40), chunks=(20, 20))
        fine = da.ones((40, 40), chunks=(10, 10))

        result = coarse + fine

        assert result.chunks == ((20, 20), (20, 20))
        assert total_chunks(result) == 4  # not 16

    def test_multiples_align(self):
        """Chunk sizes that are multiples align: (30,30) + (10,...) -> (30,30)"""
        coarse = da.ones(60, chunks=30)
        fine = da.ones(60, chunks=10)

        result = coarse + fine

        assert result.chunks == ((30, 30),)


class TestFallbackToCommonBlockdim:
    """Tests for falling back when boundaries don't align."""

    def test_misaligned_boundaries(self):
        """(15,15) vs (10,20): boundary 15 not in {10}, must subdivide."""
        a = da.ones(30, chunks=(15, 15))
        b = da.ones(30, chunks=(10, 20))

        result = a + b

        # Neither input's chunks work; uses finest common divisor
        assert result.chunks != ((15, 15),)
        assert result.chunks != ((10, 20),)

    def test_non_divisible(self):
        """(12,12) vs (8,8,8): boundary 12 not in {8,16}, must subdivide."""
        a = da.ones(24, chunks=12)
        b = da.ones(24, chunks=8)

        result = a + b

        # More chunks than either input
        assert len(result.chunks[0]) > 2

    def test_classic_uneven(self):
        """(4,6) vs (6,4): different boundaries, uses (4,2,4)."""
        a = da.arange(10, chunks=((4, 6),))
        b = da.ones(10, chunks=((6, 4),))

        result = a + b

        assert result.chunks == ((4, 2, 4),)


class TestUnifyChunksSizeLimit:
    """The size guard: coarsening must not manufacture chunks past the limit."""

    def test_refines_instead_of_merging_past_limit(self):
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

    def test_merges_under_limit(self):
        """Default limit leaves the coarse preference alone."""
        coarse = da.ones(40, chunks=20)
        fine = da.ones(40, chunks=10)

        result = coarse + fine

        assert result.chunks == ((20, 20),)

    def test_guard_scales_with_other_dims(self):
        """A 1D coarse operand must not inflate a 2D partner past the limit.

        This is the shape of the bug that motivated the guard: a coarsely
        chunked time vector met per-day-chunked (time, asset) quantities and
        merged them into multi-GB chunks.
        """
        x2d = da.ones((900, 50), chunks=(100, 50))  # 40 kB chunks
        t1d = da.ones(900, chunks=((600, 300),))  # boundaries nest in x2d's

        with dask.config.set({"array.unify-chunks-limit": "100 kiB"}):
            with pytest.warns(da.PerformanceWarning, match="unify-chunks-limit"):
                result = x2d * t1d[:, None]
                assert result.chunks[0] == (100,) * 9  # refined, not merged

        # under a permissive limit the same shapes merge (fresh arrays: the
        # refined expression above is a cached singleton)
        y2d = da.ones((960, 50), chunks=(96, 50))
        u1d = da.ones(960, chunks=((672, 288),))
        with dask.config.set({"array.unify-chunks-limit": "10 MiB"}):
            assert (y2d * u1d[:, None]).chunks[0] == (672, 288)

    def test_refine_policy(self):
        """array.unify-chunks-policy="refine" restores stock-dask unification."""
        coarse = da.ones(56, chunks=28)
        fine = da.ones(56, chunks=14)

        with dask.config.set({"array.unify-chunks-policy": "refine"}):
            result = coarse + fine
            assert result.chunks == ((14, 14, 14, 14),)
        assert_eq(result, np.full(56, 2.0))

    def test_shrinking_operand_does_not_trip(self):
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

    def test_single_chunk_operand_still_defers(self):
        """Trivial (single-chunk) axes carry no layout opinion; no guard trip."""
        x = da.ones(88, chunks=11)
        whole = da.ones(88, chunks=88)

        with dask.config.set({"array.unify-chunks-limit": "100 B"}):
            result = x + whole
            assert result.chunks == ((11,) * 8,)


class TestChunkReport:
    def test_names_the_offending_layout(self):
        x2d = da.ones((800, 40), chunks=(80, 40))
        t1d = da.ones(800, chunks=((560, 240),))
        with dask.config.set({"array.unify-chunks-limit": "1 GiB"}):
            y = x2d * t1d[:, None]
            y.chunks  # unify under the permissive limit -> merged layout

        report = da.chunk_report(y)

        assert "2ch(560..240) x 1ch(40)" in report  # the merged product layout
        assert "largest chunks" in report
        assert "layouts" in report

    def test_metadata_only(self):
        """The report never computes; a poisoned graph still reports fine."""

        def boom():
            raise AssertionError("chunk_report must not compute")

        x = da.from_delayed(dask.delayed(boom)(), shape=(10,), dtype=float)
        report = da.chunk_report(x + 1)
        assert "1ch(10)" in report
