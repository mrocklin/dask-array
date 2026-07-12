from functools import cached_property

import dask
import numpy as np
import pytest

import dask_array as da
from dask_array._expr import ArrayExpr
from dask_array._new_collection import new_collection
from dask_array._test_utils import assert_eq


class _LowerOnlyDrift(ArrayExpr):
    _parameters = ["array"]

    @cached_property
    def _name(self):
        return f"lower-only-drift-{self.array._name}"

    @cached_property
    def _meta(self):
        return self.array._meta

    @cached_property
    def chunks(self):
        return self.array.chunks

    def _lower(self):
        return self.array.rechunk(((2, 2, 2, 2),))


def test_map_blocks_explicit_chunks_preserves_rechunked_slice_block():
    x = da.ones((104, 2), chunks=(8, 2))
    arr = (x + 1)[95:103].rechunk((8, 2))

    def block_shape_code(block):
        return np.array([[100 * block.shape[0] + block.shape[1]]], dtype="int64")

    out = arr.map_blocks(
        block_shape_code,
        dtype="int64",
        chunks=(1, 1),
        meta=np.array((), dtype="int64"),
    )

    assert out.chunks == ((1,), (1,))
    for optimize_graph in (True, False):
        result = dask.compute(out, optimize_graph=optimize_graph)[0]
        np.testing.assert_array_equal(result, np.array([[802]], dtype="int64"))
    assert out.optimize().chunks == ((1,), (1,))


def test_map_blocks_explicit_chunks_preserves_multiple_input_block_shapes():
    x = da.ones((20, 2), chunks=(4, 2))
    arr = (x + 1)[3:11].rechunk(((3, 5), (2,)))

    def block_rows(block):
        return np.array([[block.shape[0]]], dtype="int64")

    out = arr.map_blocks(
        block_rows,
        dtype="int64",
        chunks=(1, 1),
        meta=np.array((), dtype="int64"),
    )

    assert arr.chunks == ((3, 5), (2,))
    assert out.chunks == ((1, 1), (1,))
    assert_eq(out, np.array([[3], [5]]))
    assert out.optimize().chunks == ((1, 1), (1,))


def test_map_blocks_explicit_chunks_preserves_nested_elemwise_slice_block():
    x = da.ones((32, 2), chunks=(4, 2))
    y = da.where(da.isnan((x + 1) * 2), 0, (x + 1) * 2)
    arr = y[1:5].rechunk((4, 2))

    def block_shape_code(block):
        return np.array([[100 * block.shape[0] + block.shape[1]]], dtype="int64")

    out = arr.map_blocks(
        block_shape_code,
        dtype="int64",
        chunks=(1, 1),
        meta=np.array((), dtype="int64"),
    )

    assert arr.chunks == ((4,), (2,))
    assert out.chunks == ((1,), (1,))
    for optimize_graph in (True, False):
        result = dask.compute(out, optimize_graph=optimize_graph)[0]
        np.testing.assert_array_equal(result, np.array([[402]], dtype="int64"))


def test_map_blocks_without_explicit_chunks_preserves_input_block_shapes():
    x = da.ones((16,), chunks=(4,))
    y = da.where(da.isnan((x + 1) * 2), 0, (x + 1) * 2)
    arr = y[1:5]

    def block_length(block):
        return np.full(block.shape, block.shape[0], dtype="int64")

    out = arr.map_blocks(block_length, dtype="int64")

    assert arr.chunks == ((3, 1),)
    assert out.chunks == ((3, 1),)
    for optimize_graph in (True, False):
        np.testing.assert_array_equal(out.compute(optimize_graph=optimize_graph), np.array([3, 3, 3, 1]))


def test_map_blocks_invalid_explicit_chunk_count_still_raises():
    x = da.ones((4,), chunks=(1,))
    y = x.map_blocks(lambda block: block, chunks=((1, 1),), dtype=x.dtype)

    with pytest.raises(ValueError, match="adjust_chunks specified with 2 blocks"):
        y.compute()
    optimized = y.optimize()
    with pytest.raises(ValueError, match="adjust_chunks specified with 2 blocks"):
        optimized.chunks
    with pytest.raises(ValueError, match="adjust_chunks specified with 2 blocks"):
        optimized.compute()


def test_map_blocks_block_info_stable_through_sliding_window_rewrite():
    # reduction(sliding_window_view(x)) advertises coarsened chunks at
    # construction, but simplify rewrites it onto the input's native chunks.
    # map_blocks freezes its block_info payloads at construction, so the
    # input layout must be pinned or the payloads silently desynchronize
    # (this raised "adjust_chunks specified with N blocks" before ChunksFreeze).
    from dask_array._overlap import sliding_window_view

    x = da.from_array(np.arange(150, dtype="f8").reshape(50, 3), chunks=(10, 3))
    r = sliding_window_view(x, 25, axis=0).sum(axis=-1)
    advertised = r.chunks
    assert r.simplify().chunks != advertised  # the rewrite really drifts

    calls = []

    def sentinel(block, block_info=None):
        info = block_info[None]
        input_info = block_info[0]
        calls.append(
            (
                info["chunk-location"],
                block.shape,
                info["num-chunks"],
                input_info["chunk-location"],
                input_info["array-location"],
                input_info["num-chunks"],
            )
        )
        return np.zeros((1, 1), dtype="uint8")

    out = r.map_blocks(sentinel, dtype="uint8", chunks=(1, 1), meta=np.array((), dtype="uint8"))
    numblocks = tuple(len(c) for c in advertised)
    assert out.numblocks == numblocks

    result = out.compute(scheduler="sync")
    assert result.shape == numblocks
    assert len(calls) == np.prod(numblocks)
    starts = [np.cumsum((0,) + c) for c in advertised]
    for loc, shape, num_chunks, input_loc, input_array_location, input_num_chunks in calls:
        assert num_chunks == numblocks
        assert shape == tuple(c[i] for c, i in zip(advertised, loc))
        assert input_loc == loc
        assert input_num_chunks == numblocks
        assert input_array_location == [(starts[axis][i], starts[axis][i + 1]) for axis, i in enumerate(loc)]


def test_map_blocks_block_info_one_task_per_day_through_rolling_slice_rechunk():
    # The daily-writer pattern: rolling -> slice -> rechunk(one block per day)
    # -> map_blocks whose function writes one day per task, indexed by
    # block_info. Must stay one task per day block with stable indexing
    # through optimization, even though the rolling subtree's chunks drift
    # under simplify.
    from dask_array._overlap import sliding_window_view

    spd, days, out_days, window = 10, 8, 3, 25
    n = spd * days
    x = da.from_array(np.arange(n * 3, dtype="f8").reshape(n, 3), chunks=(spd, 3))
    r = sliding_window_view(x, window, axis=0).sum(axis=-1)
    sliced = r[spd : spd + out_days * spd].rechunk((spd, 3))
    assert sliced.chunks == ((spd,) * out_days, (3,))

    calls = []

    def write_day(block, block_info=None):
        info = block_info[None]
        calls.append((info["chunk-location"], block.shape, info["num-chunks"]))
        return np.zeros((1, 1), dtype="uint8")

    out = sliced.map_blocks(write_day, dtype="uint8", chunks=(1, 1), meta=np.array((), dtype="uint8"))
    assert out.numblocks == (out_days, 1)

    result = out.compute(scheduler="sync")
    assert result.shape == (out_days, 1)
    assert sorted(loc for loc, _, _ in calls) == [(i, 0) for i in range(out_days)]
    for loc, shape, num_chunks in calls:
        assert shape == (spd, 3)
        assert num_chunks == (out_days, 1)


def test_map_blocks_block_id_stable_through_sliding_window_rewrite():
    from dask_array._overlap import sliding_window_view

    x = da.from_array(np.arange(150, dtype="f8").reshape(50, 3), chunks=(10, 3))
    r = sliding_window_view(x, 25, axis=0).sum(axis=-1)
    advertised = r.chunks
    assert r.simplify().chunks != advertised

    calls = []

    def sentinel(block, block_id=None):
        calls.append((block_id, block.shape))
        return np.zeros((1, 1), dtype="uint8")

    out = r.map_blocks(sentinel, dtype="uint8", chunks=(1, 1), meta=np.array((), dtype="uint8"))
    out.compute(scheduler="sync")
    numblocks = tuple(len(c) for c in advertised)
    assert len(calls) == np.prod(numblocks)
    for block_id, shape in calls:
        assert shape == tuple(c[i] for c, i in zip(advertised, block_id))


def test_freeze_chunks_pins_layout_without_materializing():
    from dask_array._overlap import sliding_window_view

    x = da.from_array(np.arange(150, dtype="f8").reshape(50, 3), chunks=(10, 3))
    r = sliding_window_view(x, 25, axis=0).sum(axis=-1)
    advertised = r.chunks
    assert r.simplify().chunks != advertised

    frozen = r.freeze_chunks()
    # Cheap: no materialization happened as a side effect.
    assert "_lowered_expr" not in frozen.__dict__
    assert "_lowered_expr" not in r.__dict__
    # Idempotent.
    assert frozen.freeze_chunks() is frozen
    # The layout promise survives optimization; the raw expression's doesn't.
    assert frozen.optimize().chunks == advertised
    assert_eq(frozen, r.compute())


def test_map_blocks_block_info_stable_through_lower_time_chunk_drift():
    x = da.from_array(np.arange(8), chunks=(4,))
    arr = new_collection(_LowerOnlyDrift(x.expr))
    assert arr.chunks == ((4, 4),)

    calls = []

    def sentinel(block, block_info=None):
        input_info = block_info[0]
        calls.append((block.shape, input_info["chunk-location"], input_info["array-location"]))
        return np.array([block.shape[0]], dtype="int64")

    out = arr.map_blocks(sentinel, dtype="int64", chunks=(1,), meta=np.array((), dtype="int64"))

    assert_eq(out, np.array([4, 4]), scheduler="sync")
    assert sorted(calls) == [
        ((4,), (0,), [(0, 4)]),
        ((4,), (1,), [(4, 8)]),
    ]


def test_map_blocks_auto_freeze_leaves_no_graph_residue_without_drift():
    # map_blocks pins block_info consumers' input layouts, but when nothing
    # rewrites the layout the pin lowers to its input: no alias layer, no
    # extra keys, and the same task count as an unpinned map_blocks.
    x = da.ones((8, 2), chunks=(4, 2))
    y = x + 1

    def with_info(block, block_info=None):
        return block

    def without_info(block):
        return block

    pinned = y.map_blocks(with_info, dtype=y.dtype)
    unpinned = y.map_blocks(without_info, dtype=y.dtype)
    pinned_graph = pinned.__dask_graph__()
    assert not any("chunks-freeze" in str(k) for k in pinned_graph)
    assert len(pinned_graph) == len(unpinned.__dask_graph__())
    assert_eq(pinned, np.full((8, 2), 2.0))
