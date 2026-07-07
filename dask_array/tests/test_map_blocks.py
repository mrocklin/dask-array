import dask
import numpy as np
import pytest

import dask_array as da


def test_map_blocks_explicit_chunks_preserves_rechunked_slice_block():
    x = da.ones((104, 2), chunks=(8, 2))
    arr = (x + 1)[95:103].rechunk((8, 2))

    seen = []

    def sentinel(block):
        seen.append(block.shape)
        return np.array([[1]], dtype="uint8")

    out = arr.map_blocks(
        sentinel,
        dtype="uint8",
        chunks=(1, 1),
        meta=np.array((), dtype="uint8"),
    )

    assert out.chunks == ((1,), (1,))
    for optimize_graph in (True, False):
        seen.clear()
        result = dask.compute(out, optimize_graph=optimize_graph)[0]
        np.testing.assert_array_equal(result, np.array([[1]], dtype="uint8"))
        assert seen == [(8, 2)]
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
    np.testing.assert_array_equal(out.compute(), np.array([[3], [5]]))
    assert out.optimize().chunks == ((1, 1), (1,))


def test_map_blocks_explicit_chunks_preserves_nested_elemwise_slice_block():
    x = da.ones((32, 2), chunks=(4, 2))
    y = da.where(da.isnan((x + 1) * 2), 0, (x + 1) * 2)
    arr = y[1:5].rechunk((4, 2))

    seen = []

    def sentinel(block):
        seen.append(block.shape)
        return np.array([[1]], dtype="uint8")

    out = arr.map_blocks(
        sentinel,
        dtype="uint8",
        chunks=(1, 1),
        meta=np.array((), dtype="uint8"),
    )

    assert arr.chunks == ((4,), (2,))
    assert out.chunks == ((1,), (1,))
    for optimize_graph in (True, False):
        seen.clear()
        result = dask.compute(out, optimize_graph=optimize_graph)[0]
        np.testing.assert_array_equal(result, np.array([[1]], dtype="uint8"))
        assert seen == [(4, 2)]


def test_map_blocks_without_explicit_chunks_preserves_input_block_shapes():
    x = da.ones((16,), chunks=(4,))
    y = da.where(da.isnan((x + 1) * 2), 0, (x + 1) * 2)
    arr = y[1:5]
    seen = []

    def observe(block):
        seen.append(block.shape)
        return block

    out = arr.map_blocks(observe, dtype=arr.dtype)

    assert arr.chunks == ((3, 1),)
    assert out.chunks == ((3, 1),)
    for optimize_graph in (True, False):
        seen.clear()
        np.testing.assert_array_equal(out.compute(optimize_graph=optimize_graph), np.full(4, 4.0))
        assert sorted(seen) == [(1,), (3,)]


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
