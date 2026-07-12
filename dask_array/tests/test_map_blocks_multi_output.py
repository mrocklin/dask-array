import numpy as np

import dask
import dask_array as da
from dask_array._map_blocks import map_blocks_multi_output
from dask_array._test_utils import assert_eq


def split_block(spec, block):
    return {
        "double": block * 2,
        "row_sum": block.sum(axis=1),
    }


def test_map_blocks_multi_output_computes_projected_arrays():
    x = da.from_array(np.arange(8).reshape(4, 2), chunks=(2, 2))
    block_specs = {(i, j): None for i in range(2) for j in range(1)}

    double, row_sum = map_blocks_multi_output(
        split_block,
        [x.expr],
        [("x", "y")],
        ("x", "y"),
        block_specs,
        [
            {
                "key": "double",
                "indices": ("x", "y"),
                "chunks": x.chunks,
                "dtype": x.dtype,
            },
            {
                "key": "row_sum",
                "indices": ("x",),
                "chunks": (x.chunks[0],),
                "dtype": x.dtype,
            },
        ],
        token="split",
    )

    assert isinstance(double, da.Array)
    assert isinstance(row_sum, da.Array)
    assert_eq(double, np.arange(8).reshape(4, 2) * 2)
    assert_eq(row_sum, np.arange(8).reshape(4, 2).sum(axis=1))

    opt_double, opt_row_sum = dask.optimize(double, row_sum)
    assert_eq(opt_double, np.arange(8).reshape(4, 2) * 2)
    assert_eq(opt_row_sum, np.arange(8).reshape(4, 2).sum(axis=1))

    persisted_double, persisted_row_sum = dask.persist(double, row_sum, scheduler="single-threaded")
    assert_eq(persisted_double, np.arange(8).reshape(4, 2) * 2)
    assert_eq(persisted_row_sum, np.arange(8).reshape(4, 2).sum(axis=1))


def test_map_blocks_multi_output_shares_block_calls():
    calls = []

    def record_block(spec, block):
        calls.append(spec)
        return {"a": block + 1, "b": block + 2}

    x = da.from_array(np.arange(6), chunks=(3,))

    a, b = map_blocks_multi_output(
        record_block,
        [x.expr],
        [("x",)],
        ("x",),
        {(0,): 0, (1,): 1},
        [
            {"key": "a", "indices": ("x",), "chunks": x.chunks, "dtype": x.dtype},
            {"key": "b", "indices": ("x",), "chunks": x.chunks, "dtype": x.dtype},
        ],
        token="record",
    )

    got_a, got_b = dask.compute(a, b, scheduler="single-threaded")

    np.testing.assert_array_equal(got_a, np.arange(6) + 1)
    np.testing.assert_array_equal(got_b, np.arange(6) + 2)
    assert sorted(calls) == [0, 1]


def test_map_blocks_multi_output_single_projection_omits_other_projection_keys():
    x = da.from_array(np.arange(6), chunks=(3,))

    a, b = map_blocks_multi_output(
        lambda spec, block: {"a": block + 1, "b": block + 2},
        [x.expr],
        [("x",)],
        ("x",),
        {(0,): None, (1,): None},
        [
            {"key": "a", "indices": ("x",), "chunks": x.chunks, "dtype": x.dtype},
            {"key": "b", "indices": ("x",), "chunks": x.chunks, "dtype": x.dtype},
        ],
        token="cull",
    )

    graph_keys = set(a.__dask_graph__())

    assert not any(key[0].startswith(b.name) for key in graph_keys if isinstance(key, tuple))
    assert_eq(a, np.arange(6) + 1)
