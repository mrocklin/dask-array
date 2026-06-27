from __future__ import annotations

import importlib.util
import sys

import dask
from dask.core import flatten
from dask.local import get_sync
import numpy as np
import pytest

import dask_array as da
from dask_array._new_collection import new_collection
from dask_array._overlap import overlap_internal


def test_dask_graph_does_not_import_frisky_modules():
    for name in list(sys.modules):
        if name == "dask_array._rust" or name.startswith("dask_array._frisky"):
            sys.modules.pop(name)

    x = da.ones((4, 4), chunks=(2, 2)) + 1
    assert "dask_array._frisky" not in sys.modules
    assert "dask_array._rust" not in sys.modules

    x.__dask_graph__()

    assert "dask_array._frisky" not in sys.modules
    assert "dask_array._rust" not in sys.modules


def test_frisky_graph_imports_frisky_modules():
    x = da.ones((4, 4), chunks=(2, 2)) + 1
    records = x.__frisky_graph__()

    assert records
    assert "dask_array._frisky" in sys.modules


def test_numeric_scalar_blockwise_uses_binary_records():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = new_collection((da.ones((4, 4), chunks=(2, 2)) + 1).expr.optimize(fuse=False))

    chunks, records, _chunk_groups = x.__frisky_records_chunks__()

    assert len(chunks) == 2
    assert records == []


def test_sliding_window_overlap_uses_binary_records():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.sliding_window_view(da.ones(12, chunks=4), 3).sum()

    chunks, records, _chunk_groups = x.__frisky_records_chunks__()

    assert chunks
    assert records == []


def test_chunk_groups_carry_name_and_metadata():
    """Each binary chunk ships its producing expr's ``_name`` (the stable layer
    identity, which a key prefix can't always recover) plus an opaque JSON blob of
    op/shape/chunks/dtype, parallel to ``chunks``."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    x = new_collection((da.ones((4, 4), chunks=(2, 2)) + 1).expr.optimize(fuse=False))
    chunks, records, chunk_groups = x.__frisky_records_chunks__()

    assert len(chunk_groups) == len(chunks) == 2
    for name, meta in chunk_groups:
        assert isinstance(name, str) and name  # the expr `_name`
        info = json.loads(meta)
        assert info["shape"] == [4, 4]
        assert info["chunks"] == [[2, 2], [2, 2]]
        assert info["numblocks"] == [2, 2]
        assert info["dtype"]  # e.g. "float64"
        assert "op" in info
        assert isinstance(info["params"], dict)  # scalar params, not array inputs
    # The two layers are distinct exprs => distinct group names.
    assert len({name for name, _ in chunk_groups}) == 2

    # The expression's scalar parameters are captured (op/shape/etc.), while its
    # array inputs (child exprs) are excluded.
    by_op = {json.loads(m)["op"]: json.loads(m)["params"] for _, m in chunk_groups}
    assert by_op["Elemwise"]["op"] == "add"  # the `+ 1` operator, not an array
    assert by_op["Ones"]["shape"] == [4, 4]  # a scalar param on the source


def test_overlap_native_layer_matches_legacy_graph():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.from_array(np.arange(4 * 6).reshape(4, 6), chunks=(2, 3))
    y = overlap_internal(x, {0: 1, 1: (0, 2)})
    expr = y.expr
    dep_graph = dict(expr.array.__dask_graph__())
    legacy_graph = expr._layer()
    native_graph = expr._frisky_layer().to_dask_graph()

    for key in flatten(expr.__dask_keys__()):
        expected = get_sync({**dep_graph, **legacy_graph}, key)
        actual = get_sync({**dep_graph, **native_graph}, key)
        np.testing.assert_array_equal(actual, expected)


def _fallback_expr_counts(monkeypatch, x):
    from collections import Counter

    from dask_array._frisky import collect as collect_mod
    from dask_array._frisky.graph_records import GraphRecordsLayer

    calls = []

    class SpyGraphRecordsLayer(GraphRecordsLayer):
        def __init__(self, expr):
            calls.append(type(expr).__name__)
            super().__init__(expr)

    monkeypatch.setattr(collect_mod, "GraphRecordsLayer", SpyGraphRecordsLayer)
    x.__frisky_records_chunks__()
    return Counter(calls)


def _require_tuple_index(block, index):
    if not isinstance(index, tuple):
        raise TypeError(f"expected tuple index, got {type(index).__name__}")
    return block


def test_map_overlap_trim_blockwise_dep_uses_native_records(monkeypatch):
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.from_array(np.arange(12), chunks=4)
    y = x.map_overlap(lambda block: block + 1, depth=1, boundary="none", trim=True)

    assert _fallback_expr_counts(monkeypatch, y) == {}
    np.testing.assert_array_equal(y.compute(scheduler="synchronous"), np.arange(12) + 1)


def test_array_slice_blockwise_dep_preserves_tuple_records(monkeypatch):
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    from dask.layers import ArraySliceDep

    x = da.from_array(np.arange(12).reshape(3, 4), chunks=(2, 2))
    y = da.blockwise(
        _require_tuple_index,
        "ij",
        x,
        "ij",
        ArraySliceDep(x.chunks),
        "ij",
        dtype=x.dtype,
        meta=np.array((), dtype=x.dtype),
    )
    expr = y.expr
    dep_graph = dict(x.__dask_graph__())
    legacy_graph = expr._layer()
    native_graph = expr._frisky_layer().to_dask_graph()

    for key in flatten(expr.__dask_keys__()):
        expected = get_sync({**dep_graph, **legacy_graph}, key)
        actual = get_sync({**dep_graph, **native_graph}, key)
        np.testing.assert_array_equal(actual, expected)

    assert _fallback_expr_counts(monkeypatch, y) == {}


def test_shuffle_native_layer_matches_legacy_graph(monkeypatch):
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    array = np.arange(4 * 8).reshape(4, 8)
    x = da.from_array(array, chunks=(2, 4))
    y = da.shuffle(x, [[6, 5, 2], [4, 1], [3, 0, 7]], axis=1)
    expr = y.expr
    dep_graph = dict(expr.array.__dask_graph__())
    legacy_graph = expr._layer()
    native_graph = expr._frisky_layer().to_dask_graph()

    for key in flatten(expr.__dask_keys__()):
        expected = get_sync({**dep_graph, **legacy_graph}, key)
        actual = get_sync({**dep_graph, **native_graph}, key)
        np.testing.assert_array_equal(actual, expected)

    assert _fallback_expr_counts(monkeypatch, y) == {}
    np.testing.assert_array_equal(
        y.compute(scheduler="synchronous"),
        array[:, [6, 5, 2, 4, 1, 3, 0, 7]],
    )


def test_shuffle_shares_takers_across_off_axis_blocks():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.from_array(np.arange(6 * 8).reshape(6, 8), chunks=(1, 4))
    y = da.shuffle(x, [[6, 5, 2], [4, 1], [3, 0, 7]], axis=1)
    records = y.expr._frisky_layer().to_task_records()
    data_name = f"{y.expr._name}-data"

    data_records = [record for record in records if record[0].startswith(f"('{data_name}',")]

    # Three output chunks, each with one sorter and two source-taker payloads.
    assert len(data_records) == 9


def test_shuffle_single_source_data_records_are_referenced():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.from_array(np.arange(4 * 8).reshape(4, 8), chunks=(2, 4))
    y = da.shuffle(x, [[3, 2, 1, 0], [7, 6, 5, 4]], axis=1)
    records = y.expr._frisky_layer().to_task_records()
    data_name = f"{y.expr._name}-data"

    data_keys = {record[0] for record in records if record[0].startswith(f"('{data_name}',")}
    deps = {dep for record in records for dep in record[4]}

    assert len(data_keys) == 2
    assert data_keys <= deps


def test_bool_scalar_blockwise_stays_on_python_records():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = new_collection((da.ones((4, 4), chunks=(2, 2)) == True).expr.optimize(fuse=False))  # noqa: E712

    chunks, records, _chunk_groups = x.__frisky_records_chunks__()

    assert len(chunks) == 1
    assert records


def test_numpy_scalar_blockwise_stays_on_python_records():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    scalar = np.float64(1)
    x = new_collection((da.ones((4, 4), chunks=(2, 2), dtype="float32") + scalar).expr.optimize(fuse=False))

    chunks, records, _chunk_groups = x.__frisky_records_chunks__()

    assert len(chunks) == 1
    assert records
    assert any(type(arg) is np.float64 for _, _, args, _, _ in records for arg in args)


def test_large_int_scalar_blockwise_stays_on_python_records():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    scalar = 10**20
    x = new_collection((da.ones((4, 4), chunks=(2, 2)) + scalar).expr.optimize(fuse=False))

    chunks, records, _chunk_groups = x.__frisky_records_chunks__()

    assert len(chunks) == 1
    assert records
    assert any(arg == scalar for _, _, args, _, _ in records for arg in args)


def _constant_block():
    return np.full((2, 2), 7.0)


def _constant_frisky_graph(self, seen=None):
    return [(self.__frisky_output_keys__()[0], _constant_block, (), {}, [])]


def test_frisky_scheduler_uses_frisky_graph(array_scheduler, monkeypatch):
    if array_scheduler != "frisky":
        pytest.skip("requires --scheduler=frisky")

    x = da.ones((2, 2), chunks=(2, 2)) + 1
    monkeypatch.setattr(type(x), "__frisky_graph__", _constant_frisky_graph)

    (result,) = dask.compute(x)

    np.testing.assert_array_equal(result, np.full((2, 2), 7.0))


def test_persisted_collection_arithmetic_roundtrips(array_scheduler):
    """Chained persist: a persisted collection lowers to a ``FromGraph`` whose
    blocks ARE frisky Futures. Doing arithmetic on it and recomputing must wire
    those futures as dependency edges through the records/expression path —
    otherwise the worker runs the consuming op on an unresolved placeholder
    (regression: ``TypeError: unsupported operand type(s) for *:
    'types.SimpleNamespace' and 'int'``). The dask-array suite otherwise only
    exercises ``_layer``, never the records path, so this is the only coverage."""
    if array_scheduler != "frisky":
        pytest.skip("requires --scheduler=frisky")

    x = da.ones((4, 4), chunks=(2, 2)) + 1
    xp = x.persist()
    # The persisted collection is FromGraph-backed (its blocks are futures).
    assert type(xp.expr).__name__ == "FromGraph"

    # Recomputing the persisted collection directly: an unresolved placeholder
    # would have no shape and crash in finalize/concatenate.
    np.testing.assert_array_equal(xp.compute(), np.full((4, 4), 2.0))
    # Arithmetic on the persisted collection, then compute (the reported crash).
    np.testing.assert_array_equal((xp * 2).compute(), np.full((4, 4), 4.0))
    # A reduction over the persisted collection also resolves its blocks.
    np.testing.assert_array_equal((xp + 5).sum().compute(), np.full((4, 4), 7.0).sum())
