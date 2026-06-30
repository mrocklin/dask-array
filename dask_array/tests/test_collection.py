from __future__ import annotations

import cloudpickle
import operator

import numpy as np
import pytest

import dask
import dask_array as da
from dask import is_dask_collection
from dask.core import flatten
from dask_array._test_utils import assert_eq
from dask_array._collection import Array
from dask_array._rechunk import Rechunk


@pytest.fixture()
def arr():
    return da.random.random((10, 10), chunks=(5, 6))


@pytest.mark.parametrize(
    "op",
    [
        "__add__",
        "__sub__",
        "__mul__",
        "__truediv__",
        "__floordiv__",
        "__pow__",
        "__radd__",
        "__rsub__",
        "__rmul__",
        "__rtruediv__",
        "__rfloordiv__",
        "__rpow__",
    ],
)
def test_arithmetic_ops(arr, op):
    result = getattr(arr, op)(2)
    expected = getattr(arr.compute(), op)(2)
    assert_eq(result, expected)


def test_rechunk(arr):
    result = arr.rechunk((7, 3))
    expected = arr.compute()
    assert_eq(result, expected)


def test_array_pickle_drops_lowered_expr_cache():
    x = da.from_array(np.arange(12).reshape(3, 4), chunks=(1, 2)) + 1

    expected_keys = x.__dask_keys__()
    assert "_lowered_expr" in vars(x)

    y = cloudpickle.loads(cloudpickle.dumps(x))
    assert "_lowered_expr" not in vars(y)
    assert y.__dask_keys__() == expected_keys
    assert "_lowered_expr" in vars(y)
    assert_eq(y, np.arange(12).reshape(3, 4) + 1)


def test_array_pickle_preserves_lowering_config_for_key_stability():
    with dask.config.set({"array.optimize-graph": True}):
        x = (da.from_array(np.arange(20), chunks=5) + 1)[:12]
        expected_keys = x.__dask_keys__()
        blob = cloudpickle.dumps(x)

    with dask.config.set({"array.optimize-graph": False}):
        y = cloudpickle.loads(blob)
        assert "_lowered_expr" not in vars(y)
        assert y.__dask_keys__() == expected_keys
        assert_eq(y, np.arange(20)[:12] + 1)


def test_array_pickle_preserves_lowering_config_for_frisky_records():
    with dask.config.set({"array.optimize-graph": True}):
        x = (da.from_array(np.arange(20), chunks=5) + 1)[:12]
        output_keys = x.__frisky_output_keys__()
        blob = cloudpickle.dumps(x)

    with dask.config.set({"array.optimize-graph": False}):
        y = cloudpickle.loads(blob)
        records = y.__frisky_graph__()

    produced = {key for key, _func, _args, _kwargs, _deps in records}
    assert y.__frisky_output_keys__() == output_keys
    assert set(output_keys) <= produced


def test_lowering_shares_work_across_collections_with_shared_ancestry():
    """Lowering many collections that share a deep ancestry must reuse a shared,
    name-keyed lowering cache so the common subtree is lowered (and tokenized)
    once, not once per collection.

    Each collection's ``_lowered_expr`` is computed independently, but the
    collections overlap: ``cols[k]`` contains ``cols[k-1]`` as a subexpression.
    Without the shared cache the overlap is re-lowered per collection, so the
    cost of lowering *all* of them is O(depth**2). We assert near-linear growth
    by tokenize-call count, which doubles (not quadruples) when depth doubles.
    """
    import dask.tokenize

    def build_chain(depth):
        a = da.ones((100, 100), chunks=(10, 10))
        cols = []
        for _ in range(depth):
            # .mean lowers into a partial-reduce/aggregate subgraph, so lowering
            # genuinely rebuilds nodes each layer (the quadratic shows up here).
            a = a + a.mean(axis=1, keepdims=True)
            cols.append(a)
        return cols

    def count_lower_tokenize(cols):
        # Counting lowering work by tokenize calls couples to a dask internal
        # (``dask.tokenize._tokenize``); there is no public hook. ``tokenize()``
        # resolves it by module-global lookup at call time, so the patch takes.
        calls = [0]
        original = dask.tokenize._tokenize

        def counted(*args, **kwargs):
            calls[0] += 1
            return original(*args, **kwargs)

        dask.tokenize._tokenize = counted
        try:
            for c in cols:
                c._lowered_expr
        finally:
            dask.tokenize._tokenize = original
        return calls[0]

    n_d = count_lower_tokenize(build_chain(16))
    n_2d = count_lower_tokenize(build_chain(32))

    # Linear growth is ~2x; quadratic (the unfixed bug) is ~4x. 3x cleanly
    # separates the two regimes with margin for incidental variation.
    assert n_2d < 3 * n_d


def test_blockwise():
    x = da.random.random((10, 10), chunks=(5, 5))
    z = da.blockwise(operator.add, "ij", x, "ij", 100, None, dtype=x.dtype)
    assert_eq(z, x.compute() + 100)

    x = da.random.random((10, 10), chunks=(5, 5))
    z = da.blockwise(operator.add, "ij", x, "ij", x, "ij", dtype=x.dtype)
    expr = z.expr.optimize()
    assert len(list(expr.find_operations(Rechunk))) == 0
    assert_eq(z, x.compute() * 2)

    # align
    x = da.random.random((10, 10), chunks=(5, 5))
    y = da.random.random((10, 10), chunks=(7, 3))
    z = da.blockwise(operator.add, "ij", x, "ij", y, "ij", dtype=x.dtype)
    expr = z.expr.optimize()
    assert len(list(expr.find_operations(Rechunk))) > 0
    assert_eq(z, x.compute() + y.compute())


@pytest.mark.parametrize("func", ["min", "max", "sum", "prod", "mean", "any", "all"])
def test_reductions(arr, func):
    # var and std need __array_function__
    result = getattr(arr, func)(axis=0)
    expected = getattr(arr.compute(), func)(axis=0)
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "func",
    [
        "sum",
        "mean",
        "any",
        "all",
        "max",
        "min",
        "nanmin",
        "nanmax",
        "nanmean",
        "nansum",
        "nanprod",
    ],
)
def test_reductions_toplevel(arr, func):
    # var and std need __array_function__
    result = getattr(da, func)(arr, axis=0)
    expected = getattr(np, func)(arr.compute(), axis=0)
    assert_eq(result, expected)


def test_from_array():
    x = np.random.random((10, 10))
    d = da.from_array(x, chunks=(5, 5))
    assert_eq(d, x)
    assert d.chunks == ((5, 5), (5, 5))


def test_from_array_name_is_exact():
    x = np.arange(6)

    d = da.from_array(x, chunks=3, name="custom-name")

    assert d.name == "custom-name"
    assert_eq(d, x)


# compute=False hands the store back lazily, so the caller's later .compute()
# decides the scheduler. Under Frisky that serializes the target and writes to a
# copy, so the in-place mutation can't be observed (the compute=True path forces a
# local scheduler in da.store; this one can't). Local schedulers only.
@pytest.mark.requires_local_scheduler
def test_delayed_can_unpack_compute_false_store():
    x = np.arange(12).reshape(3, 4)
    y = da.from_array(x, chunks=(2, 2))
    target = np.empty_like(x)

    writes = da.store(y, target, compute=False, return_stored=True)
    result = dask.delayed(lambda value: value)([writes]).compute()

    np.testing.assert_array_equal(target, x)
    np.testing.assert_array_equal(result[0], x)


def test_store_region_rechunked_exact_name_slice():
    x = np.ones(30)
    y = da.from_array(x, chunks=(10, 10, 10), name="x")[5:25].rechunk((10, 10))
    target = np.zeros(30)

    da.store(y, target, regions=(slice(5, 25),))

    expected = np.zeros(30)
    expected[5:25] = 1
    np.testing.assert_array_equal(target, expected)


def test_store_forces_local_scheduler_only_for_inmemory_targets():
    # store() forces a local scheduler only when an in-memory numpy target would
    # otherwise be mutated on a copy by a serializing scheduler. File-backed
    # targets (zarr/h5py) must keep the ambient scheduler so remote workers can
    # write them in parallel (xarray.to_zarr relies on this).
    from dask_array.io._store import _force_local_store_scheduler

    inmem = np.zeros(3)

    class FileBacked:  # not an ndarray; stands in for zarr/h5py
        def __setitem__(self, key, value):
            pass

    file_backed = FileBacked()
    client = object()  # non-string, non-local scheduler (Frisky/distributed-like)

    with dask.config.set({"scheduler": "threads"}):  # local: never force
        assert not _force_local_store_scheduler([inmem], None)
    with dask.config.set({"scheduler": client}):
        assert _force_local_store_scheduler([inmem], None)  # in-memory: force
        assert not _force_local_store_scheduler([file_backed], None)  # file-backed: don't
        assert not _force_local_store_scheduler([inmem], "sync")  # explicit: respect it


@pytest.mark.parametrize(
    "array",
    [
        da.from_array(np.arange(10), chunks=4).rechunk((10,)),
        da.from_array(np.arange(4).reshape(2, 2), chunks=1).reshape(-1),
    ],
)
def test_name_matches_dask_key_namespace_after_lowering(array):
    keys = list(flatten(array.__dask_keys__()))

    assert keys
    assert all(key[0] == array.name for key in keys)


def test_reshape_accepts_c_order_keyword():
    x = da.from_array(np.arange(6), chunks=3)

    assert_eq(x.reshape((2, 3), order="C"), np.arange(6).reshape((2, 3)))


def test_rechunk_auto_object_dtype_raises():
    data = np.array(["a", "bb", "ccc", "dddd"], dtype=object)
    x = da.from_array(data, chunks=(2,))

    with pytest.raises(NotImplementedError, match="object dtype"):
        x.rechunk("auto")


def test_from_graph_same_key_prefix_different_layers():
    from dask_array.core import from_graph

    a = from_graph(
        {("x", 0): np.array([1])},
        np.empty((0,), dtype=int),
        ((1,),),
        [("x", 0)],
        "a",
    )
    b = from_graph(
        {("x", 0): np.array([2])},
        np.empty((0,), dtype=int),
        ((1,),),
        [("x", 0)],
        "b",
    )

    assert a.expr is not b.expr
    assert_eq(a, np.array([1]))
    assert_eq(b, np.array([2]))


def test_from_graph_tracks_expression_dependencies():
    from dask._task_spec import DependenciesMapping, Task, TaskRef
    from dask_array.core import from_graph

    x = da.from_array(np.arange(6), chunks=(3,)).rechunk((2,))
    name = "plus-one"
    layer = {(name, i): Task((name, i), operator.add, TaskRef((x.name, i)), 1) for i in range(len(x.chunks[0]))}

    y = from_graph(
        layer,
        np.empty((0,), dtype=x.dtype),
        x.chunks,
        [(name, i) for i in range(len(x.chunks[0]))],
        name,
        dependencies=[x],
    )
    optimized = da.Array(y[:4].expr.optimize(fuse=True))
    graph = optimized.__dask_graph__()
    missing = [dep for deps in DependenciesMapping(graph).values() for dep in deps if dep not in graph]

    assert not missing
    assert_eq(optimized, np.arange(4) + 1)


def test_from_graph_accepts_rename_keyword():
    from dask_array.core import from_graph

    x = from_graph(
        {("x", 0): np.array([1])},
        np.empty((0,), dtype=int),
        ((1,),),
        [("x", 0)],
        "x",
    )

    rebuild, args = x.__dask_postpersist__()
    renamed = rebuild(x.__dask_graph__(), *args, rename={x.name: "renamed"})

    assert renamed.name.startswith("renamed-")
    assert_eq(renamed, np.array([1]))


@pytest.mark.xfail(reason="Requires dask core to recognize 'dask_array' module in is_dask_collection")
def test_is_dask_collection_doesnt_materialize():
    class ArrayTest(Array):
        def __dask_graph__(self):
            raise NotImplementedError

    arr = ArrayTest(da.random.random((10, 10), chunks=(5, 5)).expr)
    assert is_dask_collection(arr)
    with pytest.raises(NotImplementedError):
        arr.__dask_graph__()


def test_astype():
    x = da.random.randint(1, 100, (10, 10), chunks=(5, 5))
    result = x.astype(np.float64)
    expected = x.compute().astype(np.float64)
    assert_eq(result, expected)


def test_stack_promote_type():
    i = np.arange(10, dtype="i4")
    f = np.arange(10, dtype="f4")
    di = da.from_array(i, chunks=5)
    df = da.from_array(f, chunks=5)
    res = da.stack([di, df])
    assert_eq(res, np.stack([i, f]))


def test_field_access():
    x = np.array([(1, 1.0), (2, 2.0)], dtype=[("a", "i4"), ("b", "f4")])
    y = da.from_array(x, chunks=(1,))
    assert_eq(y["a"], x["a"])
    assert_eq(y[["b", "a"]], x[["b", "a"]])


def test_field_access_with_shape():
    dtype = [("col1", ("f4", (3, 2))), ("col2", ("f4", 3))]
    data = np.ones((100, 50), dtype=dtype)
    x = da.from_array(data, 10)
    assert_eq(x["col1"], data["col1"])
    assert_eq(x[["col1"]], data[["col1"]])
    assert_eq(x["col2"], data["col2"])
    assert_eq(x[["col1", "col2"]], data[["col1", "col2"]])


# =============================================================================
# Optimization tests (ported from dask-expr prototype)
# =============================================================================


def test_transpose_optimize():
    """Test that transpose of transpose simplifies."""
    a = np.random.random((10, 20))
    b = da.from_array(a, chunks=(2, 5))

    # T.T should be identity
    assert b.T.T.expr.optimize()._name == b.expr.optimize()._name
    assert_eq(b.T.T, a)

    # Explicit axes composition
    c = da.from_array(np.random.random((3, 4, 5)), chunks=(1, 2, 3))
    d = c.transpose((2, 0, 1)).transpose((1, 2, 0))  # Should compose to (0, 1, 2) = identity
    assert_eq(d, c)


def test_rechunk_optimize():
    """Test that rechunk of rechunk simplifies to single rechunk."""
    a = np.random.random((10, 10))
    b = da.from_array(a, chunks=(4, 4))

    c = b.rechunk((2, 5)).rechunk((5, 2))
    d = b.rechunk((5, 2))

    # Double rechunk should simplify to single rechunk
    assert c.expr.optimize()._name == d.expr.optimize()._name
    assert_eq(c, a)


def test_dask_optimize_rechunk():
    x = da.from_array(np.arange(12), chunks=3).rechunk((4,))

    (optimized,) = dask.optimize(x)

    assert_eq(optimized, np.arange(12))
    assert optimized.chunks == ((4, 4, 4),)


def test_slicing_optimize_identity():
    """Test that no-op slice simplifies to identity."""
    a = np.random.random((10, 20))
    b = da.from_array(a, chunks=(2, 5))

    # b[:] should simplify to b
    assert b[:].expr.optimize()._name == b.expr._name
    assert_eq(b[:], a)


def test_slicing_optimize_fusion():
    """Test that slice of slice fuses into single slice."""
    a = np.random.random((10, 20))
    b = da.from_array(a, chunks=(2, 5))

    # Slice fusion: b[5:, 4][::2] should equal b[5::2, 4]
    result = b[5:, 4][::2]
    expected = b[5::2, 4]
    assert result.expr.optimize()._name == expected.expr.optimize()._name
    assert_eq(result, a[5::2, 4])


def test_slicing_pushdown_elemwise():
    """Test that slice pushes through elemwise."""
    a = np.random.random((10, 20))
    b = da.from_array(a, chunks=(2, 5))

    # (b + 1)[:5] should become (b[:5] + 1)
    result = (b + 1)[:5]
    expected = b[:5] + 1
    assert result.expr.optimize()._name == expected.expr.optimize()._name
    assert_eq(result, (a + 1)[:5])

    # Test with integer index that reduces dimension
    result2 = (b + 1)[5]
    expected2 = b[5] + 1
    assert result2.expr.optimize()._name == expected2.expr.optimize()._name
    assert_eq(result2, (a + 1)[5])


def test_slicing_pushdown_elemwise_broadcast():
    """Test slice pushdown through elemwise with broadcasting."""
    a = np.random.random((10, 20))
    c = np.random.random((20,))  # broadcasts on axis 0
    aa = da.from_array(a, chunks=(2, 5))
    cc = da.from_array(c, chunks=(5,))

    # (aa + cc)[:5] should become (aa[:5] + cc)
    # cc doesn't get sliced because axis 0 is broadcast
    result = (aa + cc)[:5]
    expected = aa[:5] + cc
    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, (a + c)[:5])

    # (aa + cc)[:, ::2] should become (aa[:, ::2] + cc[::2])
    result2 = (aa + cc)[:, ::2]
    expected2 = aa[:, ::2] + cc[::2]
    assert result2.expr.simplify()._name == expected2.expr.simplify()._name
    assert_eq(result2, (a + c)[:, ::2])


def test_slicing_pushdown_transpose():
    """Test slice pushdown through transpose."""
    a = np.random.random((10, 20))
    b = da.from_array(a, chunks=(2, 5))

    # b.T[5:] should become b[:, 5:].T
    result = b.T[5:]
    expected = b[:, 5:].T
    assert result.expr.optimize()._name == expected.expr.optimize()._name
    assert_eq(result, a.T[5:])


def test_rechunk_pushdown_transpose():
    """Test rechunk pushdown through transpose."""
    a = np.random.random((10, 20))
    b = da.from_array(a, chunks=(2, 5))

    # b.T.rechunk((10, 5)) should become Transpose(Rechunk(...))
    # not Rechunk(Transpose(...))
    result = b.T.rechunk((10, 5))
    opt = result.expr.optimize()
    # Should be Transpose at top level (rechunk pushed inside)
    assert type(opt).__name__ == "Transpose"
    assert_eq(result, a.T)


def test_rechunk_pushdown_elemwise():
    """Test rechunk pushdown through elemwise."""
    a = np.random.random((10, 10))
    b = da.from_array(a, chunks=(4, 4))

    # (b + 1).rechunk((5, 5)) should become Elemwise at top level
    # not Rechunk(Elemwise(...))
    result = (b + 1).rechunk((5, 5))
    opt = result.expr.optimize()
    # Should be Elemwise at top level (rechunk pushed inside)
    assert type(opt).__name__ == "Elemwise"
    assert_eq(result, a + 1)


def test_rechunk_pushdown_elemwise_broadcast():
    """Test rechunk pushdown through elemwise with broadcasting."""
    a = np.random.random((10,))
    aa = da.from_array(a)
    b = np.random.random((10, 10))
    bb = da.from_array(b)

    # (aa + bb).rechunk((5, 2)) should become Elemwise at top level
    c = (aa + bb).rechunk((5, 2))
    # Expected: rechunk pushed to inputs
    expected = aa.rechunk((2,)) + bb.rechunk((5, 2))
    assert c.expr.simplify()._name == expected.expr.simplify()._name

    opt = c.expr.optimize()
    # Should be Elemwise at top level (rechunk pushed inside)
    assert type(opt).__name__ == "Elemwise"
    assert_eq(c, a + b)


# =============================================================================
# Optimization correctness and safety tests
# =============================================================================


def test_optimization_correctness_various_chains():
    """Verify optimized expressions produce correct results."""
    np.random.seed(42)
    a = da.random.random((15, 25), chunks=(3, 7))
    a_np = a.compute()

    # Various operation chains - verify correctness
    assert_eq(a.T.T, a_np)
    assert_eq(a.T[5:].T, a_np[:, 5:])
    assert_eq((a + 1).rechunk((5, 5))[:10], (a_np + 1)[:10])
    assert_eq(a.rechunk((5, 5)).rechunk((3, 3)), a_np)
    assert_eq(a[::2, 1:][::2], a_np[::2, 1:][::2])
    assert_eq((a * 2)[:, 10:][5:], (a_np * 2)[:, 10:][5:])


def test_optimize_empty_array():
    """Verify optimizations handle empty arrays."""
    a = da.zeros((0, 10), chunks=(1, 5))
    result = (a + 1)[:, :5]
    assert result.shape == (0, 5)
    assert_eq(result, np.zeros((0, 5)))


def test_optimize_3d_transpose():
    """Verify transpose composition works for 3D arrays."""
    np.random.seed(42)
    a = da.random.random((4, 5, 6), chunks=2)

    # (2,0,1) then (1,2,0) should compose to identity
    result = a.transpose((2, 0, 1)).transpose((1, 2, 0))
    opt = result.expr.optimize()
    # Should simplify to original (no Transpose at top)
    assert type(opt).__name__ != "Transpose" or opt.axes == tuple(range(3))
    assert_eq(result, a)


def test_optimize_scalar_in_elemwise():
    """Verify scalar handling in elemwise pushdown."""
    np.random.seed(42)
    b = da.random.random((10, 10), chunks=5)
    b_np = b.compute()

    # Scalar + array, then slice
    result = (5 + b)[:5]
    assert_eq(result, (5 + b_np)[:5])

    # Slice then rechunk with scalar
    result = (b * 2).rechunk((5, 5))
    assert_eq(result, b_np * 2)


def test_chunks_preserved_after_optimization():
    """Verify chunk structure is correct after optimization."""
    a = da.random.random((20, 20), chunks=(4, 5))

    # Transpose then rechunk
    result = a.T.rechunk((10, 10))
    assert result.chunks == ((10, 10), (10, 10))

    # Elemwise then slice
    result = (a + 1)[:10, :15]
    assert result.chunks == ((4, 4, 2), (5, 5, 5))

    # Slice then rechunk
    result = a[:12, :8].rechunk((6, 4))
    assert result.chunks == ((6, 6), (4, 4))


def test_pushdown_broadcast_both_arrays():
    """Test pushdown when both arrays broadcast to output shape."""
    # (10, 1) + (1, 20) -> (10, 20)
    a = da.from_array(np.random.random((10, 1)), chunks=(5, 1))
    b = da.from_array(np.random.random((1, 20)), chunks=(1, 10))
    a_np, b_np = a.compute(), b.compute()

    # Slice pushdown - each input sliced on its non-broadcast dimension
    result = (a + b)[:5, :10]
    opt = result.expr.optimize()
    assert type(opt).__name__ == "Elemwise"
    # Input shapes should be sliced appropriately
    assert opt.elemwise_args[0].shape == (5, 1)
    assert opt.elemwise_args[1].shape == (1, 10)
    assert_eq(result, (a_np + b_np)[:5, :10])

    # Rechunk pushdown - each input rechunked on its non-broadcast dimension
    result = (a + b).rechunk((2, 5))
    opt = result.expr.optimize()
    assert type(opt).__name__ == "Elemwise"
    # Input chunks should be rechunked appropriately
    assert opt.elemwise_args[0].chunks == ((2, 2, 2, 2, 2), (1,))
    assert opt.elemwise_args[1].chunks == ((1,), (5, 5, 5, 5))
    assert_eq(result, a_np + b_np)


def test_rechunk_pushdown_to_io():
    """Rechunk should push down into FromArray by changing chunks parameter."""
    from dask_array.io import FromArray

    a = np.random.random((10, 10))
    b = da.from_array(a, chunks=(4, 4))

    result = b.rechunk((5, 2)).expr.optimize()

    # Rechunk is pushed into FromArray with the requested chunks.
    assert type(result) is FromArray
    assert result.chunks == ((5, 5), (2, 2, 2, 2, 2))
    assert_eq(da.Array(result), a)


def test_rechunk_chain_optimize():
    """Chained rechunks should collapse to single rechunk pushed to IO."""
    from dask_array.io import FromArray

    a = np.random.random((10, 10))
    b = da.from_array(a, chunks=(4, 4))

    result = b.rechunk((2, 5)).rechunk((5, 2)).expr.optimize()

    # Both rechunks eliminated, just FromArray with the final chunks.
    assert type(result) is FromArray
    assert result.chunks == ((5, 5), (2, 2, 2, 2, 2))
    assert_eq(da.Array(result), a)


def test_rechunk_transpose_pushdown_to_io():
    """Rechunk after transpose should push through to IO."""
    from dask_array.io import FromArray
    from dask_array.manipulation._transpose import Transpose

    a = np.random.random((10, 10))
    b = da.from_array(a, chunks=(4, 4))

    result = b.T.rechunk((5, 2)).expr.optimize()

    assert type(result) is Transpose
    assert type(result.array) is FromArray
    assert result.array.chunks == ((2, 2, 2, 2, 2), (5, 5))
    assert result.chunks == ((5, 5), (2, 2, 2, 2, 2))
    assert_eq(da.Array(result), a.T)


def test_rechunk_elemwise_pushdown_to_io():
    """Rechunk after elemwise should push through to IO inputs."""
    from dask_array._blockwise import Elemwise
    from dask_array.io import FromArray

    a = np.random.random((10, 10))
    b = da.from_array(a, chunks=(4, 4))

    result = (b + 1).rechunk((5, 5)).expr.optimize()

    # Rechunk pushed through elemwise into FromArray
    assert type(result) is Elemwise
    assert type(result.elemwise_args[0]) is FromArray
    assert result.elemwise_args[0].chunks == ((5, 5), (5, 5))
    # Verify the prefix is preserved
    assert result.elemwise_args[0].name.startswith("array-")


def test_rechunk_pushdown_concatenate_other_axis():
    """Rechunk pushes through concatenate when rechunking non-concat axis."""
    a = da.ones((10, 20), chunks=(5, 10))
    b = da.ones((10, 20), chunks=(5, 10))
    concat = da.concatenate([a, b], axis=0)  # shape (20, 20)

    # Rechunk axis 1 (not concat axis)
    result = concat.rechunk({1: 5})

    # Expected: rechunk pushed to inputs
    expected = da.concatenate([a.rechunk({1: 5}), b.rechunk({1: 5})], axis=0)

    # Structure should match
    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, expected)


def test_rechunk_pushdown_concatenate_correctness():
    """Verify rechunk through concatenate produces correct values with real data."""
    a = np.arange(20).reshape(4, 5)
    b = np.arange(20, 40).reshape(4, 5)
    da_a = da.from_array(a, chunks=(2, 3))
    da_b = da.from_array(b, chunks=(2, 3))

    concat = da.concatenate([da_a, da_b], axis=0)  # shape (8, 5)

    # Rechunk non-concat axis
    result = concat.rechunk({1: 2})
    expected = da.concatenate([da_a.rechunk({1: 2}), da_b.rechunk({1: 2})], axis=0)

    # Structure should match
    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, np.concatenate([a, b], axis=0))


# --- Fusion regression tests ---


def test_fusion_broadcast_modulo():
    """Test that fusion handles broadcasting correctly with modulo.

    When fusing operations where one array broadcasts (has fewer blocks),
    the block indices must use modulo to wrap around correctly.
    This is a regression test for matmul-like operations.
    """
    # 1D array broadcasting to 2D - simulates matmul broadcast pattern
    a = da.from_array(np.arange(6).reshape(2, 3), chunks=(1, 3))
    b = da.from_array(np.arange(3), chunks=3)

    # b broadcasts: it has 1 block but a has 2 blocks in first dimension
    result = a * b  # Elemwise with broadcast
    assert_eq(result, np.arange(6).reshape(2, 3) * np.arange(3))

    # Test that the fused graph computes correctly
    opt = result.expr.optimize(fuse=True)
    assert_eq(da.Array(opt), np.arange(6).reshape(2, 3) * np.arange(3))


def test_fusion_same_array_different_indices():
    """Test conflict detection when same array used with different indices.

    When the same array appears multiple times in a computation with
    different index mappings (e.g., da.dot(x, x)), fusion must detect
    this conflict and exclude the conflicting expression.
    """
    # da.dot(x, x) uses x with indices 'ij' and 'jk', different mappings
    x = da.from_array(np.arange(9).reshape(3, 3), chunks=(2, 2))
    x_np = x.compute()

    result = da.dot(x, x)
    expected = np.dot(x_np, x_np)
    assert_eq(result, expected)

    # Test with persist (triggers the conflict path during fusion)
    result_persisted = result.persist()
    assert_eq(result_persisted, expected)


def test_fusion_elemwise_with_out_and_where_true():
    """Test that out arrays don't break fusion when where=True.

    When an Elemwise has out=array but where=True (the default),
    the out array should not be a dependency since it's not used
    in the computation - it's just a placeholder for the result.
    """
    a = da.from_array(np.arange(4), chunks=2)
    b = da.from_array(np.arange(4, 8), chunks=2)
    out = da.zeros(4, chunks=2)

    # When where=True (default), out is just a placeholder
    result = da.add(a, b, out=out)
    assert result is out

    # Should compute correctly despite fusion
    assert_eq(result, np.arange(4) + np.arange(4, 8))


def test_fusion_elemwise_with_out_and_where_array():
    """Test that out arrays are properly used when where is an array.

    When where is a mask array (not True), the out array IS used
    as a dependency and should participate in the computation.
    """
    a = da.from_array(np.arange(4), chunks=2)
    b = da.from_array(np.arange(4, 8), chunks=2)
    where = da.from_array(np.array([True, False, True, False]), chunks=2)
    out = da.zeros(4, dtype=int, chunks=2)

    result = da.add(a, b, where=where, out=out)
    assert result is out

    # Should compute correctly: only positions where=True get the sum
    expected = np.zeros(4, dtype=int)
    np.add(
        np.arange(4),
        np.arange(4, 8),
        where=np.array([True, False, True, False]),
        out=expected,
    )
    assert_eq(result, expected)


def test_fusion_out_same_as_input():
    """Test that out=x works when x is also an input argument.

    When out is the same array as an input (e.g., np.sin(x, out=x)),
    we must NOT exclude it from dependencies since it's actually used.
    """
    x = da.from_array(np.array([0.0, 0.5, 1.0, 1.5]), chunks=2)
    x_np = x.compute().copy()

    # In-place operation: out is same as input
    result = np.sin(x, out=x)
    assert result is x

    expected = np.sin(x_np, out=x_np)
    assert_eq(result, expected)


def test_fusion_transpose_conflict():
    """Test conflict detection for a + a.T pattern.

    When the same array is accessed both directly and transposed,
    fusion must detect this conflict since different output blocks
    would need different source blocks from the same expression.
    """
    a = da.from_array(np.arange(9).reshape(3, 3), chunks=(2, 2))
    a_np = a.compute()

    # a + a.T accesses 'a' with different index mappings
    result = a + a.T
    expected = a_np + a_np.T
    assert_eq(result, expected)

    # Verify fusion handles this correctly
    opt = result.expr.optimize(fuse=True)
    assert_eq(da.Array(opt), expected)


def test_fusion_chained_transpose():
    """Test fusion with chained transpose operations.

    Operations like (a + b).T should fuse correctly since there's
    no conflict - just a consistent dimension permutation.
    """
    a = da.from_array(np.arange(6).reshape(2, 3), chunks=(1, 2))
    b = da.from_array(np.arange(6, 12).reshape(2, 3), chunks=(1, 2))
    a_np, b_np = a.compute(), b.compute()

    result = (a + b).T
    expected = (a_np + b_np).T
    assert_eq(result, expected)

    # Should fuse the add and transpose
    opt = result.expr.optimize(fuse=True)
    assert_eq(da.Array(opt), expected)


def test_reduction_scalar_aggregate_meta():
    """Regression test: reduction handles aggregate returning Python scalar.

    When a custom aggregate function returns a Python scalar instead of
    preserving array dimensions, the meta computation must not fail.
    Previously failed with:
    ValueError: cannot reshape array of size 1 into shape (0,0)
    """
    arr = da.ones((10, 5, 5), chunks=(5, 5, 5))

    # Custom aggregate that returns Python int (not numpy array)
    def scalar_agg(x, axis=None, keepdims=False):
        return 42

    # Should not raise ValueError when accessing _meta
    result = da.reduction(
        arr,
        chunk=np.sum,
        aggregate=scalar_agg,
        axis=0,
        dtype=float,
    )
    assert result._meta.shape == (0, 0)
    assert result._meta.dtype == np.float64


def test_fusion_blockwise_contracted_dimensions():
    """Test fusion with Blockwise that has contracted dimensions.

    When a Blockwise expression has indices in input that are not in output
    (contracted dimensions), the fusion must correctly handle block lookups.

    This is a regression test for xarray integration where groupby operations
    create Blockwise with out_ind=(2,) for 1D output from 3D input with
    ind=(0, 1, 2). When fused with Elemwise (out_ind=(0,)), the idx_to_block
    mapping must correctly handle the contracted dimensions 0 and 1.

    Previously failed with KeyError: 0 in FusedBlockwise._task().
    """
    from dask_array._blockwise import FusedBlockwise

    # Create 3D array with single blocks in contracted dimensions
    arr_3d = da.from_array(np.ones((1, 1, 3)), chunks=(1, 1, 1))

    # Blockwise that reduces dims 0 and 1, keeps dim 2 as output
    # out_ind=(2,) means output indexed by input's dimension 2
    result = da.blockwise(
        lambda x: x.mean(axis=(0, 1)),
        (2,),  # out_ind - output dimension comes from input dim 2
        arr_3d.expr,
        (0, 1, 2),  # ind - input has all 3 dimensions
        dtype=arr_3d.dtype,
    )

    # Verify Blockwise is fusable when contracted dims have single blocks
    assert result.expr._is_blockwise_fusable

    # Elemwise comparison - has out_ind=(0,)
    expected = np.array([1.0, 1.0, 1.0])
    close = da.isclose(result, expected)

    # Should fuse Elemwise (out_ind=(0,)) with Blockwise (out_ind=(2,))
    optimized = close.expr.optimize(fuse=True)
    assert isinstance(optimized, FusedBlockwise)

    # Verify correct computation
    assert_eq(close, np.array([True, True, True]))


def test_fusion_blockwise_multiblock_contracted_prevents_fusion():
    """Test that Blockwise with multi-block contracted dims isn't fusable.

    When a Blockwise has contracted dimensions (in input but not output) with
    multiple blocks, fusion is not possible since each output block would need
    to reference multiple input blocks from the contracted dimension.
    """
    from dask_array._blockwise import FusedBlockwise

    # Create 3D array with multiple blocks in contracted dimension 0
    arr_3d = da.from_array(np.ones((2, 1, 3)), chunks=(1, 1, 1))

    result = da.blockwise(
        lambda x: x.sum(),
        (2,),  # output indexed by dim 2
        arr_3d.expr,
        (0, 1, 2),
        dtype=arr_3d.dtype,
    )

    # Should NOT be fusable due to multi-block contracted dimension
    assert not result.expr._is_blockwise_fusable

    # Elemwise wrapping the Blockwise
    close = da.isclose(result, np.array([1.0, 1.0, 1.0]))

    # Should NOT fuse since Blockwise isn't fusable
    optimized = close.expr.optimize(fuse=True)
    assert not isinstance(optimized, FusedBlockwise)
