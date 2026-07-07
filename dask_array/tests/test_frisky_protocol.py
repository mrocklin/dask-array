from __future__ import annotations

import importlib.util
import inspect
import struct
import sys

import dask
from dask.core import flatten
from dask.local import get_sync
import numpy as np
import pytest

import dask_array as da
from dask_array._new_collection import new_collection
from dask_array._overlap import overlap_internal


def _decode_layer_chunk(chunk):
    pos = 0

    def take(n):
        nonlocal pos
        out = chunk[pos : pos + n]
        pos += n
        return out

    def u8():
        return take(1)[0]

    def u32():
        return struct.unpack("<I", take(4))[0]

    def i64():
        return struct.unpack("<q", take(8))[0]

    def coord():
        return tuple(u32() for _ in range(u8()))

    def string():
        return take(u32()).decode()

    def opt_i64():
        return i64() if u8() else None

    def slot(dep_names):
        tag = u8()
        if tag == 0:
            return ("dep", dep_names[u32()], coord())
        if tag == 1:
            items = []
            for _ in range(u8()):
                elem_tag = u8()
                if elem_tag == 0:
                    items.append(("slice", opt_i64(), opt_i64(), opt_i64()))
                else:
                    items.append(("int", i64()))
            return ("index", tuple(items))
        if tag == 2:
            return ("inttuple", tuple(i64() for _ in range(u8())))
        if tag == 3:
            return ("list", tuple(slot(dep_names) for _ in range(u32())))
        if tag == 4:
            num_tag = u8()
            return ("scalar", i64() if num_tag == 0 else struct.unpack("<d", take(8))[0])
        raise AssertionError(f"unknown slot tag {tag}")

    assert u8() == 1
    names = [string() for _ in range(u32())]
    dep_names = [string() for _ in range(u32())]
    for _ in range(u32()):
        take(u32())
    tasks = []
    for _ in range(u32()):
        name = names[u32()]
        task_coord = coord()
        compute_tag = u8()
        if compute_tag == 0:
            compute = ("call", u32())
        elif compute_tag == 2:
            compute = ("alias",)
        else:
            raise AssertionError(f"unknown compute tag {compute_tag}")
        tasks.append((name, task_coord, compute, tuple(slot(dep_names) for _ in range(u8()))))
    assert pos == len(chunk)
    return names, dep_names, tasks


def _xarray_sliding_window_uses_chunk_manager():
    try:
        import xarray.compat.dask_array_compat as compat
    except ImportError:
        return False

    try:
        source = inspect.getsource(compat.sliding_window_view)
    except OSError:
        return False
    return "get_chunked_array_type" in source and ".array_api" in source


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


def test_numeric_scalar_materialized_graph_uses_binary_alias_and_fused_chunk():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    x = new_collection((da.ones((4, 4), chunks=(2, 2)) + 1).expr.optimize(fuse=False))

    chunks, records, chunk_groups = x.__frisky_records_chunks__()

    assert records == []
    assert len(chunks) == 2
    assert [json.loads(meta)["op"] for _, meta, _ in chunk_groups] == ["RootAlias", "FusedBlockwise"]


def test_fused_blockwise_uses_binary_records_chunk():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    x = da.ones((40, 40), chunks=(10, 10))
    y = (x + 1) * 2 - x / 3

    chunks, records, chunk_groups = y.__frisky_records_chunks__()

    assert records == []
    assert len(chunks) == 2
    assert [json.loads(meta)["op"] for _, meta, _ in chunk_groups] == ["RootAlias", "FusedBlockwise"]


def test_source_backed_fused_blockwise_binary_chunk_tracks_deps():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    x = da.from_array(np.arange(40 * 40).reshape(40, 40), chunks=(10, 10))
    y = (x + 1) * 2 - x / 3

    chunks, records, chunk_groups = y.__frisky_records_chunks__()

    assert records
    assert len(chunks) == 2
    assert [json.loads(meta)["op"] for _, meta, _ in chunk_groups] == ["RootAlias", "FusedBlockwise"]

    _names, dep_names, tasks = _decode_layer_chunk(chunks[1])
    fused = y._lowered_expr.dependencies()[0]
    source = fused.dependencies()[0]
    assert dep_names == [source._name]
    assert len(tasks) == 16
    for name, coord, compute, slots in tasks:
        assert name == fused._name
        assert compute[0] == "call"
        assert slots == (("dep", source._name, coord),)


def test_transposed_fused_blockwise_binary_chunk_tracks_remapped_deps():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    base = da.from_array(np.ones((20, 4)), chunks=(5, 2))
    x = base.rechunk((4, 2))
    y = x.T + 1

    chunks, records, chunk_groups = y.__frisky_records_chunks__()

    assert len(chunks) == 2
    assert [json.loads(meta)["op"] for _, meta, _ in chunk_groups] == ["RootAlias", "FusedBlockwise"]

    _names, dep_names, tasks = _decode_layer_chunk(chunks[1])
    fused = y._lowered_expr.dependencies()[0]
    source = fused.dependencies()[0]
    assert not any(record[0].startswith(f"('{fused._name}',") for record in records)
    assert dep_names == [source._name]
    assert len(tasks) == 10
    for name, coord, compute, slots in tasks:
        assert name == fused._name
        assert compute[0] == "call"
        assert slots == (("dep", source._name, (coord[1], coord[0])),)


def test_contracted_einsum_fused_blockwise_binary_chunk_tracks_remapped_deps():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    a = da.from_array(np.arange(3 * 4 * 5).reshape(3, 4, 5), chunks=(1, 2, 5)).rechunk((1, 1, 5))
    b = da.from_array(np.arange(3 * 5 * 6).reshape(3, 5, 6), chunks=(1, 5, 3)).rechunk((1, 5, 2))
    y = da.einsum("iab,ibc->iac", a, b)

    chunks, records, chunk_groups = y.__frisky_records_chunks__()

    assert len(chunks) == 3
    assert [json.loads(meta)["op"] for _, meta, _ in chunk_groups] == [
        "RootAlias",
        "PartialReduce",
        "FusedBlockwise",
    ]

    _names, dep_names, tasks = _decode_layer_chunk(chunks[2])
    fused = y._lowered_expr.dependencies()[0].dependencies()[0]
    a_dep, b_dep = fused.dependencies()
    a_source, b_source = a_dep._name, b_dep._name
    assert not any(record[0].startswith(f"('{fused._name}',") for record in records)
    assert set(dep_names) == {a_source, b_source}
    assert len(tasks) == 36
    for name, coord, compute, slots in tasks:
        assert name == fused._name
        assert compute[0] == "call"
        # The slot *order* is a deterministic implementation detail (sites are
        # ordered by (source, coord)); verify the coord remapping as a set.
        assert set(slots) == {
            ("dep", a_source, (coord[0], coord[1], coord[3])),
            ("dep", b_source, (coord[0], coord[3], coord[2])),
        }


@pytest.mark.parametrize(
    "label, build",
    [
        ("transpose_plus_self", lambda x: x.T + x),
        ("self_matmul", lambda x: x @ x),
        ("gram", lambda x: x @ x.T),
    ],
)
def test_repeated_operand_fused_blockwise_uses_binary_records(label, build):
    """A fused block that reads the SAME source more than once — Gram matrices
    (``A @ A.T``), ``x.T + x``, self-matmul — must still take the binary records
    path. The block structure is fine: distinct-source contraction (see the
    einsum test above) is already binary. Today ``_fast_spec`` labels each fused
    input by its source *name*, so two reads of one source collide and it bails
    to the O(N)-Python-tuple ``_slow_records``. This pins the fast path back on.

    Note the diagonal wrinkle these cases carry: where the two reads of the one
    source land on the *same* block (``i == j`` in a Gram matrix), dask dedups
    them to a single dependency, so per-block fan-in varies (2 off-diagonal, 1
    on) — the fast path must reproduce that, not just relabel."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    from dask_array._frisky.fused_blockwise import FusedBlockwiseLayer

    x = da.from_array(np.arange(16.0).reshape(4, 4), chunks=(2, 2))
    y = build(x)
    fused = [e for e in y._lowered_expr.walk() if type(e).__name__ == "FusedBlockwise"]
    assert len(fused) == 1
    layer = FusedBlockwiseLayer(fused[0])

    # (1) Faithful: the fast (binary) records reproduce the slow path's
    # per-block key -> {upstream deps} exactly, including the deduped diagonal.
    fast = layer._fast_records()
    assert fast is not None, "repeated-operand fused block should take the binary/fast path"

    def key_deps(recs):
        return {key: sorted({str(d) for d in deps}) for key, _f, _a, _k, deps in recs}

    assert key_deps(fast) == key_deps(layer._slow_records())

    # (2) Engages the binary encoder (bytes, not a NotImplementedError fall-through).
    assert isinstance(layer.to_records_chunk(), bytes)


def test_contractions_use_analytical_derivation():
    """matmul / Gram / transpose+self engage the O(1)-per-block analytical
    projection derivation, not the O(N) per-block exact fallback — a guard against
    a silent perf regression back onto per-block ``_task()``. The analytical slots
    match the value-verified exact ``_site_based_spec`` (as per-block dep sets),
    and a fused block that reads one source at two always-coincident sites
    (``a*b+a``) correctly bails to the exact path."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    from dask_array._frisky.fused_blockwise import FusedBlockwiseLayer

    a = da.from_array(np.arange(64.0).reshape(8, 8), chunks=(4, 4))

    def fused(coll):
        return [e for e in coll._lowered_expr.walk() if type(e).__name__ == "FusedBlockwise"][0]

    def per_block_dep_sets(spec):
        return [sorted({str(s) for s in slots}) for slots in spec[2]]

    for build in (lambda x: x @ x, lambda x: x @ x.T, lambda x: x.T + x):
        layer = FusedBlockwiseLayer(fused(build(a)))
        analytical = layer._analytical_site_spec()
        assert analytical is not None  # the arithmetic (no per-block _task) path
        assert per_block_dep_sets(analytical) == per_block_dep_sets(layer._site_based_spec())

    # 'a' read at two always-coincident sites -> no all-distinct maximal block ->
    # analytical bails; the exact uniform/site-based path still handles it.
    b = da.from_array(np.arange(64.0).reshape(8, 8) + 100.0, chunks=(4, 4))
    fbc = [e for e in (a * b + a)._lowered_expr.walk() if type(e).__name__ == "FusedBlockwise"]
    assert fbc and FusedBlockwiseLayer(fbc[0])._analytical_site_spec() is None


def test_cumulative_over_unknown_chunks_uses_binary_records():
    """A cumulative reduction over unknown (nan) chunk sizes generates a complete
    binary records graph rather than declining. The sequential plan is fixed by
    the block *count* (numblocks); the one place a size is used — the ``extra``
    identity block's shape — is forced to 1 along the reduction axis and otherwise
    only needs to broadcast, so an unknown size maps to 1 instead of crashing
    ``int(nan)``. Record generation must complete without a bare-ValueError crash."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    x = da.from_array(np.arange(20.0), chunks=5)
    y = x[x > 7].cumsum(axis=0)  # genuinely unknown (nan) chunk sizes
    assert any(np.isnan(s) for dim in y.chunks for s in dim)

    cum = [e for e in y._lowered_expr.walk() if type(e).__name__ == "CumReduction"][0]
    assert cum._frisky_layer().to_records_chunk()  # goes binary, no decline, no crash

    # Record generation completes and the CumReduction went binary — its group is
    # present among the binary chunks and no CumReduction task fell to the plain
    # records tail. (nan-chunk layers carry no JSON metadata, so match the layer
    # name, not the op.)
    chunks, records, chunk_groups = y.__frisky_records_chunks__()
    assert cum._name in {name for name, _meta, _up in chunk_groups}
    assert not any(r[0].startswith(f"('{cum._name}'") for r in records)


def test_blelloch_cumulative_uses_binary_records():
    """The Blelloch parallel-scan cumulative (``method="blelloch"``) has a native
    binary layer — the preop batches, the upsweep/downsweep combine tree, and the
    prefix-scan outputs all emit as one binary chunk instead of running on the
    ``GraphRecordsLayer`` adapter."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    y = da.cumsum(da.ones((40,), chunks=5), axis=0, method="blelloch")  # 8 axis chunks
    bl = [e for e in y._lowered_expr.walk() if type(e).__name__ == "CumReductionBlelloch"][0]
    assert bl._frisky_layer().to_records_chunk()  # binary, not the adapter

    chunks, records, chunk_groups = y.__frisky_records_chunks__()
    assert bl._name in {name for name, _meta, _up in chunk_groups}
    assert not any(r[0].startswith(f"('{bl._name}'") for r in records)


def test_unknown_chunks_run_on_frisky_not_stock_dask():
    """Unknown (nan) chunk *sizes* no longer force the whole graph to stock dask.
    Records are keyed by block coordinate and numblocks is known even when sizes
    are not, so a boolean-mask graph is fully static and generates a complete
    records graph — only ops that truly need concrete sizes decline per-layer."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.from_array(np.arange(20.0), chunks=5)
    y = x[x > 7] + 1
    assert any(np.isnan(s) for dim in y.chunks for s in dim)  # genuinely unknown sizes

    y._check_frisky_supported()  # no longer raises on nan chunks

    chunks, records, _chunk_groups = y.__frisky_records_chunks__()
    assert chunks or records  # a real graph was generated, not a stock-dask fallback


def test_from_array_getter_uses_binary_records_chunk():
    """An array-like (zarr/h5py/…) from_array puts its N getter tasks on the binary
    records path and ships the source array ONCE as a plain holder record, instead
    of N Python getter tuples. The holder is keyed ('original-<name>',) so each
    getter task references it with a Dep(empty coord) slot — no grammar change."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    class Lazy:  # a slicing target that is NOT a plain ndarray -> the getter path
        def __init__(self, a):
            self.a, self.shape, self.dtype, self.ndim = a, a.shape, a.dtype, a.ndim

        def __getitem__(self, i):
            return self.a[i]

    src = Lazy(np.arange(48.0).reshape(6, 8))
    x = da.from_array(src, chunks=(3, 4))  # 6 getter blocks

    chunks, records, chunk_groups = x.__frisky_records_chunks__()

    ops = [json.loads(m)["op"] for _, m, _ in chunk_groups]
    assert "FromArray" in ops  # the getter tasks went binary, not to plain records
    # The only plain record is the shared holder — no per-block getter tuples.
    assert len(records) == 1
    holder = records[0]
    assert holder[0] == str((f"original-{x.name}",))
    assert holder[1].__name__ == "identity" and holder[2][0] is src and holder[4] == []


def test_arg_reduction_chunk_uses_binary_records():
    """The per-block chunk step of an arg reduction (argmax/argmin) goes binary.
    It carries the shared reduction ``axis`` as an int-tuple slot rather than a
    ``Literal`` (which the binary grammar can't express), so ArgChunk emits one
    records chunk instead of N Python getter tuples."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    y = da.ones((20, 20), chunks=(5, 5)).argmax(axis=1)  # 4x4 chunk-step blocks

    chunks, records, chunk_groups = y.__frisky_records_chunks__()

    ops = [json.loads(m)["op"] for _, m, _ in chunk_groups]
    assert "ArgChunk" in ops  # the chunk step went binary, not to plain records
    argchunk = [e for e in y._lowered_expr.walk() if type(e).__name__ == "ArgChunk"][0]
    assert not any(r[0].startswith(f"('{argchunk._name}',") for r in records)


def test_map_overlap_lifts_block_id_seed_to_binary_records():
    """A fused map_overlap block bakes its own ``block_id`` into the subgraph (the
    ``_trim`` ``(block_id, numblocks)`` literal), so the block-independent fast
    paths correctly decline. ``_seed_spec`` lifts that literal into a per-block
    seed, making the fused layer binary — a guard against silently regressing back
    onto the O(N) ``_slow_records`` path for stencil/overlap graphs."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json
    from itertools import product

    from dask_array._frisky.fused_blockwise import FusedBlockwiseLayer

    y = da.ones((12, 12), chunks=(4, 4)).map_overlap(lambda b: b + 1, depth=1, boundary="none")
    fb = [e for e in y._lowered_expr.walk() if type(e).__name__ == "FusedBlockwise"][0]

    chunks, records, chunk_groups = y.__frisky_records_chunks__()
    ops = [json.loads(m)["op"] for _, m, _ in chunk_groups]
    assert "FusedBlockwise" in ops  # the fused trim went binary, not to plain records
    assert not any(r[0].startswith(f"('{fb._name}',") for r in records)

    # The block-independent exact paths decline (the baked block_id differs per
    # block); the seed path handles it, emitting one (block_id, numblocks) seed.
    layer = FusedBlockwiseLayer(fb)
    assert layer._analytical_site_spec() is None
    _shared, _dep_names, _dep_slots, seed_slots = layer._seed_spec()
    nb = fb.numblocks
    for bid, seeds in zip(product(*(range(n) for n in nb)), seed_slots):
        assert seeds == [(bid, nb)]


def test_xarray_rolling_sum_where_literal_uses_binary_records():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    xr = pytest.importorskip("xarray")
    if not _xarray_sliding_window_uses_chunk_manager():
        pytest.skip("requires xarray sliding_window_view dispatch through the chunk manager array API")

    x = da.from_array(np.ones((100, 5)), chunks=(20, 5))
    xda = xr.DataArray(x, dims=("time", "asset"))
    y = xda.rolling({"time": 10}, min_periods=1).sum().data

    chunks, records, chunk_groups = y.__frisky_records_chunks__()

    # The rolling sum lowers to overlap + a where-with-literal (min_periods mask)
    # + trim. With the overlap input properly simplified, the where-elemwise
    # fuses into the FusedBlockwise trim node, so every compute layer rides the
    # binary records path end to end -- the only Python records are the
    # from_array data source. (This test is the direct binary coverage for a
    # where-with-literal -- now via the fused node; the *_scalar_fused_blockwise
    # tests cover the same literal-slot mechanism for other scalar ops.)
    assert chunks
    ops = [json.loads(meta)["op"] for _, meta, _ in chunk_groups]
    assert "FusedBlockwise" in ops
    # no compute layer falls back: every Python record is a from_array source block
    assert all("array-" in record[0] for record in records)

    fused = [e for e in y._lowered_expr.walk() if type(e).__name__ == "FusedBlockwise"]
    assert fused
    for fb in fused:
        assert fb._frisky_layer().to_records_chunk()  # the fused where-literal goes binary
        assert not any(record[0].startswith(f"('{fb._name}',") for record in records)

    # the binary path is numerically identical to computing on numpy.
    ref = (
        xr.DataArray(np.ones((100, 5)), dims=("time", "asset"))
        .rolling({"time": 10}, min_periods=1)
        .sum()
        .data
    )
    np.testing.assert_array_equal(np.asarray(y), np.asarray(ref))


@pytest.mark.parametrize(
    "axis, shape, chunk_spec, expected_identity_shapes",
    [
        (0, (40, 4), (10, 4), [(1, 4)]),
        (1, (12, 10), ((5, 7), (2, 3, 5)), [(5, 1), (7, 1)]),
    ],
)
def test_cumreduction_uses_binary_records_chunk(axis, shape, chunk_spec, expected_identity_shapes):
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    x = da.ones(shape, chunks=chunk_spec)
    y = x.cumsum(axis=axis)

    chunks, records, chunk_groups = y.__frisky_records_chunks__()

    assert records == []
    assert len(chunks) == 2
    assert [json.loads(meta)["op"] for _, meta, _ in chunk_groups] == ["CumReduction", "Ones"]

    _names, _dep_names, tasks = _decode_layer_chunk(chunks[0])
    identity_shapes = sorted(
        slots[0][1]
        for name, _coord, compute, slots in tasks
        if name.endswith("-extra") and compute[0] == "call" and slots and slots[0][0] == "inttuple"
    )
    assert identity_shapes == sorted(expected_identity_shapes)


def test_sliding_window_overlap_uses_binary_records():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    x = da.sliding_window_view(da.ones(12, chunks=4), 3).sum()

    chunks, records, chunk_groups = x.__frisky_records_chunks__()

    assert chunks
    assert records == []
    ops = {json.loads(meta)["op"] for _, meta, _ in chunk_groups}
    assert {"RootAlias", "PartialReduce", "FusedBlockwise", "OverlapInternal", "Ones"} <= ops


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_stack_uses_binary_records(axis):
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    x = da.ones((4, 6), chunks=(2, 3))
    y = new_collection(da.stack([x, x + 1], axis=axis).expr.optimize(fuse=False))

    _chunks, records, chunk_groups = y.__frisky_records_chunks__()

    assert records == []
    assert "Stack" in {json.loads(meta)["op"] for _, meta, _ in chunk_groups}


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_stack_native_layer_matches_legacy_graph(axis):
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.from_array(np.arange(4 * 6).reshape(4, 6), chunks=(2, 3))
    y = da.stack([x, x + 100], axis=axis)
    expr = y.expr
    dep_graph = {}
    for dep in expr.dependencies():
        dep_graph.update(dict(dep.__dask_graph__()))
    legacy_graph = expr._layer()
    native_graph = expr._frisky_layer().to_dask_graph()

    for key in flatten(expr.__dask_keys__()):
        expected = get_sync({**dep_graph, **legacy_graph}, key)
        actual = get_sync({**dep_graph, **native_graph}, key)
        np.testing.assert_array_equal(actual, expected)


def test_chunk_groups_carry_name_and_metadata():
    """Each binary chunk ships its producing expr's ``_name`` (the stable layer
    identity, which a key prefix can't always recover), an opaque JSON blob of
    op/shape/chunks/dtype, and its upstream group names (the child layers' ``_name``s
    == the layer-DAG edges), parallel to ``chunks``."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    x = new_collection((da.ones((4, 4), chunks=(2, 2)) + 1).expr.optimize(fuse=False))
    chunks, records, chunk_groups = x.__frisky_records_chunks__()

    assert len(chunk_groups) == len(chunks) == 2
    for name, meta, upstream in chunk_groups:
        assert isinstance(name, str) and name  # the expr `_name`
        assert isinstance(upstream, list)  # upstream group names
        info = json.loads(meta)
        assert info["shape"] == [4, 4]
        assert info["chunks"] == [[2, 2], [2, 2]]
        assert info["numblocks"] == [2, 2]
        assert info["dtype"]  # e.g. "float64"
        assert "op" in info
        assert isinstance(info["params"], dict)  # scalar params, not array inputs

    # The submitted graph is materialized before record collection, so the root
    # group is the stable output-key alias and its upstream is the fused producer.
    info = json.loads(chunk_groups[0][1])
    assert info["op"] == "RootAlias"
    assert chunk_groups[0][2] == [x._lowered_expr.dependencies()[0]._name]
    assert json.loads(chunk_groups[1][1])["op"] == "FusedBlockwise"

    params = info["params"]
    assert params["name"] == x.name


def test_summarize_chunks_bounds_finely_chunked_dims():
    """A dim with many chunks is summarized, not listed — so the metadata blob
    can't blow up on a 100k-chunk array."""
    from dask_array._frisky.collect import _MAX_CHUNKS_PER_DIM, _summarize_chunks

    # Few chunks -> listed in full.
    assert _summarize_chunks([(2, 2, 2)]) == [[2, 2, 2]]
    # Many chunks -> compact {nchunks, min, max}, not the full list.
    out = _summarize_chunks([tuple([1] * 1000), (5, 5)])
    assert out[0] == {"nchunks": 1000, "min": 1, "max": 1}
    assert out[1] == [5, 5]
    # Varied chunk sizes -> distinct min/max (a min/max swap would be caught).
    assert _summarize_chunks([tuple(range(1, 21))])[0] == {"nchunks": 20, "min": 1, "max": 20}
    # Boundary: == max lists, > max summarizes.
    assert _summarize_chunks([tuple(range(1, _MAX_CHUNKS_PER_DIM + 1))])[0] == list(range(1, _MAX_CHUNKS_PER_DIM + 1))
    assert isinstance(_summarize_chunks([tuple(range(_MAX_CHUNKS_PER_DIM + 1))])[0], dict)


def test_layer_metadata_stays_small_for_many_chunks():
    """End-to-end: a finely-chunked array yields a tiny metadata blob."""
    import json

    from dask_array._frisky.collect import _layer_metadata

    x = da.ones((4000,), chunks=1)  # 4000 chunks in one dim
    meta = _layer_metadata(x.expr)
    assert meta is not None
    info = json.loads(meta)
    assert info["chunks"][0] == {"nchunks": 4000, "min": 1, "max": 1}
    assert len(meta) < 1000, f"blob should stay small, got {len(meta)} bytes"


def test_layer_metadata_drops_oversized_params(monkeypatch):
    """A pathological nested `params` can't bloat the blob: when the assembled
    metadata exceeds the cap, `params` is dropped but op/shape/chunks/dtype
    survive — so the scheduler never has to drop the whole (useful) blob."""
    import json

    from dask_array._frisky import collect as collect_mod
    from dask_array._frisky.collect import _MAX_METADATA_BYTES, _layer_metadata

    monkeypatch.setattr(collect_mod, "_expr_params", lambda e: {"big": "x" * (_MAX_METADATA_BYTES * 2)})
    meta = _layer_metadata(da.ones((4,), chunks=2).expr)
    assert meta is not None
    assert len(meta) <= _MAX_METADATA_BYTES
    info = json.loads(meta)
    assert "params" not in info  # dropped...
    assert info["op"] and info["shape"] == [4] and info["dtype"]  # ...but these survive


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


def test_bool_scalar_fused_blockwise_uses_binary_records():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    x = new_collection((da.ones((4, 4), chunks=(2, 2)) == True).expr.optimize(fuse=False))  # noqa: E712

    chunks, records, chunk_groups = x.__frisky_records_chunks__()

    assert records == []
    assert len(chunks) == 2
    assert [json.loads(meta)["op"] for _, meta, _ in chunk_groups] == ["RootAlias", "FusedBlockwise"]


def test_numpy_scalar_fused_blockwise_uses_binary_records():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    scalar = np.float64(1)
    x = new_collection((da.ones((4, 4), chunks=(2, 2), dtype="float32") + scalar).expr.optimize(fuse=False))

    chunks, records, chunk_groups = x.__frisky_records_chunks__()

    assert records == []
    assert len(chunks) == 2
    assert [json.loads(meta)["op"] for _, meta, _ in chunk_groups] == ["RootAlias", "FusedBlockwise"]


def test_large_int_scalar_fused_blockwise_uses_binary_records():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    scalar = 10**20
    x = new_collection((da.ones((4, 4), chunks=(2, 2)) + scalar).expr.optimize(fuse=False))

    chunks, records, chunk_groups = x.__frisky_records_chunks__()

    assert records == []
    assert len(chunks) == 2
    assert [json.loads(meta)["op"] for _, meta, _ in chunk_groups] == ["RootAlias", "FusedBlockwise"]


def _constant_block():
    return np.full((2, 2), 7.0)


def _constant_frisky_records_chunks(self, seen=None):
    return [], [(self.__frisky_output_keys__()[0], _constant_block, (), {}, [])], []


def _unexpected_frisky_graph(self, seen=None):
    raise AssertionError("Frisky should use __frisky_records_chunks__ before __frisky_graph__")


def test_frisky_scheduler_uses_records_chunks_protocol(array_scheduler, monkeypatch):
    if array_scheduler != "frisky":
        pytest.skip("requires --scheduler=frisky")

    x = da.ones((2, 2), chunks=(2, 2)) + 1
    monkeypatch.setattr(type(x), "__frisky_records_chunks__", _constant_frisky_records_chunks)
    monkeypatch.setattr(type(x), "__frisky_graph__", _unexpected_frisky_graph)

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


def test_persist_name_preserving_lifecycle(array_scheduler):
    """Name-preserving persist under frisky, exercised repeatedly.

    A persisted collection's futures are borrowed for later computes of the
    same keys, so a) gathering the same handle more than once must work
    (regression: ``Client.gather`` dropped a key's notify after delivery, so a
    second gather of that handle hung forever), and b) a transient compute of
    the original collection must not drop the persisted lease (regression:
    the scheduler's per-client desire is boolean, so the client counts its
    handles per key and only releases on the last one)."""
    if array_scheduler != "frisky":
        pytest.skip("requires --scheduler=frisky")

    x = da.ones((4, 4), chunks=(2, 2)) + 1
    xp = x.persist()
    assert xp.name == x.name

    # Repeated computes of the persisted collection gather the same borrowed
    # handles each time.
    np.testing.assert_array_equal(xp.compute(), np.full((4, 4), 2.0))
    np.testing.assert_array_equal(xp.compute(), np.full((4, 4), 2.0))

    # A transient compute of the ORIGINAL collection resubmits the same keys;
    # its release must not invalidate the persisted lease.
    np.testing.assert_array_equal(x.compute(), np.full((4, 4), 2.0))
    np.testing.assert_array_equal(xp.compute(), np.full((4, 4), 2.0))

    # Persisting again keeps the same identity and stays computable.
    xp2 = xp.persist()
    assert xp2.name == x.name
    np.testing.assert_array_equal(xp2.compute(), np.full((4, 4), 2.0))
    np.testing.assert_array_equal((xp2 + 1).compute(), np.full((4, 4), 3.0))
