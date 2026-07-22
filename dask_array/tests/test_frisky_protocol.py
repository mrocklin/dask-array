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
    # Local import: chunks only exist when the Rust extension built them, but
    # this module must import without it (the pure-Python CI lane).
    from dask_array._rust import RECORDS_PROTOCOL_VERSION

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
        if tag == 5:
            return ("str", string())
        raise AssertionError(f"unknown slot tag {tag}")

    assert u8() == RECORDS_PROTOCOL_VERSION
    names = [string() for _ in range(u32())]
    dep_names = [string() for _ in range(u32())]
    for _ in range(u32()):
        take(u32())
    tasks = []
    for _ in range(u32()):
        name = names[u32()]
        task_coord = coord()
        expected_nbytes = i64()
        compute_tag = u8()
        if compute_tag == 0:
            compute = ("call", u32())
        elif compute_tag == 2:
            compute = ("alias",)
        else:
            raise AssertionError(f"unknown compute tag {compute_tag}")
        tasks.append(
            (
                name,
                task_coord,
                expected_nbytes,
                compute,
                tuple(slot(dep_names) for _ in range(u8())),
            )
        )
    assert pos == len(chunk)
    return names, dep_names, tasks


def _block_nbytes(chunks, coord, dtype):
    return int(np.prod([chunks[axis][i] for axis, i in enumerate(coord)]) * np.dtype(dtype).itemsize)


def _chunk_for_expr(collection, expr):
    chunks, _records, chunk_groups = collection.__frisky_records_chunks__()
    by_name = {name: chunk for chunk, (name, _meta, _upstream) in zip(chunks, chunk_groups)}
    return by_name[expr._name]


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


def test_creation_binary_chunk_carries_expected_nbytes():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.ones((5, 6), chunks=(2, 3), dtype="float32")

    _names, dep_names, tasks = _decode_layer_chunk(_chunk_for_expr(x, x._lowered_expr))

    assert dep_names == []
    assert {
        coord: expected_nbytes for _name, coord, expected_nbytes, compute, _slots in tasks if compute[0] == "call"
    } == {
        (0, 0): 2 * 3 * 4,
        (0, 1): 2 * 3 * 4,
        (1, 0): 2 * 3 * 4,
        (1, 1): 2 * 3 * 4,
        (2, 0): 1 * 3 * 4,
        (2, 1): 1 * 3 * 4,
    }


def test_blockwise_binary_chunk_carries_expected_nbytes():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.ones((5, 6), chunks=(2, 3), dtype="float32")
    y = x + np.float32(1)

    _names, _dep_names, tasks = _decode_layer_chunk(_chunk_for_expr(y, y._lowered_expr))

    assert {
        coord: expected_nbytes
        for name, coord, expected_nbytes, _compute, _slots in tasks
        if name == y._lowered_expr._name
    } == {
        (0, 0): 2 * 3 * 4,
        (0, 1): 2 * 3 * 4,
        (1, 0): 2 * 3 * 4,
        (1, 1): 2 * 3 * 4,
        (2, 0): 1 * 3 * 4,
        (2, 1): 1 * 3 * 4,
    }


def test_arange_binary_chunk_carries_expected_nbytes():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.arange(10, chunks=4, dtype="int32")

    _names, dep_names, tasks = _decode_layer_chunk(_chunk_for_expr(x, x._lowered_expr))

    assert dep_names == []
    assert {
        coord: expected_nbytes for _name, coord, expected_nbytes, compute, _slots in tasks if compute[0] == "call"
    } == {
        (0,): 4 * 4,
        (1,): 4 * 4,
        (2,): 2 * 4,
    }


def test_linspace_binary_chunk_carries_expected_nbytes():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.linspace(0, 1, 10, chunks=4, dtype="float32")

    _names, dep_names, tasks = _decode_layer_chunk(_chunk_for_expr(x, x._lowered_expr))

    assert dep_names == []
    assert {
        coord: expected_nbytes for _name, coord, expected_nbytes, compute, _slots in tasks if compute[0] == "call"
    } == {
        (0,): 4 * 4,
        (1,): 4 * 4,
        (2,): 2 * 4,
    }


def test_slicing_binary_chunk_carries_expected_nbytes_without_layer_plumbing():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.from_array(np.arange(6 * 7, dtype=np.int16).reshape(6, 7), chunks=(2, 3))
    y = x[1:6, 2:7]

    expr = y._lowered_expr
    _names, _dep_names, tasks = _decode_layer_chunk(_chunk_for_expr(y, expr))

    assert {
        coord: expected_nbytes for name, coord, expected_nbytes, _compute, _slots in tasks if name == expr._name
    } == {coord: _block_nbytes(expr.chunks, coord, expr.dtype) for coord in np.ndindex(expr.numblocks)}


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


def test_frisky_modules_route_rust_through_base_guard():
    """Package code must get ``_rust`` via ``dask_array._frisky.base``
    (``from dask_array._frisky.base import _rust``), whose import-time check
    fails loudly on a stale native build. Any other import of the extension
    bypasses that guard — from_array once had this hole, so a from_array-only
    walk called a stale extension silently. AST-based so every import spelling
    (aliased, parenthesized, relative) is caught; tests are exempt (they poke
    the extension deliberately)."""
    import ast
    import pathlib

    import dask_array

    pkg_root = pathlib.Path(dask_array.__file__).parent

    def rust_imports(path):
        # The package a relative import resolves against: for __init__.py the
        # module IS its package, for foo.py it's the containing package —
        # either way, drop the last path component.
        pkg_parts = ["dask_array", *path.relative_to(pkg_root).with_suffix("").parts[:-1]]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[:2] == ["dask_array", "_rust"] for a in node.names):
                    yield node
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    anchor = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                    target = ".".join(anchor + (node.module.split(".") if node.module else []))
                else:
                    target = node.module or ""
                if target.split(".")[:2] == ["dask_array", "_rust"] or (
                    target == "dask_array" and any(a.name == "_rust" for a in node.names)
                ):
                    yield node

    offenders = {}
    for path in sorted(pkg_root.rglob("*.py")):
        rel = path.relative_to(pkg_root)
        if rel.parts[0] == "tests" or rel.as_posix() == "_frisky/base.py":
            continue
        lines = [f"line {n.lineno}" for n in rust_imports(path)]
        if lines:
            offenders[rel.as_posix()] = lines
    assert not offenders, f"import _rust via dask_array._frisky.base, not directly: {offenders}"


def test_stale_native_build_fails_loudly_on_from_array_import():
    """A stale ``.so`` (Rust source changed, extension not rebuilt) must raise
    on import of any layer module — here the from_array module, which once
    bypassed the generation check in ``base`` — rather than silently calling
    the stale extension with possibly-changed argument conventions."""
    import subprocess

    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    code = (
        "import sys\n"
        "import dask_array._rust\n"
        "# Precondition: the guard must not have run yet, or patching the\n"
        "# generation below tests nothing (a failure here means dask_array\n"
        "# started importing _frisky eagerly, losing laziness).\n"
        "assert 'dask_array._frisky.base' not in sys.modules\n"
        "dask_array._rust.native_build_generation = lambda: -1  # simulate a stale build\n"
        "try:\n"
        "    import dask_array._frisky.from_array\n"
        "except ImportError as exc:\n"
        "    assert 'native build generation' in str(exc), exc\n"
        "else:\n"
        "    raise SystemExit('stale native extension was not detected')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_records_protocol_version_pinned_to_frisky_grammar():
    """``RECORDS_PROTOCOL_VERSION`` is a WIRE constant: it moves only in
    lockstep with Frisky's ``records_proto::CHUNK_GRAMMAR_VERSION``, when the
    chunk byte grammar itself changes (unlike the local build-freshness
    generation, which moves on any Rust change). This pin forces a human to
    acknowledge that cross-repo coordination on any bump."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    from dask_array._rust import RECORDS_PROTOCOL_VERSION

    assert RECORDS_PROTOCOL_VERSION == 3  # v3 added expected_nbytes


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
    for name, coord, expected_nbytes, compute, slots in tasks:
        assert name == fused._name
        assert expected_nbytes == _block_nbytes(fused.chunks, coord, fused.dtype)
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
    for name, coord, expected_nbytes, compute, slots in tasks:
        assert name == fused._name
        assert expected_nbytes == _block_nbytes(fused.chunks, coord, fused.dtype)
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
    for name, coord, expected_nbytes, compute, slots in tasks:
        assert name == fused._name
        assert expected_nbytes == _block_nbytes(fused.chunks, coord, fused.dtype)
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

    def per_block_dep_sets(layer, spec):
        # Decode through the Rust layer (the analytical path ships a compact
        # projection, the exact path a materialized list — both expand in Rust),
        # then read each output block's dependency set from the records.
        records = layer._build_rust(spec).to_task_records()
        return {key: sorted(deps) for key, _func, _args, _kw, deps in records}

    for build in (lambda x: x @ x, lambda x: x @ x.T, lambda x: x.T + x):
        layer = FusedBlockwiseLayer(fused(build(a)))
        analytical = layer._analytical_site_spec()
        assert analytical is not None  # the arithmetic (no per-block _task) path
        assert per_block_dep_sets(layer, analytical) == per_block_dep_sets(layer, layer._site_based_spec())

    # 'a' read at two always-coincident sites -> no all-distinct maximal block ->
    # analytical bails; the exact uniform/site-based path still handles it.
    b = da.from_array(np.arange(64.0).reshape(8, 8) + 100.0, chunks=(4, 4))
    fbc = [e for e in (a * b + a)._lowered_expr.walk() if type(e).__name__ == "FusedBlockwise"]
    assert fbc and FusedBlockwiseLayer(fbc[0])._analytical_site_spec() is None


def test_projected_fused_records_match_materialized_byte_for_byte():
    """The compact projection path (Rust generates each output block's dep coords
    and seed values by arithmetic) must produce records BYTE-IDENTICAL to
    materializing those same slots in Python and shipping them through the exact
    constructor. This pins the closed-form Rust encoder against a mis-encode — a
    drift in keys/dep-order/arg-encoding would cause cross-process key-agreement
    hangs, not just wrong bytes."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    from itertools import product

    from dask_array._frisky.fused_blockwise import _MatSpec, _ProjSpec, FusedBlockwiseLayer

    a = da.from_array(np.arange(96.0).reshape(8, 12), chunks=(4, 4))
    b = da.from_array(np.arange(144.0).reshape(12, 12), chunks=(4, 4))
    v = da.from_array(np.arange(12.0), chunks=(4,))
    ones = da.ones((12, 12), chunks=(4, 4))
    builds = {
        "elemwise": a + 1.0 + 2.0,  # two ops fuse into one FusedBlockwise
        "broadcast": (a + v) * 2.0,
        "transpose": a.T + 1.0,
        "matmul": a @ b,
        "gram": a @ a.T,
        "einsum": da.einsum("ij,jk->ik", a, b),
        "overlap_seed": ones.map_overlap(lambda x: x + 1, depth=1, boundary="none"),
        "overlap_reflect": ones.map_overlap(lambda x: x + 1, depth=1, boundary="reflect"),
        "diff_chunk_seed": da.diff(ones, axis=0),  # inner Ones shape -> chunk seed
    }
    checked = 0
    for label, coll in builds.items():
        fbs = [e for e in coll._lowered_expr.walk() if type(e).__name__ == "FusedBlockwise"]
        projected = 0
        for fb in fbs:
            layer = FusedBlockwiseLayer(fb)
            spec = layer._fast_spec()
            if not isinstance(spec, _ProjSpec):
                continue  # only the projected specs are exercised here
            projected += 1
            nb = tuple(int(n) for n in fb.numblocks)
            bids = list(product(*(range(n) for n in nb)))
            chunks = [dim if dim is not None else [] for dim in layer._axis_chunks()]
            dep_slots = [
                [(di, tuple(bid[co] if kind == "bid" else co for kind, co in pj)) for di, pj in spec.projections]
                for bid in bids
            ]
            seed_slots = (
                [[layer._apply_template(t, bid, chunks) for t in spec.seed_templates] for bid in bids]
                if spec.seed_templates
                else []
            )
            mat = _MatSpec(spec.shared, spec.dep_names, dep_slots, seed_slots)
            assert layer._build_rust(spec).to_records_chunk() == layer._build_rust(mat).to_records_chunk(), label
            checked += 1
        assert projected, f"{label} produced no projected FusedBlockwise"
    assert checked == len(builds)  # each shape reached the projected path exactly once


def test_seed_template_collapse_matches_materialized_for_crafted_shapes():
    """``from_projections`` compiles each seed template once (resolving the
    tuple/list collapse at build time) and evaluates it per block; the
    materialized ``new`` path runs ``_apply_template`` then ``seed_to_slot`` per
    block. They must agree byte-for-byte for EVERY template shape. Real dask
    graphs only reach the nested ``(block_id, numblocks)`` case, so this drives
    the crafted shapes (scalars, flat, empty, deeply-nested, mixed leaf +
    container) directly through both Rust constructors."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    from itertools import product

    import toolz

    from dask_array._frisky.base import _rust
    from dask_array._frisky.fused_blockwise import FusedBlockwiseLayer

    numblocks = [2, 3]
    dep_names = ["src"]
    func = toolz.identity  # any picklable callable; same object on both layers
    projections = [(0, [(1, 0), (1, 1)])]  # one site tracking both output dims
    bids = list(product(*(range(n) for n in numblocks)))
    dep_slots = [[(0, [b0, b1])] for b0, b1 in bids]
    # Per-axis output chunk sizes (ragged, so ``chunk`` leaves differ per block).
    chunk_tables = [[7, 4], [5, 5, 6]]

    def C(v):
        return ("const", v)

    def Bd(o):
        return ("bid", o)

    def Ch(a):
        return ("chunk", a)

    def T(*xs):
        return ("tuple", list(xs))

    def L(*xs):
        return ("list", list(xs))

    template_sets = [
        [],  # no seed
        [C(5)],  # scalar const
        [Bd(0)],  # scalar bid
        [Ch(0)],  # scalar chunk size
        [T(Bd(0), C(7))],  # flat int tuple
        [L(Bd(1), C(2))],  # flat list-kind (collapses to a tuple, like seed_to_slot)
        [T(Ch(0), Ch(1))],  # an inner Ones/full block-shape (chunk, chunk)
        [T()],  # empty tuple
        [L()],  # empty list
        [T(T(Bd(0), Bd(1)), T(C(2), C(3)))],  # (block_id, numblocks)
        [T(T(T(Bd(0))))],  # deep nesting
        [T(Bd(0), T(C(1), Ch(1)))],  # mixed leaf + subcontainer -> outer List
        [T(T(), C(4))],  # container holding an empty subcontainer
        [C(1), T(Ch(0), Ch(1))],  # multiple seeds incl. a block shape
    ]
    for templates in template_sets:
        seed_slots = [[FusedBlockwiseLayer._apply_template(t, bid, chunk_tables) for t in templates] for bid in bids]
        proj = _rust.FusedBlockwiseLayer.from_projections(
            "out", func, numblocks, dep_names, projections, templates, chunk_tables
        )
        mat = _rust.FusedBlockwiseLayer("out", func, numblocks, dep_names, dep_slots, seed_slots)
        assert proj.to_records_chunk() == mat.to_records_chunk(), templates

        # Loose-records converter too: normalize (TaskRef has no cross-instance
        # __eq__) and compare keys, arg reprs, kwargs, and dep strings.
        def norm(recs):
            return [(k, repr(a), kw, sorted(d)) for k, _f, a, kw, d in recs]

        assert norm(proj.to_task_records()) == norm(mat.to_task_records()), templates


def test_fast_path_block_independence_is_tokenize_free(monkeypatch):
    """The fused-block block-independence check must not tokenize (== cloudpickle)
    the shared subgraph func. Doing so once per probe block — via ``Task.__eq__``
    on canonicalized subgraphs — dominated scheduler-side graph build ~10x on large
    contractions (the shared func is a closure cloudpickle serializes by value,
    scanning ``sys.modules``). ``_canon_fingerprint`` / ``_hole_fingerprint``
    compare funcs by identity instead; this guards against a silent regression back
    onto ``tokenize``. Pure-Python (``_fast_spec``), so it needs no Rust extension."""
    import dask.tokenize

    from dask_array._frisky.fused_blockwise import FusedBlockwiseLayer

    def fused(coll):
        return FusedBlockwiseLayer([e for e in coll._lowered_expr.walk() if type(e).__name__ == "FusedBlockwise"][0])

    a = da.from_array(np.arange(64.0).reshape(8, 8), chunks=(4, 4))
    d = da.ones((12, 12), chunks=(4, 4))
    layers = {
        "contraction (Gram A@A.T)": fused(a @ a.T),  # analytical / exact fast paths
        "seed (map_overlap block_id)": fused(d.map_overlap(lambda b: b + 1, depth=1, boundary="none")),
    }

    calls = {"n": 0}
    real = dask.tokenize.tokenize

    def counting_tokenize(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    # ``Task.__eq__`` and ``Task._get_token`` both re-resolve ``tokenize`` off these
    # modules at call time, so patching here intercepts every Task tokenization.
    monkeypatch.setattr(dask.tokenize, "tokenize", counting_tokenize)
    monkeypatch.setattr("dask.base.tokenize", counting_tokenize)

    for label, layer in layers.items():
        calls["n"] = 0
        spec = layer._fast_spec()
        assert spec is not None, f"{label} should take a binary fast path"
        assert calls["n"] == 0, f"{label}: block-independence check tokenized {calls['n']}x"


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


@pytest.mark.parametrize("reduction, keepdims", [("sum", False), ("nanmean", False), ("min", True)])
def test_sliding_window_reduction_uses_binary_records(reduction, keepdims):
    """The native-chunk banded sliding-window reduction (long rolling windows)
    emits one binary chunk — per output block a reduce task over its own block,
    a list of middle-block totals, the right-edge band blocks, and two scalar
    slots — instead of falling to the legacy-graph adapter."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    rng = np.random.default_rng(0)
    data = rng.normal(size=(96, 6))
    data[rng.random(data.shape) < 0.2] = np.nan
    x = da.from_array(data, chunks=((7, 12, 9, 14, 8, 12, 6, 12, 16), (4, 2)))
    view = da.sliding_window_view(x, window_shape=20, axis=0)
    y = getattr(da, reduction)(view, axis=-1, keepdims=keepdims)

    expr = [e for e in y._lowered_expr.walk() if type(e).__name__ == "SlidingWindowReduction"][0]
    assert expr._frisky_layer().to_records_chunk()  # binary, not the adapter

    chunks, records, chunk_groups = y.__frisky_records_chunks__()
    assert expr._name in {name for name, _meta, _up in chunk_groups}
    assert not any(r[0].startswith(f"('{expr._name}'") for r in records)

    # binary chunk decodes to the banded task shape: per output block
    # (block, [middle totals], [band blocks], out_len, band_offset)
    _names, _dep_names, tasks = _decode_layer_chunk(_chunk_for_expr(y, expr))
    out_tasks = [t for t in tasks if t[0] == expr._name]
    assert len(out_tasks) == len(list(flatten(expr.__dask_keys__())))
    assert any(t[0] == f"{expr._name}-total" for t in tasks)  # window spans middle blocks
    for _name, _coord, _nbytes, compute, slots in out_tasks:
        assert compute == ("call", 0)
        dep, mids, band, out_len, band_offset = slots
        assert dep[:2] == ("dep", expr.array._name)
        assert mids[0] == "list" and all(s[:2] == ("dep", f"{expr._name}-total") for s in mids[1])
        assert band[0] == "list" and band[1] and all(s[:2] == ("dep", expr.array._name) for s in band[1])
        assert out_len[0] == "scalar" and band_offset[0] == "scalar"

    # native records graph is value-identical to the legacy graph
    dep_graph = {}
    for dep in expr.dependencies():
        dep_graph.update(dict(dep.__dask_graph__()))
    legacy_graph = expr._layer()
    native_graph = expr._frisky_layer().to_dask_graph()
    for key in flatten(expr.__dask_keys__()):
        expected = get_sync({**dep_graph, **legacy_graph}, key)
        actual = get_sync({**dep_graph, **native_graph}, key)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("min_count", [1, None])
def test_moving_window_reduction_uses_binary_records(min_count):
    """The native-chunk trailing-window reduction (xarray's bottleneck rolling
    path) emits one binary chunk, including the array-start block with no band
    (empty list slots) and truncated leading windows."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    bn = pytest.importorskip("bottleneck")
    rng = np.random.default_rng(0)
    data = rng.normal(size=(96, 4))
    data[rng.random(data.shape) < 0.2] = np.nan
    x = da.from_array(data, chunks=((7, 12, 9, 14, 8, 12, 6, 12, 16), (4,)))
    y = x.map_overlap(bn.move_sum, depth={0: (19, 0)}, dtype="f8", window=20, min_count=min_count, axis=0)

    expr = [e for e in y._lowered_expr.walk() if type(e).__name__ == "MovingWindowReduction"][0]
    assert expr._frisky_layer().to_records_chunk()  # binary, not the adapter

    chunks, records, chunk_groups = y.__frisky_records_chunks__()
    assert expr._name in {name for name, _meta, _up in chunk_groups}
    assert not any(r[0].startswith(f"('{expr._name}'") for r in records)

    # binary chunk decodes to the banded task shape; the array-start block has
    # no band and no middles (empty list slots)
    _names, _dep_names, tasks = _decode_layer_chunk(_chunk_for_expr(y, expr))
    out_tasks = [t for t in tasks if t[0] == expr._name]
    assert len(out_tasks) == len(list(flatten(expr.__dask_keys__())))
    for _name, coord, _nbytes, compute, slots in out_tasks:
        assert compute == ("call", 0)
        dep, mids, band, n_trunc, band_offset = slots
        assert dep[:2] == ("dep", expr.array._name)
        assert mids[0] == "list" and band[0] == "list"
        assert n_trunc[0] == "scalar" and band_offset[0] == "scalar"
        if coord[0] == 0:
            assert mids[1] == () and band[1] == ()
        else:
            assert band[1] and all(s[:2] == ("dep", expr.array._name) for s in band[1])

    dep_graph = {}
    for dep in expr.dependencies():
        dep_graph.update(dict(dep.__dask_graph__()))
    legacy_graph = expr._layer()
    native_graph = expr._frisky_layer().to_dask_graph()
    for key in flatten(expr.__dask_keys__()):
        expected = get_sync({**dep_graph, **legacy_graph}, key)
        actual = get_sync({**dep_graph, **native_graph}, key)
        np.testing.assert_array_equal(actual, expected)


def test_banded_layer_gate_drift_degrades_instead_of_panicking():
    """The Rust banded layers validate the plan invariants the Python gates
    guarantee (band past the block's own start/end).  If gate and layer ever
    drift, the constructor raises NotImplementedError — which the collect walk
    turns into the adapter tier — rather than panicking mid-emission."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    from dask_array.reductions._sliding_window import MovingWindowReduction, SlidingWindowReduction

    x = da.from_array(np.arange(36.0), chunks=((3, 30, 3),))
    # both violate the gates: a chunk larger than the window
    moving = MovingWindowReduction(x.expr, 5, 1, 0, "nansum", np.dtype("f8"))
    with pytest.raises(NotImplementedError):
        moving._frisky_layer()
    sliding = SlidingWindowReduction(x.expr, 10, 0, 1, False, "sum", np.dtype("f8"))
    with pytest.raises(NotImplementedError):
        sliding._frisky_layer()


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


@pytest.mark.parametrize("multi", [False, True])
def test_gufunc_leaf_uses_binary_records(multi):
    """The output-splitting leaf of ``apply_gufunc`` goes binary — an alias per
    block (single output) or ``getitem(block, i)`` (multiple outputs) — instead of
    the per-task ``GraphRecordsLayer`` adapter."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    a = da.ones((20, 30), chunks=(10, 30))
    if multi:
        out = da.apply_gufunc(lambda x: (np.mean(x, -1), np.std(x, -1)), "(i)->(),()", a, output_dtypes=(float, float))[
            0
        ]
    else:
        out = da.apply_gufunc(lambda x: np.sum(x, -1), "(i)->()", a, output_dtypes=float)

    leaf = [e for e in out._lowered_expr.walk() if type(e).__name__ == "GUfuncLeafExpr"][0]
    assert leaf._frisky_layer().to_records_chunk()  # binary, not the adapter

    chunks, records, chunk_groups = out.__frisky_records_chunks__()
    ops = [json.loads(m)["op"] for _, m, _ in chunk_groups]
    assert "GUfuncLeafExpr" in ops
    assert not any(r[0].startswith(f"('{leaf._name}'") for r in records)


@pytest.mark.parametrize(
    "shape, chunks_spec, axis",
    [
        ((8, 4), (4, 4), 1),
        ((8, 4), (4, 2), (0, 3)),
        ((12,), (4,), 0),
        ((6, 6), (3, 3), (0, 2, 4)),
    ],
)
def test_expand_dims_uses_binary_records(shape, chunks_spec, axis):
    """expand_dims carries its expansion positions as an int-tuple ``axis`` slot
    for ``np.expand_dims`` rather than an opaque ``None``-bearing getitem indexer
    (a ``Literal`` the binary grammar can't express), so the layer emits one
    records chunk instead of N Python tuples -- and still matches numpy."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    base = np.arange(int(np.prod(shape)), dtype="f8").reshape(shape)
    x = da.from_array(base, chunks=chunks_spec)
    y = da.expand_dims(x, axis)

    chunks, records, chunk_groups = y.__frisky_records_chunks__()
    ops = [json.loads(m)["op"] for _, m, _ in chunk_groups]
    assert "ExpandDims" in ops  # went binary, not to plain records
    ed = [e for e in y._lowered_expr.walk() if type(e).__name__ == "ExpandDims"][0]
    assert not any(r[0].startswith(f"('{ed._name}',") for r in records)

    # native records graph is value-identical to the legacy graph, and to numpy.
    dep_graph = {}
    for dep in ed.dependencies():
        dep_graph.update(dict(dep.__dask_graph__()))
    legacy_graph = ed._layer()
    native_graph = ed._frisky_layer().to_dask_graph()
    for key in flatten(ed.__dask_keys__()):
        expected = get_sync({**dep_graph, **legacy_graph}, key)
        actual = get_sync({**dep_graph, **native_graph}, key)
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.asarray(y), np.expand_dims(base, axis))


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
    # block); the seed path handles it, lifting the ``(block_id, numblocks)``
    # literal into a per-block seed. Decode through Rust and check each output
    # block's seed (the last arg, after the dep refs). ``(block_id, numblocks)``
    # is nested, so the records grammar renders it as a list of int tuples.
    layer = FusedBlockwiseLayer(fb)
    assert layer._analytical_site_spec() is None
    spec = layer._seed_spec()
    assert spec is not None
    nb = tuple(int(n) for n in fb.numblocks)
    args_by_key = {key: args for key, _func, args, _kw, _deps in layer._build_rust(spec).to_task_records()}
    for bid in product(*(range(n) for n in nb)):
        assert args_by_key[str((fb._name, *bid))][-1] == [bid, nb]


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda x: da.diff(x, axis=0), id="diff-axis0-ragged-first"),
        pytest.param(lambda x: da.diff(x, axis=1), id="diff-axis1"),
        pytest.param(lambda x: da.diff(da.diff(x, axis=0), axis=1), id="diff-both"),
    ],
)
def test_diff_lifts_chunk_shape_seed_to_binary_records(build):
    """A fused block that bakes an inner Ones/full whose SHAPE is the output block's
    chunk shape declines the block-independent and affine fast paths — the shape
    differs at a ragged chunk boundary and is not an affine function of the block
    id. ``da.diff`` is the canonical case (it makes a ragged first chunk, so the
    inner ``ones`` shape is e.g. ``(39, 40)`` at the boundary and ``(40, 40)``
    elsewhere). ``_seed_spec`` lifts that shape to a per-block ``chunk`` seed
    (``chunks[a][bid[a]]``), so the layer goes binary instead of the O(N)
    ``_slow_records`` path — while computing identically to dask's own graph."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    from itertools import product

    from dask.core import flatten
    from dask.local import get_sync

    from dask_array._frisky.fused_blockwise import _ProjSpec, FusedBlockwiseLayer

    y = build(da.ones((240, 240), chunks=(40, 40)))
    fb = [e for e in y._lowered_expr.walk() if type(e).__name__ == "FusedBlockwise"][0]
    layer = FusedBlockwiseLayer(fb)

    # Block-independent + affine paths decline; the seed path lifts a chunk leaf.
    assert layer._analytical_site_spec() is None
    spec = layer._seed_spec()
    assert isinstance(spec, _ProjSpec)
    assert any("chunk" in repr(t) for t in spec.seed_templates)
    assert isinstance(layer.to_records_chunk(), bytes)  # binary, not a decline

    # Each block's decoded seed is that block's chunk shape.
    args_by_key = {key: args for key, _f, args, _kw, _d in layer._build_rust(spec).to_task_records()}
    for bid in product(*(range(n) for n in fb.numblocks)):
        want = tuple(fb.chunks[d][bid[d]] for d in range(len(bid)))
        assert args_by_key[str((fb._name, *bid))][-1] == want

    # End-to-end: the Rust-decoded graph computes identically to dask's own.
    full = dict(y.__dask_graph__())
    spliced = {**full, **layer._build_rust(spec).to_dask_graph()}
    for key in flatten(fb.__dask_keys__()):
        assert np.array_equal(get_sync(full, key), get_sync(spliced, key))


def test_xarray_rolling_sum_where_literal_uses_binary_records():
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import json

    xr = pytest.importorskip("xarray")
    if not _xarray_sliding_window_uses_chunk_manager():
        pytest.skip("requires xarray sliding_window_view dispatch through the chunk manager array API")

    da.xarray.register()  # rolling dispatches through the chunk manager

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
    ref = xr.DataArray(np.ones((100, 5)), dims=("time", "asset")).rolling({"time": 10}, min_periods=1).sum().data
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
        for name, _coord, _expected_nbytes, compute, slots in tasks
        if name.endswith("-extra") and compute[0] == "call" and slots and slots[0][0] == "inttuple"
    )
    assert identity_shapes == sorted(expected_identity_shapes)

    expr = y._lowered_expr
    assert {
        coord: expected_nbytes
        for name, coord, expected_nbytes, _compute, _slots in tasks
        if name == expr._name and len(coord) == len(expr.chunks)
    } == {coord: _block_nbytes(expr.chunks, coord, expr.dtype) for coord in np.ndindex(expr.numblocks)}


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


def _blockinfo_block(block, block_info=None):
    # A block_info consumer shaped like pipeline/store.py's day writer: it reads
    # the output chunk-location off block_info and folds it into the result.
    loc = block_info[None]["chunk-location"][0]
    return block + loc


def test_block_info_map_blocks_uses_binary_records(monkeypatch):
    """``map_blocks`` with a ``block_info=`` consumer must ride the binary records
    path. ``block_info`` arrives as an ``ArrayValuesDep`` whose per-block value is a
    nested dict — not expressible as a binary slot on its own — so we emit the block
    coord as an ``IntTuple`` and bind the ``{coord: block_info}`` map into a lookup
    shim. ``block_id=`` (a plain int tuple) already went binary; this closes the
    ``block_info`` gap that left the pipeline's day writer generating Python task
    records."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")

    x = da.from_array(np.arange(24.0).reshape(6, 4), chunks=(2, 4))
    y = x.map_blocks(_blockinfo_block, dtype=x.dtype)

    expr = y._lowered_expr
    bw = [e for e in expr.walk() if type(e).__name__ == "Blockwise"][0]
    assert bw._frisky_layer().to_records_chunk()  # goes binary, no decline

    # No task for the Blockwise layer falls to the plain records tail — it is fully
    # binary — and nothing hits the stock-dask adapter.
    chunks, records, chunk_groups = y.__frisky_records_chunks__()
    assert not any(r[0].startswith(f"('{bw._name}'") for r in records)
    assert _fallback_expr_counts(monkeypatch, y) == {}

    # The native layer's own expansion (which drives the binary records) reconstructs
    # each block's block_info through _BlockwiseDepLookup and matches the stock-dask
    # graph block for block. Exercising to_dask_graph here covers the shim's runtime
    # in the default (non-Frisky) test env, where ``compute`` would otherwise route
    # around it through the stock ``_layer``.
    dep_graph = dict(x.__dask_graph__())
    legacy_graph = bw._layer()
    native_graph = bw._frisky_layer().to_dask_graph()
    for key in flatten(bw.__dask_keys__()):
        expected = get_sync({**dep_graph, **legacy_graph}, key)
        actual = get_sync({**dep_graph, **native_graph}, key)
        np.testing.assert_array_equal(actual, expected)

    # And it still computes the right thing (chunk-location folded into each block).
    expected = np.arange(24.0).reshape(6, 4)
    expected[2:4] += 1
    expected[4:6] += 2
    np.testing.assert_array_equal(y.compute(scheduler="synchronous"), expected)


def test_block_info_map_blocks_binary_records_end_to_end(array_scheduler):
    """The same block_info consumer computes correctly through the real Frisky
    records submission path (not just the synchronous graph)."""
    x = da.from_array(np.arange(24.0).reshape(6, 4), chunks=(2, 4))
    y = x.map_blocks(_blockinfo_block, dtype=x.dtype)
    expected = np.arange(24.0).reshape(6, 4)
    expected[2:4] += 1
    expected[4:6] += 2
    np.testing.assert_array_equal(y.compute(), expected)


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
