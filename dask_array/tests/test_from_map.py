"""Behavior spec for ``from_map`` and the Stack/Concatenate -> FromMap rewrite.

``from_map`` builds an array by calling ``func`` once per block. The per-block
arguments live in a numpy *object array whose shape is the block grid*
(``values[idx]`` is the argument for block ``idx``), so the primitive is N-D
native and -- crucially -- merging two ``from_map``s is just ``np.concatenate`` /
``np.stack`` on their ``values`` grids.

Because each block is a *single* task under a tuple key ``(name, *idx)``, the
whole layer groups cleanly for Frisky (no opaque delayed bodies floating
ungrouped) and the expression is one node instead of N single-chunk
``FromDelayed`` nodes plus a combining layer.

Two compositional ``_simplify_down`` steps fold the common delayed pattern in:

  * normalize: a ``FromDelayed`` with a single-call body -> a 1-block ``FromMap``.
  * merge:     ``stack``/``concatenate`` of ``FromMap``s -> one ``FromMap``
               (``np.stack``/``np.concatenate`` the ``values`` grids + chunks).

The merge fires for ``concatenate(from_map)`` / ``stack(from_map)`` regardless of
whether the ``FromMap`` children were normalized from ``FromDelayed`` or built
directly by the user. ``simplify`` runs to a fixpoint, so nested
``concatenate(stack(...))`` collapses in stages with no bespoke nesting code.

These tests are the alignment artifact: they pin the behavior before FromMap /
from_map / the rewrite exist, so they are RED until implemented.
"""

from __future__ import annotations

import importlib.util

import dask
import numpy as np
import pytest

import dask_array as da
from dask_array._test_utils import assert_eq
from dask_array._frisky.collect import collect_task_records
from dask_array.io._from_map import FromMap


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _frisky_group(key_str):
    """Mirror Frisky's ``group_name_from_key`` (frisky-core ``types.rs``): a
    tuple-repr key ``('name', ...)`` groups under ``name``; a bare string key is
    ungrouped (returns ``None``). This is exactly how the scheduler key-derives a
    group for a residual (non-native-layer) record, so asserting on it here tells
    us what Frisky would show without a running scheduler."""
    s = key_str
    if not s.startswith("("):
        return None
    s = s[1:].lstrip()
    if not s or s[0] not in "'\"":
        return None
    quote = s[0]
    s = s[1:]
    end = s.find(quote)
    if end <= 0:
        return None
    return s[:end]


def _groups(arr):
    """The set of Frisky groups the array's records would land in. ``None`` in
    the set means at least one task is ungrouped (the delayed-body problem)."""
    recs = collect_task_records(arr)
    return {_frisky_group(k) for k, *_ in recs}


def _obj(values):
    """Build an object ndarray without numpy trying to broadcast the cells."""
    a = np.empty(len(values), dtype=object)
    a[:] = list(values)
    return a


def _load(val):
    """A single-call block body: one function, literal arg, no sub-tasks."""
    return np.full(5, val, dtype="int64")


def _load_multi():
    """A delayed value whose body is NOT a single call (root op depends on two
    other delayed tasks). The rewrite must decline this and leave it untouched."""
    a = dask.delayed(np.ones)(5)
    b = dask.delayed(np.zeros)(5)
    return dask.delayed(lambda x, y: (x + y).astype("int64"))(a, b)


# ---------------------------------------------------------------------------
# from_map: the primitive
# ---------------------------------------------------------------------------


def test_from_map_values_and_structure():
    a = da.from_map(_load, _obj([1, 2, 3]), chunks=((5, 5, 5),), dtype="int64")
    assert a.shape == (15,)
    assert a.chunks == ((5, 5, 5),)
    expected = np.concatenate([np.full(5, 1), np.full(5, 2), np.full(5, 3)]).astype("int64")
    assert_eq(a, expected)


def test_from_map_is_one_grouped_layer_no_ungrouped_tasks():
    a = da.from_map(_load, _obj([1, 2, 3]), chunks=((5, 5, 5),), dtype="int64")
    recs = collect_task_records(a)
    # exactly one task per block -- no lifted sub-tasks, no opaque bodies
    assert len(recs) == 3
    groups = {_frisky_group(k) for k, *_ in recs}
    assert None not in groups  # nothing floats ungrouped
    assert len(groups) == 1  # one clean layer
    assert next(iter(groups)).startswith("from-map")


def test_from_map_passes_constant_kwargs():
    def scaled(val, *, scale=1):
        return np.full(5, val * scale, dtype="int64")

    a = da.from_map(scaled, _obj([1, 2, 3]), chunks=((5, 5, 5),), dtype="int64", scale=10)
    expected = np.concatenate([np.full(5, 10), np.full(5, 20), np.full(5, 30)]).astype("int64")
    assert_eq(a, expected)


def test_from_map_over_2d_block_grid():
    """The values array's shape IS the block grid -- N-D falls out directly."""

    def make(val):
        return np.full((2, 3), val, dtype="int64")

    values = np.empty((2, 2), dtype=object)
    values[:] = [[1, 2], [3, 4]]
    a = da.from_map(make, values, chunks=((2, 2), (3, 3)), dtype="int64")
    assert a.shape == (4, 6)
    assert a.numblocks == (2, 2)
    expected = np.block([[np.full((2, 3), 1), np.full((2, 3), 2)], [np.full((2, 3), 3), np.full((2, 3), 4)]]).astype(
        "int64"
    )
    assert_eq(a, expected)
    assert None not in _groups(a)


def test_from_map_rejects_reordering_shape_mismatch():
    """A func whose output has the chunk's element count but a permuted shape
    must fail loudly, not get silently reshaped (reordered) into place."""
    values = np.empty((1, 1), dtype=object)
    values[0, 0] = 0

    def bad(_):
        return np.arange(6).reshape(3, 2)  # chunk is (2, 3): same size, wrong order

    a = da.from_map(bad, values, chunks=((2,), (3,)), dtype="int64")
    with pytest.raises(ValueError, match="incompatible with the declared chunk shape"):
        a.compute()


def test_from_map_over_3d_block_grid():
    """A 3-D values grid -> a 3-D array, one func call per block."""

    def make(val):
        return np.full((2, 2, 2), val, dtype="int64")

    values = np.empty((2, 2, 2), dtype=object)
    values[:] = np.arange(8).reshape(2, 2, 2)  # distinct value per block
    a = da.from_map(make, values, chunks=((2, 2), (2, 2), (2, 2)), dtype="int64")
    assert a.shape == (4, 4, 4)
    assert a.numblocks == (2, 2, 2)

    expected = np.empty((4, 4, 4), dtype="int64")
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected[i * 2 : (i + 1) * 2, j * 2 : (j + 1) * 2, k * 2 : (k + 1) * 2] = int(values[i, j, k])
    assert_eq(a, expected)
    assert None not in _groups(a)


# ---------------------------------------------------------------------------
# normalize + merge: stack/concatenate of single-call FromDelayed -> FromMap
# ---------------------------------------------------------------------------


def test_concatenate_of_from_delayed_becomes_from_map():
    pieces = [da.from_delayed(dask.delayed(_load)(v), (5,), dtype="int64") for v in [1, 2, 3]]
    arr = da.concatenate(pieces)
    simplified = arr.simplify().expr
    assert isinstance(simplified, FromMap)
    assert list(simplified.dependencies()) == []  # a pure source, no residual layers

    expected = np.concatenate([np.full(5, v) for v in [1, 2, 3]]).astype("int64")
    assert_eq(arr, expected)
    assert None not in _groups(arr)  # zero ungrouped delayed bodies


def test_stack_of_from_delayed_becomes_from_map():
    pieces = [da.from_delayed(dask.delayed(_load)(v), (5,), dtype="int64") for v in [1, 2, 3]]
    arr = da.stack(pieces)  # new axis: (3, 5)
    simplified = arr.simplify().expr
    assert isinstance(simplified, FromMap)

    expected = np.stack([np.full(5, v) for v in [1, 2, 3]]).astype("int64")
    assert arr.shape == (3, 5)
    assert_eq(arr, expected)
    # The per-block reshape (5,)->(1,5) must stay INSIDE the block task, else it
    # gets lifted to a `<key>-subN` string key and floats ungrouped.
    assert None not in _groups(arr)


def test_stack_grouping_beats_the_unoptimized_baseline():
    """Sanity: the grouping metric really does detect ungrouped tasks, so the
    'None not in groups' assertions above can't pass vacuously. A multi-task
    delayed body is never normalized, so it stays ungrouped; the single-call
    pattern, by contrast, is fully grouped after the rewrite."""
    multi = da.stack([da.from_delayed(_load_multi(), (5,), dtype="int64") for _ in range(2)])
    assert None in _groups(multi)  # opaque bodies float ungrouped

    single = da.stack([da.from_delayed(dask.delayed(_load)(v), (5,), dtype="int64") for v in [1, 2, 3]])
    assert None not in _groups(single)  # rewrite fully groups the single-call pattern


# ---------------------------------------------------------------------------
# merge: stack/concatenate of FromMap children (built directly, not via delayed)
# ---------------------------------------------------------------------------


def test_concatenate_of_from_map_merges_into_one():
    a = da.from_map(_load, _obj([1, 2]), chunks=((5, 5),), dtype="int64")
    b = da.from_map(_load, _obj([3, 4]), chunks=((5, 5),), dtype="int64")
    arr = da.concatenate([a, b])
    simplified = arr.simplify().expr
    assert isinstance(simplified, FromMap)
    assert list(simplified.dependencies()) == []

    expected = np.concatenate([np.full(5, v) for v in [1, 2, 3, 4]]).astype("int64")
    assert_eq(arr, expected)
    assert None not in _groups(arr)


def test_stack_of_from_map_merges_into_one():
    a = da.from_map(_load, _obj([1, 2]), chunks=((5, 5),), dtype="int64")  # (10,)
    b = da.from_map(_load, _obj([3, 4]), chunks=((5, 5),), dtype="int64")  # (10,)
    arr = da.stack([a, b])  # (2, 10)
    simplified = arr.simplify().expr
    assert isinstance(simplified, FromMap)
    assert arr.shape == (2, 10)

    row = lambda x, y: np.concatenate([np.full(5, x), np.full(5, y)])
    expected = np.stack([row(1, 2), row(3, 4)]).astype("int64")
    assert_eq(arr, expected)
    assert None not in _groups(arr)


# ---------------------------------------------------------------------------
# fixpoint: nested concatenate(stack(...)) -> single FromMap
# ---------------------------------------------------------------------------


def test_nested_concatenate_of_stacks_collapses_to_one_from_map():
    def block(vals):
        return da.stack([da.from_delayed(dask.delayed(_load)(v), (5,), dtype="int64") for v in vals])

    arr = da.concatenate([block([1, 2]), block([3, 4])])  # (4, 5)
    simplified = arr.simplify().expr
    assert isinstance(simplified, FromMap)
    assert list(simplified.dependencies()) == []

    expected = np.stack([np.full(5, v) for v in [1, 2, 3, 4]]).astype("int64")
    assert arr.shape == (4, 5)
    assert_eq(arr, expected)
    assert None not in _groups(arr)


def test_nested_stacks_build_3d_from_map():
    """Nested stacks add two new axes -> (2, 2, 5). Each block must apply BOTH
    unit-axis reshapes inside a single grouped task."""

    def leaf(v):
        return da.from_delayed(dask.delayed(_load)(v), (5,), dtype="int64")

    arr = da.stack([da.stack([leaf(1), leaf(2)]), da.stack([leaf(3), leaf(4)])])
    simplified = arr.simplify().expr
    assert isinstance(simplified, FromMap)
    assert list(simplified.dependencies()) == []
    assert arr.shape == (2, 2, 5)

    expected = np.stack([np.stack([np.full(5, 1), np.full(5, 2)]), np.stack([np.full(5, 3), np.full(5, 4)])]).astype(
        "int64"
    )
    assert_eq(arr, expected)
    assert None not in _groups(arr)


def test_expand_dims_folds_into_from_map():
    """A unit-axis expansion folds into the FromMap source (new (1,) chunks +
    unit dims in the values grid), staying a single grouped layer."""
    a = da.from_map(_load, _obj([1, 2, 3]), chunks=((5, 5, 5),), dtype="int64")  # (15,)
    b = da.expand_dims(a, 0)  # (1, 15)
    simplified = b.simplify().expr
    assert isinstance(simplified, FromMap)
    assert b.shape == (1, 15)

    expected = np.concatenate([np.full(5, v) for v in [1, 2, 3]]).astype("int64")[None, :]
    assert_eq(b, expected)
    assert None not in _groups(b)


def test_mixed_rank_block_collapses_to_one_from_map():
    """da.block over lower-rank pieces wraps each source in expand_dims; that folds
    into FromMap, so even the mixed-rank block collapses to one grouped layer."""

    def leaf1d(v):
        return da.from_delayed(dask.delayed(np.full)((3,), v, dtype="int64"), (3,), dtype="int64")

    arr = da.block([[leaf1d(1)], [leaf1d(2)]])  # 1-D leaves -> (2, 3)
    simplified = arr.simplify().expr
    assert isinstance(simplified, FromMap)
    assert arr.shape == (2, 3)

    expected = np.block([[np.full((3,), 1)], [np.full((3,), 2)]]).astype("int64")
    assert_eq(arr, expected)
    assert None not in _groups(arr)


def test_block_of_from_delayed_collapses_to_one_from_map():
    """da.block is sugar over nested concatenate, so a uniform-ndim block grid of
    single-call from_delayed pieces folds into one FromMap for free (the fixpoint
    merges innermost concatenates outward). No block-specific rule needed."""

    def leaf(v):
        return da.from_delayed(dask.delayed(np.full)((2, 3), v, dtype="int64"), (2, 3), dtype="int64")

    arr = da.block([[leaf(1), leaf(2)], [leaf(3), leaf(4)]])
    simplified = arr.simplify().expr
    assert isinstance(simplified, FromMap)
    assert list(simplified.dependencies()) == []
    assert arr.shape == (4, 6)

    expected = np.block([[np.full((2, 3), 1), np.full((2, 3), 2)], [np.full((2, 3), 3), np.full((2, 3), 4)]]).astype(
        "int64"
    )
    assert_eq(arr, expected)
    assert None not in _groups(arr)


# ---------------------------------------------------------------------------
# decline: don't over-fire on multi-task delayed bodies
# ---------------------------------------------------------------------------


def test_named_from_delayed_output_key_is_preserved():
    """A user-supplied name pins the output key, so a named from_delayed is left
    as built (not renamed onto a FromMap). An unnamed one normalizes freely."""
    a = da.from_delayed(dask.delayed(_load)(7), (5,), dtype="int64", name="myblock")
    assert a._lowered_expr._name == "myblock"  # key survives optimization
    assert not isinstance(a._lowered_expr, FromMap)
    assert_eq(a, np.full(5, 7).astype("int64"))

    unnamed = da.from_delayed(dask.delayed(_load)(7), (5,), dtype="int64")
    # Unnamed still normalizes to FromMap; materialization pins the renamed
    # root back to the collection's keys with a RootAlias wrapper.
    assert isinstance(unnamed._lowered_expr.array, FromMap)


def test_multi_task_delayed_body_is_left_untouched():
    pieces = [da.from_delayed(_load_multi(), (5,), dtype="int64") for _ in range(2)]
    arr = da.stack(pieces)
    simplified = arr.simplify().expr
    assert not isinstance(simplified, FromMap)  # rule declined -> still Stack/FromDelayed

    expected = np.stack([np.ones(5) for _ in range(2)]).astype("int64")
    assert_eq(arr, expected)  # and still correct


def test_merge_declines_when_func_differs():
    """Sibling FromMaps with different funcs can't share one layer -> no merge,
    but the concatenation is still correct."""
    a = da.from_map(lambda v: np.full(5, v, dtype="int64"), _obj([1, 2]), chunks=((5, 5),), dtype="int64")
    b = da.from_map(lambda v: np.full(5, v * 100, dtype="int64"), _obj([3, 4]), chunks=((5, 5),), dtype="int64")
    arr = da.concatenate([a, b])
    assert not isinstance(arr.simplify().expr, FromMap)

    expected = np.concatenate([np.full(5, 1), np.full(5, 2), np.full(5, 300), np.full(5, 400)]).astype("int64")
    assert_eq(arr, expected)


def test_merge_declines_when_kwargs_differ():
    def scaled(v, *, s=1):
        return np.full(5, v * s, dtype="int64")

    a = da.from_map(scaled, _obj([1, 2]), chunks=((5, 5),), dtype="int64", s=1)
    b = da.from_map(scaled, _obj([3, 4]), chunks=((5, 5),), dtype="int64", s=10)
    arr = da.concatenate([a, b])
    assert not isinstance(arr.simplify().expr, FromMap)

    expected = np.concatenate([np.full(5, 1), np.full(5, 2), np.full(5, 30), np.full(5, 40)]).astype("int64")
    assert_eq(arr, expected)


# ---------------------------------------------------------------------------
# edge cases and error paths
# ---------------------------------------------------------------------------


def test_from_map_scalar_return_supports_0d_block():
    """A 0-d block whose func returns a bare Python scalar is accepted (coerced),
    not rejected as a shape mismatch."""
    values = np.empty((), dtype=object)
    values[()] = 7
    a = da.from_map(lambda v: v * 2, values, chunks=(), dtype="int64")
    assert a.shape == ()
    assert_eq(a, np.asarray(14, dtype="int64"))


def test_from_map_rejects_values_shape_mismatch():
    with pytest.raises(ValueError, match="block grid"):
        da.from_map(_load, _obj([1, 2]), chunks=((5, 5, 5),), dtype="int64")  # 2 values, 3 blocks


def test_from_map_requires_chunks():
    with pytest.raises(ValueError, match="chunks"):
        da.from_map(_load, _obj([1, 2, 3]), dtype="int64")


# ---------------------------------------------------------------------------
# native (Rust) FromMapLayer: parity with the generic fallback
# ---------------------------------------------------------------------------


def _norm_records(recs):
    """Comparable form: key string, func name, repr(args), repr(kwargs), deps."""
    return sorted(
        (str(k), getattr(f, "__name__", repr(f)), repr(a), repr(kw), sorted(str(d) for d in deps))
        for k, f, a, kw, deps in recs
    )


def test_native_from_map_layer_records_match_fallback():
    """The Rust FromMapLayer must emit exactly the records the generic
    GraphRecordsLayer fallback does -- same keys, func, args, kwargs, deps."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    from dask_array._frisky.graph_records import GraphRecordsLayer

    vals2d = np.empty((2, 2), dtype=object)
    vals2d[:] = [[1, 2], [3, 4]]
    zerod = np.empty((), dtype=object)
    zerod[()] = 7
    cases = {
        "1d": da.from_map(_load, _obj([1, 2, 3]), chunks=((5, 5, 5),), dtype="int64"),
        "2d": da.from_map(lambda v: np.full((2, 3), v, dtype="int64"), vals2d, chunks=((2, 2), (3, 3)), dtype="int64"),
        "0d": da.from_map(lambda v: v, zerod, chunks=(), dtype="int64"),
        "kwargs": da.from_map(lambda v, *, s=1: np.full(5, v * s), _obj([1, 2]), chunks=((5, 5),), dtype="int64", s=3),
        "folded-stack": da.stack([da.from_delayed(dask.delayed(_load)(v), (5,), dtype="int64") for v in [1, 2, 3]]),
    }
    for label, coll in cases.items():
        expr = coll._lowered_expr
        native = expr._frisky_layer().to_task_records()
        fallback = GraphRecordsLayer(expr).to_task_records()
        assert _norm_records(native) == _norm_records(fallback), label


def test_native_from_map_layer_graph_matches_legacy():
    """Per-block values from the native graph equal the legacy _layer graph.
    Uses a NON-SQUARE grid built from a transposed (non-contiguous) values view
    with distinct per-cell values -- this exercises the load-bearing C-order
    ravel and would catch a row/column-swap that a symmetric contiguous grid
    hides."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    from dask.core import flatten
    from dask.local import get_sync

    base = np.empty((3, 2), dtype=object)
    base[:] = np.arange(6).reshape(3, 2) * 10
    vals = base.T  # (2, 3), non-C-contiguous
    assert vals.shape == (2, 3) and not vals.flags["C_CONTIGUOUS"]
    a = da.from_map(lambda v: np.full((2, 4), v, dtype="int64"), vals, chunks=((2, 2), (4, 4, 4)), dtype="int64")
    expr = a._lowered_expr
    legacy = expr._layer()
    native = expr._frisky_layer().to_dask_graph()
    for key in flatten(a.__dask_keys__()):
        np.testing.assert_array_equal(get_sync(native, key), get_sync(legacy, key))


def test_collect_selects_native_from_map_layer(monkeypatch):
    """collect_task_records must route FromMap through the native FromMapLayer,
    not the generic GraphRecordsLayer fallback -- guards the _frisky_layer wiring
    (a regression there is silent, since the fallback is parity-equal)."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    from dask_array._frisky import collect as collect_mod
    from dask_array._frisky.graph_records import GraphRecordsLayer

    fell_back = []

    class SpyGraphRecordsLayer(GraphRecordsLayer):
        def __init__(self, expr):
            fell_back.append(type(expr).__name__)
            super().__init__(expr)

    monkeypatch.setattr(collect_mod, "GraphRecordsLayer", SpyGraphRecordsLayer)
    a = da.from_map(_load, _obj([1, 2, 3]), chunks=((5, 5, 5),), dtype="int64")
    collect_mod.collect_task_records(a)
    assert "FromMap" not in fell_back


# ---------------------------------------------------------------------------
# binary (pure-Rust) FromMap layer: coalesced delayed calls that share one
# function and carry only binary-expressible per-block args (scalars, strings
# like filenames, int-tuples, and lists thereof) emit a native binary records
# CHUNK -- no per-block Python task tuples. Anything else falls back to the
# plain-records FromMapLayer (and logs why).
# ---------------------------------------------------------------------------


# A store keyed by "filename": the common delayed pattern is one loader function
# called per block with a *string* path. This pins that path onto the binary
# grammar (a per-block string arg), not just the scalar-index case.
_FAKE_FILES = {"a.npy": 1, "bb.npy": 2, "ccc.npy": 3, "dddd.npy": 4}


def _load_named(path):
    """One shared loader, called once per block with a filename string."""
    return np.full(5, _FAKE_FILES[path], dtype="int64")


def _load_arr(block):
    """A loader whose per-block arg is an ndarray -- NOT binary-expressible, so
    the coalesced FromMap must decline the binary path and fall back."""
    return block.astype("int64")


def _read_scaled(path, storage_options):
    """A loader taking a POSITIONAL storage_options-style dict alongside the
    varying path. When that dict is uniform across blocks it should be baked into
    the shared pickled partial (once), not block the binary path."""
    return np.full(5, len(path) * storage_options["scale"], dtype="int64")


def _the_from_map(arr):
    """The single merged FromMap in a (lowered) collection's expression."""
    fms = [e for e in arr._lowered_expr.walk() if isinstance(e, FromMap)]
    assert len(fms) == 1, f"expected one FromMap, got {len(fms)}"
    return fms[0]


def _from_map_block_records(arr, fm):
    """The plain (Python-tuple) records whose key belongs to this FromMap layer.
    Empty when the layer went binary (its blocks live in a records CHUNK)."""
    _chunks, records, _groups = arr.__frisky_records_chunks__()
    return [r for r in records if r[0].startswith(f"('{fm._name}'")]


def test_concatenated_delayeds_with_scalar_args_use_binary_records():
    """The headline case: many `from_delayed(delayed(f)(scalar))` concatenated
    merge to one FromMap that emits a pure-Rust binary chunk -- zero per-block
    Python task tuples."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    pieces = [da.from_delayed(dask.delayed(_load)(v), (5,), dtype="int64") for v in [1, 2, 3]]
    arr = da.concatenate(pieces)
    fm = _the_from_map(arr)

    # (1) The layer engages the binary encoder (bytes, not a NotImplementedError).
    assert isinstance(fm._frisky_layer().to_records_chunk(), bytes)

    # (2) End to end: the FromMap's blocks land in a binary chunk group, and NO
    # FromMap block shows up as a plain Python-tuple record.
    _chunks, _records, chunk_groups = arr.__frisky_records_chunks__()
    assert fm._name in {name for name, _m, _u in chunk_groups}
    assert _from_map_block_records(arr, fm) == []

    assert_eq(arr, np.concatenate([np.full(5, v) for v in [1, 2, 3]]).astype("int64"))


def test_concatenated_delayeds_with_string_args_use_binary_records():
    """Filenames are the common delayed argument. A concat of
    `from_delayed(delayed(load)(path))` over string paths must ALSO go binary --
    strings are a first-class per-block slot, not a fallback."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    names = ["a.npy", "bb.npy", "ccc.npy"]
    pieces = [da.from_delayed(dask.delayed(_load_named)(n), (5,), dtype="int64") for n in names]
    arr = da.concatenate(pieces)
    fm = _the_from_map(arr)

    assert isinstance(fm._frisky_layer().to_records_chunk(), bytes)

    _chunks, _records, chunk_groups = arr.__frisky_records_chunks__()
    assert fm._name in {name for name, _m, _u in chunk_groups}
    assert _from_map_block_records(arr, fm) == []

    expected = np.concatenate([np.full(5, _FAKE_FILES[n]) for n in names]).astype("int64")
    assert_eq(arr, expected)


def test_binary_from_map_layer_graph_matches_legacy():
    """The native binary layer's per-block graph must reproduce the legacy
    `_layer()` block-for-block. Covers scalar args, string args, and a NON-square
    multi-axis grid (nested stacks -> one 3-D FromMap with a (2, 3, 1) block grid
    and distinct per-block values), which would catch a C-order row/column swap
    in the per-block arg placement."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    from dask.core import flatten
    from dask.local import get_sync

    scalars = da.concatenate([da.from_delayed(dask.delayed(_load)(v), (5,), dtype="int64") for v in [1, 2, 3]])
    strings = da.concatenate(
        [da.from_delayed(dask.delayed(_load_named)(n), (5,), dtype="int64") for n in ["a.npy", "bb.npy"]]
    )
    grid = da.stack(
        [
            da.stack([da.from_delayed(dask.delayed(_load)(v), (5,), dtype="int64") for v in row])
            for row in [[1, 2, 3], [4, 5, 6]]
        ]
    )  # (2, 3, 5), block grid (2, 3, 1), all-distinct values
    for arr in (scalars, strings, grid):
        fm = _the_from_map(arr)
        assert isinstance(fm._frisky_layer().to_records_chunk(), bytes)  # all go binary
        legacy = fm._layer()
        native = fm._frisky_layer().to_dask_graph()
        for key in flatten(fm.__dask_keys__()):
            nblock, lblock = get_sync(native, key), get_sync(legacy, key)
            np.testing.assert_array_equal(nblock, lblock)
            # dtype too: a coerced arg type (e.g. np.int8 -> int) would silently
            # change the block's inferred dtype without changing its values.
            assert nblock.dtype == lblock.dtype


def test_nonsimple_arg_delayeds_fall_back_to_plain_records_and_warn(caplog):
    """A per-block arg the binary grammar can't hold (here an ndarray) makes the
    coalesced FromMap decline the binary path -- it stays on the plain-records
    FromMapLayer and logs why, so a missed fast path is visible."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    import logging

    pieces = [da.from_delayed(dask.delayed(_load_arr)(np.full(5, v)), (5,), dtype="int64") for v in [1, 2, 3]]
    arr = da.concatenate(pieces)
    fm = _the_from_map(arr)

    # No binary chunk for this layer -- the encoder declines.
    with pytest.raises(NotImplementedError):
        fm._frisky_layer().to_records_chunk()

    # It falls back to plain Python-tuple records (still correct)...
    with caplog.at_level(logging.WARNING):
        assert _from_map_block_records(arr, fm) != []
    # ...and it says so, naming the layer.
    assert any(fm._name in rec.getMessage() for rec in caplog.records)

    expected = np.concatenate([np.full(5, v) for v in [1, 2, 3]]).astype("int64")
    assert_eq(arr, expected)


def test_binary_from_map_declines_numpy_scalar_args():
    """A numpy-scalar per-block arg is DECLINED (not silently coerced to a plain
    Python scalar), so the binary path can't change the argument type -- or the
    block's inferred dtype -- the block function sees vs the legacy path. It falls
    back to the faithful Python-tuple layer, still correct."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    pieces = [da.from_delayed(dask.delayed(_load)(np.int64(v)), (5,), dtype="int64") for v in [1, 2, 3]]
    arr = da.concatenate(pieces)
    fm = _the_from_map(arr)
    with pytest.raises(NotImplementedError):
        fm._frisky_layer().to_records_chunk()  # numpy scalar arg -> declined, not binary
    assert _from_map_block_records(arr, fm) != []  # fell back to python-tuple records
    assert_eq(arr, np.concatenate([np.full(5, v) for v in [1, 2, 3]]).astype("int64"))


def test_uniform_positional_object_arg_is_baked_and_goes_binary():
    """A positional arg that is UNIFORM across blocks but not slot-expressible (a
    storage_options-style dict) is baked into the shared pickled partial (once), so
    the layer still goes binary -- only the varying arg (the path) is slotted."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    so = {"scale": 10, "anon": False}
    paths = ["a", "bb", "ccc"]
    pieces = [da.from_delayed(dask.delayed(_read_scaled)(p, so), (5,), dtype="int64") for p in paths]
    arr = da.concatenate(pieces)
    fm = _the_from_map(arr)

    assert isinstance(fm._frisky_layer().to_records_chunk(), bytes)  # baked -> binary
    _chunks, _records, chunk_groups = arr.__frisky_records_chunks__()
    assert fm._name in {name for name, _m, _u in chunk_groups}
    assert _from_map_block_records(arr, fm) == []

    # byte + dtype parity of the native (baked-template) graph vs legacy _layer.
    from dask.core import flatten
    from dask.local import get_sync

    legacy, native = fm._layer(), fm._frisky_layer().to_dask_graph()
    for key in flatten(fm.__dask_keys__()):
        nblock, lblock = get_sync(native, key), get_sync(legacy, key)
        np.testing.assert_array_equal(nblock, lblock)
        assert nblock.dtype == lblock.dtype

    assert_eq(arr, np.concatenate([np.full(5, len(p) * 10) for p in paths]).astype("int64"))


def test_varying_nonexpressible_positional_arg_still_declines():
    """An arg that BOTH varies per block AND isn't slot-expressible (a distinct
    dict per block) has nothing to bake and nothing to slot -- so the layer still
    declines to the faithful Python-tuple fallback."""
    if importlib.util.find_spec("dask_array._rust") is None:
        pytest.skip("requires Rust extension")
    combos = [("a", 1), ("bb", 2), ("ccc", 3)]
    pieces = [da.from_delayed(dask.delayed(_read_scaled)(p, {"scale": s}), (5,), dtype="int64") for p, s in combos]
    arr = da.concatenate(pieces)
    fm = _the_from_map(arr)
    with pytest.raises(NotImplementedError):
        fm._frisky_layer().to_records_chunk()  # varying dict: can't bake, can't slot
    assert _from_map_block_records(arr, fm) != []
    assert_eq(arr, np.concatenate([np.full(5, len(p) * s) for p, s in combos]).astype("int64"))


def test_binary_from_map_executes_under_frisky(array_scheduler):
    """End-to-end: coalesced from_delayed concats that became a binary FromMap
    compute correctly under the Frisky scheduler -- scalar args, string (filename)
    args (the v2 Str-slot grammar across both extensions), and a uniform positional
    dict arg baked into the shared partial. Runs only on --scheduler=frisky (needs
    both native extensions in one interpreter); the dask suite otherwise exercises
    only _layer / to_dask_graph, never the wire."""
    if array_scheduler != "frisky":
        pytest.skip("requires --scheduler=frisky")

    scalars = da.concatenate([da.from_delayed(dask.delayed(_load)(v), (5,), dtype="int64") for v in [1, 2, 3]])
    names = ["a.npy", "bb.npy", "ccc.npy", "dddd.npy"]
    strings = da.concatenate([da.from_delayed(dask.delayed(_load_named)(n), (5,), dtype="int64") for n in names])
    so = {"scale": 10, "anon": False}
    paths = ["a", "bb", "ccc"]
    baked = da.concatenate([da.from_delayed(dask.delayed(_read_scaled)(p, so), (5,), dtype="int64") for p in paths])

    # Guard against a vacuous pass: all must actually produce a binary chunk (a
    # version-mismatched Frisky would fall back to Python records and still pass).
    for coll in (scalars, strings, baked):
        assert isinstance(_the_from_map(coll)._frisky_layer().to_records_chunk(), bytes)

    np.testing.assert_array_equal(scalars.compute(), np.concatenate([np.full(5, v) for v in [1, 2, 3]]).astype("int64"))
    np.testing.assert_array_equal(
        strings.compute(), np.concatenate([np.full(5, _FAKE_FILES[n]) for n in names]).astype("int64")
    )
    np.testing.assert_array_equal(
        baked.compute(), np.concatenate([np.full(5, len(p) * 10) for p in paths]).astype("int64")
    )
