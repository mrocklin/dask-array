"""Every task in a Frisky binary-records chunk carries an honest
``expected_nbytes`` stamp: exactly the ``.nbytes`` of the chunk that task
produces when run.

Frisky's scheduler consumes the stamps to simulate sequential peak memory when
choosing between task orderings; a stamp of 0 means "unknown" and the task is
invisible to the memory model. So two properties matter, and both are asserted
here against ground truth — the emitted chunks are decoded and every task is
actually executed:

- **honesty**: stamp == the executed output's ``.nbytes``, per task. This is
  stronger than per-layer conservation (sum of stamps ~= layer output bytes)
  and also catches helpers stamped at full-block size when they produce a
  small carry, or vice versa.
- **coverage**: no task is stamped 0 when the expression's chunk sizes are
  known — helper tasks (rechunk splits, scan carries, overlap halo getitems,
  shuffle splits, window totals) included.

Representative expression shapes are parameterized below; add a case here when
a new layer kind starts emitting binary records.
"""

from __future__ import annotations

import importlib.util
import struct

import numpy as np
import pytest

import dask_array as da

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("dask_array._rust") is None,
    reason="requires the Rust extension",
)


def _decode_funcs(chunk):
    """The pickled shared funcs of a binary LAYER chunk (see the grammar in
    crates/dask-array-python/src/common.rs)."""
    import cloudpickle

    pos = 1  # version byte

    def u32():
        nonlocal pos
        v = struct.unpack_from("<I", chunk, pos)[0]
        pos += 4
        return v

    for _ in range(u32()):  # names
        n = u32()  # nb: `pos += u32()` would read the old pos
        pos += n
    for _ in range(u32()):  # dep_names
        n = u32()
        pos += n
    funcs = []
    for _ in range(u32()):
        n = u32()
        funcs.append(cloudpickle.loads(chunk[pos : pos + n]))
        pos += n
    return funcs


def _key(name, coord):
    """The Frisky-side string key for a task/dep (str of the key tuple)."""
    if not coord:
        return name
    return f"('{name}', {', '.join(str(c) for c in coord)})"


def _decoded_graph(collection):
    """All binary-chunk tasks plus plain records of ``collection``, as one
    executable dict, and the per-task expected_nbytes stamps (binary only)."""
    from dask_array.tests.test_frisky_protocol import _decode_layer_chunk

    chunks, records, _chunk_groups = collection.__frisky_records_chunks__()
    graph = {}
    stamps = {}
    for chunk in chunks:
        funcs = _decode_funcs(chunk)
        _names, _dep_names, tasks = _decode_layer_chunk(chunk)
        for name, coord, nbytes, compute, slots in tasks:
            key = _key(name, coord)
            graph[key] = ("binary", funcs, compute, slots)
            stamps[key] = nbytes
    for key, func, args, kwargs, _deps in records:
        graph[str(key)] = ("record", func, args, kwargs)
    return graph, stamps


def _execute(graph):
    """Compute every key of a decoded graph; returns {key: value}."""
    import sys

    from dask._task_spec import TaskRef

    memo = {}

    def slot_value(slot):
        tag = slot[0]
        if tag == "dep":
            _, name, coord = slot
            key = _key(name, coord)
            if key not in graph and name in graph:
                key = name
            return compute(key)
        if tag == "index":
            return tuple(slice(e[1], e[2], e[3]) if e[0] == "slice" else e[1] for e in slot[1])
        if tag == "inttuple":
            return tuple(slot[1])
        if tag == "list":
            return [slot_value(s) for s in slot[1]]
        if tag in ("scalar", "str"):
            return slot[1]
        raise AssertionError(f"unhandled slot {slot!r}")

    def sub(obj):
        if isinstance(obj, TaskRef):
            return compute(str(obj.key))
        if isinstance(obj, (tuple, list)):
            return type(obj)(sub(o) for o in obj)
        if isinstance(obj, dict):
            return {k: sub(v) for k, v in obj.items()}
        return obj

    def compute(key):
        if key in memo:
            return memo[key]
        kind, *rest = graph[key]
        if kind == "binary":
            funcs, comp, slots = rest
            if comp[0] == "alias":
                val = slot_value(slots[0])
            else:
                val = funcs[comp[1]](*[slot_value(s) for s in slots])
        else:
            func, args, kwargs = rest
            val = func(*sub(args), **sub(kwargs))
        memo[key] = val
        return val

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(100_000)
    try:
        for key in graph:
            compute(key)
    finally:
        sys.setrecursionlimit(limit)
    return memo


def _actual_nbytes(value):
    if isinstance(value, dict):  # sliding-window totals: {"total": ..., "count": ...}
        return sum(v.nbytes for v in value.values())
    return value.nbytes if hasattr(value, "nbytes") else np.asarray(value).nbytes


def _rng():
    return np.random.default_rng(0)


def _ew_scan():
    # ew-style cumreduction over a state array with a trailing state axis
    # (the statarb ew_running_sum shape): a map_blocks state builder, then a
    # sequential scan whose carries are single hyperplanes.
    from dask_array.reductions import cumreduction

    def make_state(block):
        out = np.empty(block.shape + (2,), dtype=block.dtype)
        out[..., 0] = 0
        out[..., 1] = block
        return out

    def scan_state(block, axis=0):
        return np.cumsum(block, axis=axis)

    x = da.from_array(_rng().standard_normal((40, 12)), chunks=(10, 6))
    state = x.map_blocks(make_state, dtype=x.dtype, chunks=x.chunks + ((2,),), new_axis=[x.ndim])
    scanned = cumreduction(scan_state, np.add, 0, state, axis=0, dtype=x.dtype)
    return scanned[..., 1]


def _moving_window():
    bn = pytest.importorskip("bottleneck")
    data = _rng().normal(size=(96, 4))
    data[_rng().random(data.shape) < 0.2] = np.nan
    x = da.from_array(data, chunks=((7, 12, 9, 14, 8, 12, 6, 12, 16), (4,)))
    return x.map_overlap(bn.move_sum, depth={0: (19, 0)}, dtype="f8", window=20, min_count=1, axis=0)


def _sliding_window():
    data = _rng().normal(size=(96, 6))
    data[_rng().random(data.shape) < 0.2] = np.nan
    x = da.from_array(data, chunks=((7, 12, 9, 14, 8, 12, 6, 12, 16), (4, 2)))
    return da.nanmean(da.sliding_window_view(x, window_shape=20, axis=0), axis=-1)


def _shuffle():
    x = da.from_array(_rng().standard_normal((20, 8)), chunks=(5, 4)) + 0
    idx = [int(i) for i in _rng().permutation(20)]
    return x.shuffle([idx[:12], idx[12:]], axis=0)


CASES = {
    "ew_scan_cumreduction": _ew_scan,
    "cumsum_sequential": lambda: da.cumsum(da.from_array(_rng().standard_normal((64, 8)), chunks=(8, 4)), axis=0),
    "cumsum_blelloch": lambda: da.cumsum(
        da.from_array(_rng().standard_normal((64, 8)), chunks=(8, 4)), axis=0, method="blelloch"
    ),
    "cumsum_blelloch_int8_output_promotes": lambda: da.cumsum(
        da.from_array(np.arange(64 * 4, dtype="i1").reshape(64, 4), chunks=(8, 4)),
        axis=0,
        dtype="i1",
        method="blelloch",
    ),
    "cumsum_blelloch_float32_to_float64": lambda: da.cumsum(
        da.from_array(_rng().standard_normal((64, 4)).astype("f4"), chunks=(8, 4)),
        axis=0,
        dtype="f8",
        method="blelloch",
    ),
    "cumsum_flattened": lambda: da.cumsum(da.from_array(_rng().standard_normal((12, 10)), chunks=(4, 5))),
    "slice": lambda: (da.from_array(_rng().standard_normal((60, 42)), chunks=(15, 7)) + 1)[3:41, 5:30],
    "rechunk": lambda: (da.from_array(_rng().standard_normal((60, 42)), chunks=(15, 7)) + 1).rechunk((25, 14)),
    "rechunk_multi_step": lambda: (da.from_array(_rng().standard_normal((64, 64)), chunks=(2, 64)) + 1).rechunk(
        (64, 2)
    ),
    "map_overlap_reflect": lambda: da.from_array(_rng().standard_normal((40, 30)), chunks=(10, 10)).map_overlap(
        lambda b: b * 2.0, depth=2, boundary="reflect"
    ),
    "map_overlap_none": lambda: da.from_array(_rng().standard_normal((40, 30)), chunks=(10, 10)).map_overlap(
        lambda b: b + 1.0, depth={0: (1, 0)}, boundary="none"
    ),
    "bool_mask": lambda: da.from_array(_rng().standard_normal((60, 42)), chunks=(15, 7)) > 0,
    "stack": lambda: da.stack(
        [da.from_array(_rng().standard_normal((10, 6)), chunks=(5, 3)) for _ in range(4)], axis=0
    ),
    "reshape": lambda: da.from_array(_rng().standard_normal((60, 42)), chunks=(15, 7)).reshape(20, 3, 42),
    "tree_reduction": lambda: da.from_array(_rng().standard_normal((60, 42)), chunks=(15, 7)).sum(
        axis=0, split_every=2
    ),
    "shuffle": _shuffle,
    "sliding_window_nanmean": _sliding_window,
    "moving_window_move_sum": _moving_window,
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_expected_nbytes_stamps_match_executed_output(case):
    collection = CASES[case]()
    graph, stamps = _decoded_graph(collection)
    assert stamps, "expected at least one binary-records layer"
    values = _execute(graph)

    mismatches = {}
    for key, stamp in stamps.items():
        actual = _actual_nbytes(values[key])
        if stamp != actual:
            mismatches[key] = (stamp, actual)
    assert not mismatches, f"{len(mismatches)}/{len(stamps)} stamps dishonest, e.g. " + str(
        dict(list(mismatches.items())[:5])
    )

    # Full coverage: chunk sizes are known in every case above, so no task may
    # be stamped 0/unknown (0 hides the task from Frisky's memory model).
    unstamped = [k for k, s in stamps.items() if s <= 0]
    assert not unstamped, f"{len(unstamped)}/{len(stamps)} tasks unstamped, e.g. {unstamped[:5]}"


def test_unknown_chunks_leave_stamps_zero_without_breaking_records():
    # Boolean-mask indexing produces unknown (nan) chunk sizes; the layers must
    # still emit binary records, with stamps either left 0 (unknown) or an
    # upper bound. (The fused gt+getitem block task declares source-block
    # chunks, so its data-dependent output stamps at source size — the bounded
    # conservative choice, since 0 would read as "free".)
    x = da.from_array(_rng().standard_normal(30), chunks=10)
    masked = x[x > 0]
    y = da.cumsum(masked, axis=0)
    assert np.isnan(y.chunks[0]).any()

    graph, stamps = _decoded_graph(y)
    values = _execute(graph)
    for key, stamp in stamps.items():
        assert stamp == 0 or stamp >= _actual_nbytes(values[key])
