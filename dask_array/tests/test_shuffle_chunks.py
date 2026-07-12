"""Tests for shuffle output chunk sizing with input chunk locality grouping."""

from __future__ import annotations

import numpy as np
import pytest

import dask_array as da
from dask_array._test_utils import assert_eq


def test_contiguous_indexing_splits_to_input_chunk_size():
    """np.repeat pattern: output chunks stay close to input chunk size."""
    np_x = np.arange(100 * 10).reshape(100, 10)
    x = da.from_array(np_x, chunks=(25, 10))  # 4 input chunks of 25 each

    # Contiguous: each input element repeated 3 times
    # Each input chunk of 25 elements becomes 75 output elements
    # These get split into chunks of 25, so 3 output chunks per input chunk = 12 total
    indexer = np.repeat(np.arange(100), 3)  # [0,0,0,1,1,1,...,99,99,99]
    result = x[indexer, :]

    assert max(result.chunks[0]) == 25
    assert result.numblocks[0] == 12  # 4 input chunks * 3 splits each
    assert_eq(result, np_x[indexer, :])


def test_scattered_indexing_correctness():
    """np.tile pattern: scattered access still produces correct results."""
    np_x = np.arange(100 * 10).reshape(100, 10)
    x = da.from_array(np_x, chunks=(25, 10))

    indexer = np.tile(np.arange(100), 3)  # [0,1,...,99,0,1,...,99,...]
    result = x[indexer, :]

    assert_eq(result, np_x[indexer, :])


def test_identity_indexing_no_shuffle():
    """Identity indexing should not create a shuffle."""
    from dask_array._shuffle import Shuffle

    np_x = np.arange(120).reshape(12, 10)
    x = da.from_array(np_x, chunks=(3, 10))

    result = x[np.arange(12), :]

    assert not isinstance(result.expr, Shuffle)
    assert_eq(result, np_x)


def test_large_repeat_splits_oversized_groups():
    """np.repeat with large factor should not create oversized chunks.

    When each element is repeated many times, the output chunks should be
    split to match input chunk sizes, not grow unboundedly.
    """
    np_x = np.arange(100 * 10).reshape(100, 10)
    x = da.from_array(np_x, chunks=(25, 10))  # 4 input chunks, 25 elements each

    # Each element repeated 100 times -> naive would give chunks of 25*100=2500
    # With max input chunk size of 25, groups get split into chunks of 25
    indexer = np.repeat(np.arange(100), 100)
    result = x[indexer, :]

    assert max(result.chunks[0]) == 25
    assert_eq(result, np_x[indexer, :])


def _assert_optimized_graph_complete(collection):
    """Every dependency referenced by a task in the optimized graph must be
    produced by some task in that graph.

    This is the completeness property the Frisky records collector enforces
    (``dask_array/_frisky/collect.py::_check_complete``): an optimized graph
    that references keys no task produces is malformed. The scheduler culls
    unreachable tasks, so such a graph can still *compute* the right answer
    whenever the dangling key happens to be culled -- which is exactly why the
    bug below is latent and a value-only (``assert_eq``) check misses it.
    """
    dsk = dict(collection.expr.optimize().__dask_graph__())
    produced = set(dsk)
    dangling = {dep for task in dsk.values() for dep in getattr(task, "dependencies", ()) if dep not in produced}
    assert not dangling, f"optimized graph references unproduced keys, e.g. {next(iter(dangling))}"


def test_take_through_concatenate_keeps_graph_complete():
    """A fancy index (Shuffle) that drifts Concatenate inputs out of alignment
    must not leave the optimized graph referencing keys no task produces.

    Root cause: a Shuffle's advertised chunks are not stable under optimization
    (a shuffle over an Elemwise re-optimizes to a different layout than one over
    a FromArray), so once a shuffle is distributed over the concat inputs they
    drift out of alignment on the non-concat axis. ``Concatenate._layer`` reuses
    the output's non-concat block coordinates as each source's, so misaligned
    inputs make it emit source keys no task produces. Depending on input order
    this surfaces as a *culled* dangling dep (compute still passes -- the latent
    form) or as wrong values. ``Concatenate._lower`` re-pins the inputs to a
    shared non-concat layout.

    Covers both entry points to the drift: the take pushed *through* the concat
    (``concatenate([...])[idx]``, via ``_accept_shuffle``) and written *under*
    it (``concatenate([x[idx], y[idx]])``).
    """
    # Take pushed THROUGH the concat -- the hypothesis-shrunk falsifying example
    # (test_fuzz_optimize.test_optimized_matches_numpy, with an optimized-graph
    # completeness assertion added):
    #   leaf (3,1,1) chunks=((1,1,1),(1,),(1,)) | add leaf ((2,1),(1,),(1,))
    #   | concatenate axis=1 | take [0, 0]
    np_a = np.arange(3.0).reshape(3, 1, 1)
    a = da.from_array(np_a, chunks=((1, 1, 1), (1,), (1,)))
    np_b = (np.arange(3.0) + 10).reshape(3, 1, 1)
    b = da.from_array(np_b, chunks=((2, 1), (1,), (1,)))  # Elemwise axis-0 chunks (2, 1)
    np_c = (np.arange(3.0) + 20).reshape(3, 1, 1)
    c = da.from_array(np_c, chunks=((1, 1, 1), (1,), (1,)))
    elemwise, np_e = a + b, np_a + np_b
    # Both orders: Elemwise-first is the culled-dangling (latent) form, the
    # other the wrong-values form.
    for da_seq, np_seq in (([elemwise, c], [np_e, np_c]), ([c, elemwise], [np_c, np_e])):
        result = da.concatenate(da_seq, axis=1)[[0, 0]]
        expected = np.concatenate([x[[0, 0]] for x in np_seq], axis=1)
        _assert_optimized_graph_complete(result)
        assert_eq(result, expected)

    # Take written UNDER the concat. Needs a 2-D input mix whose per-input
    # shuffles settle to different layouts (asymmetric chunk grids).
    np_m = np.arange(9.0).reshape(3, 3)
    m = da.from_array(np_m, chunks=((1, 1, 1), (3,)))
    n = da.from_array(np_m + 10, chunks=((2, 1), (2, 1)))
    p = da.from_array(np_m + 20, chunks=((2, 1), (1, 1, 1)))
    elemwise2, np_e2, np_p = m + n, np_m + np_m + 10, np_m + 20
    for da_seq, np_seq in (([elemwise2, p], [np_e2, np_p]), ([p, elemwise2], [np_p, np_e2])):
        result = da.concatenate([x[[0, 0]] for x in da_seq], axis=1)
        expected = np.concatenate([x[[0, 0]] for x in np_seq], axis=1)
        _assert_optimized_graph_complete(result)
        assert_eq(result, expected)


def test_take_through_stack_keeps_graph_complete():
    """Same root cause as ``test_take_through_concatenate_keeps_graph_complete``,
    via ``Stack``.

    Stack's layer *requires* every input to share one identical chunk layout,
    so the misalignment surfaces as a hard "Missing dependency" at compute time,
    not merely a culled key. ``Stack._lower`` re-pins the inputs to the first
    input's layout. Covers both the take pushed through the stack and written
    under it.
    """
    # Take pushed THROUGH the stack.
    np_a = np.arange(3.0).reshape(3, 1, 1)
    a = da.from_array(np_a, chunks=((1, 1, 1), (1,), (1,)))
    np_b = (np.arange(3.0) + 10).reshape(3, 1, 1)
    b = da.from_array(np_b, chunks=((2, 1), (1,), (1,)))
    np_c = (np.arange(3.0) + 20).reshape(3, 1, 1)
    c = da.from_array(np_c, chunks=((1, 1, 1), (1,), (1,)))
    elemwise, np_e = a + b, np_a + np_b
    for da_seq, np_seq in (([elemwise, c], [np_e, np_c]), ([c, elemwise], [np_c, np_e])):
        result = da.stack(da_seq, axis=1)[[0, 0]]
        expected = np.stack([x[[0, 0]] for x in np_seq], axis=1)
        _assert_optimized_graph_complete(result)
        assert_eq(result, expected)

    # Take written UNDER the stack (2-D asymmetric inputs, stacked on a new axis).
    np_m = np.arange(9.0).reshape(3, 3)
    m = da.from_array(np_m, chunks=((1, 1, 1), (3,)))
    n = da.from_array(np_m + 10, chunks=((2, 1), (2, 1)))
    p = da.from_array(np_m + 20, chunks=((2, 1), (1, 1, 1)))
    elemwise2, np_e2, np_p = m + n, np_m + np_m + 10, np_m + 20
    for da_seq, np_seq in (([elemwise2, p], [np_e2, np_p]), ([p, elemwise2], [np_p, np_e2])):
        result = da.stack([x[[0, 0]] for x in da_seq], axis=0)
        expected = np.stack([x[[0, 0]] for x in np_seq], axis=0)
        _assert_optimized_graph_complete(result)
        assert_eq(result, expected)


def test_nested_empty_concatenate_keeps_graph_complete():
    """Concatenating along a zero-length axis multiplies its blocks, and a
    later concatenate on another axis must not reference blocks no task produces.

    Regression for a graph-completeness bug shrunk by hypothesis
    (test_fuzz_optimize.test_optimized_shared_matches_numpy, with an added
    optimized-graph completeness assertion): a zero-length axis can carry any
    number of zero-blocks -- ``concatenate([empty, empty])`` along it yields a
    ``(0, 0)`` grid where a plain empty has ``(0,)`` -- and ``unify_chunks``
    leaves empty axes untouched, so an outer ``concatenate`` on a *different*
    axis meets inputs whose block counts disagree on the (empty) non-concat
    axis. ``Concatenate._layer`` reused the output's non-concat coordinate as
    the source's, so it addressed a source block that doesn't exist. It fails
    hard ("Missing dependency") at compute; ``_layer`` now clamps empty-axis
    coordinates into the source's range (every such block is empty, so any is a
    correct stand-in). The empties are made by slicing, matching how they arise
    in practice (a plain ``from_array`` of an empty does not reproduce it).
    """
    np_row = np.arange(3.0).reshape(1, 3)
    row = da.from_array(np_row, chunks=(1, 1))
    np_empty = np_row[0:0, 0:0]
    empty = row[0:0, 0:0]  # (0, 0) via slicing
    other = da.from_array(np_empty, chunks=(1, 1))

    # inner concat along axis 0 -> a (0, 0) block grid on axis 0
    inner = da.concatenate([empty, other], axis=0)
    np_inner = np.concatenate([np_empty, np_empty], axis=0)
    tail = da.from_array(np_empty, chunks=(1, 1))  # a (0,) block grid on axis 0

    # outer concat along axis 1: inner (axis-0 blocks (0, 0)) vs tail (axis-0 (0,))
    result = da.concatenate([inner, tail], axis=1)
    expected = np.concatenate([np_inner, np_empty], axis=1)
    _assert_optimized_graph_complete(result)
    assert_eq(result, expected)

    # also the reverse order and a 3-input mix of (0,) and (0, 0) grids
    result = da.concatenate([tail, inner, da.concatenate([other, other], axis=0)], axis=1)
    expected = np.concatenate([np_empty, np_inner, np_inner], axis=1)
    _assert_optimized_graph_complete(result)
    assert_eq(result, expected)
