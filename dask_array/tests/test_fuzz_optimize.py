"""Property-based fuzz of the optimization pipeline.

Generates random expression chains (IO, elemwise, transpose, slicing,
rechunk, reductions, expand_dims, concatenate) alongside a numpy mirror of
the same computation, then checks that the optimized dask result matches
numpy exactly and that optimization converges while preserving shape and
dtype.

The values are arange-based so every element is distinct — index-mapping
bugs (a transposed slice, an off-by-one block boundary) change values and
get caught; constant test data would mask them.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

import dask_array as da
from dask_array._test_utils import assert_eq

MAX_DIM = 6


@st.composite
def _leaf(draw, shape=None):
    """A (numpy, dask) pair for the same random small array."""
    if shape is None:
        ndim = draw(st.integers(1, 3))
        shape = tuple(draw(st.integers(1, MAX_DIM)) for _ in range(ndim))
    offset = draw(st.integers(0, 100))
    np_arr = (np.arange(int(np.prod(shape))) + offset).astype("f8").reshape(shape)
    chunks = tuple(draw(st.integers(1, max(1, s))) for s in shape)  # dims may be empty
    return np_arr, da.from_array(np_arr, chunks=chunks)


@st.composite
def _index(draw, shape):
    """A random basic index for ``shape``: slices and integers."""
    index = []
    for size in shape:
        kinds = ["full", "slice"] + (["int"] if size > 0 else [])
        kind = draw(st.sampled_from(kinds))
        if kind == "full":
            index.append(slice(None))
        elif kind == "int":
            index.append(draw(st.integers(-size, size - 1)))
        else:
            start = draw(st.integers(0, size))
            stop = draw(st.integers(start, size))
            index.append(slice(start, stop))
    return tuple(index)


@st.composite
def _apply_op(draw, np_arr, da_arr):
    """Apply one random operation to both representations.

    Returns (np_result, da_result, description) — the description makes
    falsifying examples self-describing.
    """
    ndim = np_arr.ndim
    shape = np_arr.shape
    ops = ["neg", "add_scalar", "mul_scalar", "add_leaf", "mul_broadcast_leaf"]
    if ndim >= 1:
        ops += ["transpose", "getitem", "rechunk", "sum", "mean", "expand_dims", "concatenate"]

    op = draw(st.sampled_from(ops))

    if op == "neg":
        return -np_arr, -da_arr, "neg"
    if op == "add_scalar":
        c = draw(st.integers(-5, 5))
        return np_arr + c, da_arr + c, f"add {c}"
    if op == "mul_scalar":
        c = draw(st.integers(-3, 3))
        return np_arr * c, da_arr * c, f"mul {c}"
    if op == "add_leaf":
        np_other, da_other = draw(_leaf(shape=shape))
        return np_arr + np_other, da_arr + da_other, f"add leaf chunks={da_other.chunks}"
    if op == "mul_broadcast_leaf":
        # Same shape with random dims collapsed to 1 (broadcasting)
        bshape = tuple(1 if draw(st.booleans()) else s for s in shape)
        np_other, da_other = draw(_leaf(shape=bshape))
        return np_arr * np_other, da_arr * da_other, f"mul broadcast leaf {bshape}"
    if op == "transpose":
        axes = tuple(draw(st.permutations(range(ndim))))
        return np_arr.transpose(axes), da_arr.transpose(axes), f"transpose {axes}"
    if op == "getitem":
        index = draw(_index(shape))
        return np_arr[index], da_arr[index], f"getitem {index}"
    if op == "rechunk":
        chunks = tuple(draw(st.integers(1, max(1, s))) for s in shape)
        return np_arr, da_arr.rechunk(chunks), f"rechunk {chunks}"
    if op == "sum":
        axis = draw(st.integers(0, ndim - 1))
        keepdims = draw(st.booleans())
        return (
            np_arr.sum(axis=axis, keepdims=keepdims),
            da_arr.sum(axis=axis, keepdims=keepdims),
            f"sum axis={axis} keepdims={keepdims}",
        )
    if op == "mean":
        axis = draw(st.integers(0, ndim - 1))
        if np_arr.shape[axis] == 0:
            return np_arr, da_arr, "noop"  # avoid mean-of-empty warnings
        return np_arr.mean(axis=axis), da_arr.mean(axis=axis), f"mean axis={axis}"
    if op == "expand_dims":
        axis = draw(st.integers(0, ndim))
        return np.expand_dims(np_arr, axis), da.expand_dims(da_arr, axis), f"expand_dims {axis}"
    if op == "concatenate":
        axis = draw(st.integers(0, ndim - 1))
        np_other, da_other = draw(_leaf(shape=shape))
        return (
            np.concatenate([np_arr, np_other], axis=axis),
            da.concatenate([da_arr, da_other], axis=axis),
            f"concatenate axis={axis} chunks={da_other.chunks}",
        )
    raise AssertionError(op)


@st.composite
def _pipelines(draw):
    np_arr, da_arr = draw(_leaf())
    steps = [f"leaf shape={np_arr.shape} chunks={da_arr.chunks}"]
    for _ in range(draw(st.integers(0, 5))):
        if np_arr.ndim > 4 or np_arr.size > 20_000:
            break  # keep examples small and fast
        np_arr, da_arr, step = draw(_apply_op(np_arr, da_arr))
        steps.append(step)
    return np_arr, da_arr, " | ".join(steps)


@given(_pipelines())
@settings(max_examples=200, deadline=None)
def test_optimized_matches_numpy(pair):
    np_arr, da_arr, steps = pair

    opt = da_arr.expr.optimize()  # simplify/lower/fuse converge without error
    assert opt.shape == np_arr.shape, steps
    assert opt.dtype == np_arr.dtype, steps

    # Value correctness through the fully optimized graph
    assert_eq(da_arr, np_arr)
