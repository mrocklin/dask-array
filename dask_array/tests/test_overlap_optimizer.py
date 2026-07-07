"""map_overlap must expose its input array(s) to the optimizer.

Regression tests for a bug where ``MapOverlap`` stored its inputs in a ``tuple``
operand, hiding them from ``Expr.dependencies()`` and the simplify/lower
traversal (which only descend into *direct* ``Expr`` operands). The input was
therefore never simplified, and lowering re-materialized the un-optimized
original — e.g. a mergeable ``concatenate([from_delayed, ...])`` source was
stranded as per-block ``FromDelayed`` instead of collapsing to one ``FromMap``.
"""

from __future__ import annotations

import numpy as np
import pytest
from dask import delayed
from dask._expr import Expr

import dask_array as da


def _count(expr, cls_name):
    seen, stack, n = set(), [expr], 0
    while stack:
        e = stack.pop()
        if e._name in seen:
            continue
        seen.add(e._name)
        n += type(e).__name__ == cls_name
        stack.extend(e.dependencies())
    return n


def _mergeable_concat(n_blocks=5, shape=(8, 4)):
    # n single-self-contained-call from_delayed leaves -> each normalizes to a
    # 1-block FromMap, and concatenate merges them into one FromMap.
    days = [
        da.from_delayed(delayed(lambda i=i: np.zeros(shape))(), shape=shape, dtype="f8")
        for i in range(n_blocks)
    ]
    return da.concatenate(days, axis=0)


def test_map_overlap_exposes_input_as_dependency():
    y = _mergeable_concat().map_overlap(lambda b: b, depth={0: 1}, boundary="none")
    deps = y.expr.dependencies()
    assert deps, "MapOverlap must expose its input array as a dependency"
    # no Expr child may be hidden inside a collection operand
    hidden = sum(
        isinstance(it, Expr)
        for o in y.expr.operands
        if isinstance(o, (list, tuple))
        for it in o
    )
    assert hidden == 0


def test_map_overlap_input_is_simplified_not_stranded():
    # The mergeable source must survive lowering as a merged FromMap, not be
    # re-expanded into per-block FromDelayed.
    y = _mergeable_concat().map_overlap(lambda b: b, depth={0: 1}, boundary="none")
    opt = y.optimize().expr
    assert _count(opt, "FromDelayed") == 0


def test_map_overlap_still_correct():
    x = da.arange(20, chunks=5, dtype="f8")
    y = x.map_overlap(lambda b: b + 1.0, depth=1, boundary="none")
    np.testing.assert_array_equal(np.asarray(y), np.arange(20, dtype="f8") + 1.0)


def test_multi_array_map_overlap_correct_and_exposed():
    a = da.arange(20, chunks=5, dtype="f8")
    b = da.arange(20, chunks=5, dtype="f8") * 10
    y = da.map_overlap(lambda p, q: p + q, a, b, depth=1, boundary="none")
    assert len(y.expr.dependencies()) == 2
    np.testing.assert_array_equal(np.asarray(y), np.arange(20, dtype="f8") * 11)
