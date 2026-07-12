from __future__ import annotations

import numpy as np
import pytest

import dask_array as da
from dask_array._test_utils import assert_eq


@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("chunks", [(4, 8), (12, 8)])
def test_apply_gufunc_single_output(keepdims, chunks):
    rng = np.random.default_rng(0)
    base = rng.random((12, 8))
    a = da.from_array(base, chunks=chunks)
    got = da.apply_gufunc(lambda x: np.sum(x, -1), "(i)->()", a, output_dtypes=float, keepdims=keepdims)
    assert_eq(got, np.sum(base, -1, keepdims=keepdims))


def test_apply_gufunc_multiple_outputs():
    rng = np.random.default_rng(1)
    base = rng.random((12, 8))
    a = da.from_array(base, chunks=(4, 8))
    m, s = da.apply_gufunc(lambda x: (np.mean(x, -1), np.std(x, -1)), "(i)->(),()", a, output_dtypes=(float, float))
    assert_eq(m, base.mean(-1))
    assert_eq(s, base.std(-1))


def test_apply_gufunc_core_output_dim():
    rng = np.random.default_rng(2)
    base = rng.random((10, 6))
    a = da.from_array(base, chunks=(5, 6))
    got = da.apply_gufunc(lambda x: x * 2, "(i)->(i)", a, output_dtypes=float)
    assert_eq(got, base * 2)


def test_apply_gufunc_multiple_outputs_with_core_dims():
    rng = np.random.default_rng(3)
    base = rng.random((10, 6))
    a = da.from_array(base, chunks=(5, 6))
    lo, hi = da.apply_gufunc(
        lambda x: (x[..., :2], x[..., 2:]),
        "(i)->(a),(b)",
        a,
        output_dtypes=(float, float),
        output_sizes={"a": 2, "b": 4},
    )
    assert_eq(lo, base[..., :2])
    assert_eq(hi, base[..., 2:])


def test_apply_gufunc_3d_loop():
    rng = np.random.default_rng(4)
    base = rng.random((6, 8, 10))
    a = da.from_array(base, chunks=(3, 4, 10))
    got = da.apply_gufunc(lambda x: np.sum(x, -1), "(i)->()", a, output_dtypes=float)
    assert_eq(got, base.sum(-1))
