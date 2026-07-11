"""Round-trip tests for ``da.to_npy_stack`` / ``da.from_npy_stack``.

The write test is ported from upstream ``dask/array/tests/test_array_core.py``
``test_to_npy_stack`` (dask 2025.12.0), using pytest's ``tmp_path`` instead of
``tmpdir``. The rechunk/slice tests are project additions.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import dask_array as da
from dask_array._test_utils import assert_eq


def test_to_npy_stack(tmp_path):
    x = np.arange(5 * 10 * 10).reshape((5, 10, 10))
    d = da.from_array(x, chunks=(2, 4, 4))

    stackdir = os.path.join(tmp_path, "test")
    da.to_npy_stack(stackdir, d, axis=0)
    assert os.path.exists(os.path.join(stackdir, "0.npy"))
    assert (np.load(os.path.join(stackdir, "1.npy")) == x[2:4]).all()

    e = da.from_npy_stack(stackdir)
    assert_eq(d, e)


def test_npy_stack_roundtrip_axis(tmp_path):
    # Stacking along a non-zero axis coalesces the other axes into one chunk.
    x = np.arange(6 * 8).reshape((6, 8))
    d = da.from_array(x, chunks=(3, 4))

    stackdir = os.path.join(tmp_path, "stack")
    da.to_npy_stack(stackdir, d, axis=1)

    e = da.from_npy_stack(stackdir)
    assert e.chunks == ((6,), (4, 4))
    assert_eq(e, x)


@pytest.mark.parametrize("mmap_mode", ["r", None])
def test_npy_stack_mmap_mode(tmp_path, mmap_mode):
    x = np.arange(4 * 5).reshape((4, 5)).astype("f8")
    d = da.from_array(x, chunks=(2, 5))

    stackdir = os.path.join(tmp_path, "stack")
    da.to_npy_stack(stackdir, d, axis=0)

    e = da.from_npy_stack(stackdir, mmap_mode=mmap_mode)
    assert_eq(e, x)


def test_npy_stack_sliced_and_rechunked_read(tmp_path):
    x = np.arange(6 * 4 * 4).reshape((6, 4, 4))
    d = da.from_array(x, chunks=(2, 4, 4))

    stackdir = os.path.join(tmp_path, "stack")
    da.to_npy_stack(stackdir, d, axis=0)

    e = da.from_npy_stack(stackdir)

    # Sliced read
    assert_eq(e[1:5], x[1:5])
    assert_eq(e[:, 2, :], x[:, 2, :])

    # Rechunked read
    assert_eq(e.rechunk((3, 2, 2)), x)

    # Sliced + rechunked + computed through an operation
    assert_eq(e.rechunk((3, 4, 4))[::2].sum(axis=0), x[::2].sum(axis=0))
