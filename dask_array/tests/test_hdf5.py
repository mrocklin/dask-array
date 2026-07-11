"""Behavioral tests for ``da.to_hdf5``.

Ported from upstream ``dask/array/tests/test_array_core.py`` ``test_to_hdf5``
(dask 2025.12.0), split into one test per chunking mode and using pytest's
``tmp_path`` instead of ``tmpfile``.
"""

from __future__ import annotations

import numpy as np
import pytest

import dask_array as da
from dask_array._test_utils import assert_eq

h5py = pytest.importorskip("h5py")

# to_hdf5 stores into h5py dataset handles opened on the client; h5py handles
# don't pickle, so a serializing scheduler (Frisky) can't ship the store tasks.
# These need a local (by-reference) scheduler.
pytestmark = pytest.mark.requires_local_scheduler


def test_to_hdf5_method(tmp_path):
    x = da.ones((4, 4), chunks=(2, 2))

    fn = str(tmp_path / "a.hdf5")
    x.to_hdf5(fn, "/x")
    with h5py.File(fn, mode="r") as f:
        d = f["/x"]

        assert_eq(d[:], x)
        assert d.chunks == (2, 2)


def test_to_hdf5_chunks_none(tmp_path):
    x = da.ones((4, 4), chunks=(2, 2))

    fn = str(tmp_path / "a.hdf5")
    x.to_hdf5(fn, "/x", chunks=None)
    with h5py.File(fn, mode="r") as f:
        d = f["/x"]

        assert_eq(d[:], x)
        assert d.chunks is None


def test_to_hdf5_explicit_chunks(tmp_path):
    x = da.ones((4, 4), chunks=(2, 2))

    fn = str(tmp_path / "a.hdf5")
    x.to_hdf5(fn, "/x", chunks=(1, 1))
    with h5py.File(fn, mode="r") as f:
        d = f["/x"]

        assert_eq(d[:], x)
        assert d.chunks == (1, 1)


def test_to_hdf5_multiple_datasets(tmp_path):
    x = da.ones((4, 4), chunks=(2, 2))
    y = da.ones(4, chunks=2, dtype="i4")

    fn = str(tmp_path / "a.hdf5")
    da.to_hdf5(fn, {"/x": x, "/y": y})

    with h5py.File(fn, mode="r") as f:
        assert_eq(f["/x"][:], x)
        assert f["/x"].chunks == (2, 2)
        assert_eq(f["/y"][:], y)
        assert f["/y"].chunks == (2,)


def test_to_hdf5_roundtrip_values(tmp_path):
    # Non-trivial values survive the write and read back through from_array.
    x = np.arange(4 * 6, dtype="f8").reshape(4, 6)
    d = da.from_array(x, chunks=(2, 3))

    fn = str(tmp_path / "a.hdf5")
    da.to_hdf5(fn, "/data/x", d)

    with h5py.File(fn, mode="r") as f:
        dset = f["/data/x"]
        # An explicit name= is required today; see
        # test_from_array_hdf5_dataset_without_name below.
        e = da.from_array(dset, chunks=(2, 3), name="x-roundtrip")
        assert e.dtype == x.dtype
        assert_eq(e, x)


@pytest.mark.xfail(
    reason="FromArray.__dask_tokenize__ uses dask.tokenize._tokenize_deterministic "
    "unconditionally; h5py.Dataset has no deterministic tokenize handler, so "
    "da.from_array(h5py_dataset) without an explicit name= raises "
    "TokenizationError. Upstream dask falls back to a non-deterministic token "
    "here. Workaround: pass name=.",
    strict=True,
)
def test_from_array_hdf5_dataset_without_name(tmp_path):
    x = np.arange(4 * 6, dtype="f8").reshape(4, 6)
    d = da.from_array(x, chunks=(2, 3))

    fn = str(tmp_path / "a.hdf5")
    da.to_hdf5(fn, "/data/x", d)

    with h5py.File(fn, mode="r") as f:
        e = da.from_array(f["/data/x"], chunks=(2, 3))
        assert_eq(e, x)


def test_to_hdf5_bad_args(tmp_path):
    x = da.ones((4, 4), chunks=(2, 2))

    fn = str(tmp_path / "a.hdf5")
    with pytest.raises(ValueError, match="provide"):
        da.to_hdf5(fn, x)
