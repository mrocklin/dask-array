"""Behavioral tests for ``da.to_hdf5`` and ``da.from_array`` on h5py datasets.

The ``to_hdf5`` tests are ported from upstream
``dask/array/tests/test_array_core.py`` ``test_to_hdf5`` (dask 2025.12.0),
split into one test per chunking mode and using pytest's ``tmp_path`` instead
of ``tmpfile``. The ``from_array`` tests cover reading h5py datasets, whose
lack of a deterministic tokenize handler exercises FromArray's uuid-token
fallback.
"""

from __future__ import annotations

import numpy as np
import pytest

import dask
from dask.base import tokenize
from dask.core import flatten

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
        e = da.from_array(dset, chunks=(2, 3), name="x-roundtrip")
        assert e.dtype == x.dtype
        assert_eq(e, x)


@pytest.fixture
def hdf5_file(tmp_path):
    """An HDF5 file holding a (4, 6) float dataset at /data/x."""
    x = np.arange(4 * 6, dtype="f8").reshape(4, 6)
    fn = str(tmp_path / "a.hdf5")
    with h5py.File(fn, mode="w") as f:
        f.create_dataset("/data/x", data=x, chunks=(2, 3))
    return fn, x


def test_from_array_hdf5_dataset_without_name(hdf5_file):
    # The classic "read HDF5 into dask" pattern. h5py datasets cannot be
    # deterministically tokenized; from_array falls back to a uuid-based
    # name (as upstream dask does) rather than raising TokenizationError.
    fn, x = hdf5_file
    with h5py.File(fn, mode="r") as f:
        e = da.from_array(f["/data/x"], chunks=(2, 3))
        assert e.dtype == x.dtype
        assert_eq(e, x)


def test_from_array_hdf5_dataset_name_is_stable(hdf5_file):
    # The fallback token is computed once at construction: the name must not
    # flap across accesses, tokenizations, or compute — the advertised keys
    # are the graph's keys.
    fn, x = hdf5_file
    with h5py.File(fn, mode="r") as f:
        dset = f["/data/x"]
        e = da.from_array(dset, chunks=(2, 3))

        name = e.name
        assert name.startswith("array-")
        assert e.expr._name == name
        assert tokenize(e.expr) == tokenize(e.expr)

        graph = e.__dask_graph__()
        assert all(key in graph for key in flatten(e.__dask_keys__()))

        assert_eq(e, x)
        assert e.name == name  # unchanged by materialization

        # Two from_array calls on the same dataset get independent uuid
        # tokens and don't dedup — same semantics as upstream dask.
        e2 = da.from_array(dset, chunks=(2, 3))
        assert e2.name != name

        # name=False explicitly requests a random unique name; it must also
        # work on sources that can't be tokenized.
        e3 = da.from_array(dset, chunks=(2, 3), name=False)
        assert e3.name != name
        assert_eq(e3, x)


def test_from_array_hdf5_dataset_explicit_name(hdf5_file):
    # Explicit name= is used verbatim, exactly as before the uuid fallback.
    fn, x = hdf5_file
    with h5py.File(fn, mode="r") as f:
        e = da.from_array(f["/data/x"], chunks=(2, 3), name="my-hdf5-data")
        assert e.name == "my-hdf5-data"
        assert_eq(e, x)


def test_from_array_hdf5_dataset_ensure_deterministic_still_raises(hdf5_file):
    # The uuid fallback honors upstream's strictness knob: users auditing for
    # accidental non-dedup still get the error.
    from dask.tokenize import TokenizationError

    fn, _ = hdf5_file
    with h5py.File(fn, mode="r") as f:
        with dask.config.set({"tokenize.ensure-deterministic": True}):
            with pytest.raises(TokenizationError):
                da.from_array(f["/data/x"], chunks=(2, 3))


def test_to_hdf5_bad_args(tmp_path):
    x = da.ones((4, 4), chunks=(2, 2))

    fn = str(tmp_path / "a.hdf5")
    with pytest.raises(ValueError, match="provide"):
        da.to_hdf5(fn, x)
