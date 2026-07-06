"""Naming and dtype stay cheap: neither drags in semantic metadata.

Two couplings used to make graph construction and optimization needlessly
slow, and a regression in either is silent (results stay correct, just slow):

- Naming a ``Blockwise``/``Elemwise`` tokenized *derived* properties, so merely
  *constructing* an ``Elemwise`` ran numpy dtype inference (``_info``) and
  broadcasting (``out_ind``). Because every rewrite mints a fresh node, this
  ran hundreds of times per ``optimize()``. ``Elemwise`` now tokenizes its raw
  operands, which fully determine it.
- ``dtype`` -> ``_meta`` -> ``ndim`` computed ``ndim`` as ``len(shape)``, which
  pulls the full ``chunks`` (chunk alignment) just to count dimensions.
  ``Blockwise.ndim`` is now ``len(out_ind)``.

The ``vars()`` checks mirror ``test_stable_names.py`` -- a cached_property only
lands in the instance ``__dict__`` once it has actually been computed.
"""

from __future__ import annotations

import numpy as np

import dask_array as da
from dask_array._blockwise import Elemwise
from dask_array._test_utils import assert_eq

_META = {"_info", "out_ind", "chunks", "shape", "_meta", "dtype", "ndim"}


def _cached(expr):
    return {k for k in vars(expr) if k in _META}


def test_constructing_elemwise_does_not_infer_metadata():
    x = da.ones((512, 512), chunks=(64, 64))
    y = da.ones((512, 512), chunks=(64, 64))
    expr = (x + y).expr
    assert isinstance(expr, Elemwise)
    # Naming happens eagerly in __new__; it must not run dtype inference
    # (_info) or broadcasting (out_ind), nor compute any chunk metadata.
    assert _cached(expr) == set()


def test_elemwise_dtype_does_not_compute_chunks():
    x = da.ones((512, 512), chunks=(64, 64))
    y = da.ones((512, 512), chunks=(64, 64))
    expr = (x * 2 + y).expr
    assert expr.dtype == np.float64
    assert "chunks" not in vars(expr)
    assert "shape" not in vars(expr)


def test_blockwise_dtype_does_not_compute_chunks():
    x = da.ones((512, 512), chunks=(64, 64))
    expr = x.map_blocks(lambda b: b + 1, dtype=x.dtype).expr
    assert expr.dtype == np.float64
    assert "chunks" not in vars(expr)
    assert "shape" not in vars(expr)


def test_ndim_does_not_compute_chunks():
    x = da.ones((512, 512), chunks=(64, 64))
    expr = x.map_blocks(lambda b: b + 1, dtype=x.dtype).expr
    assert expr.ndim == 2
    assert "chunks" not in vars(expr)
    assert "shape" not in vars(expr)


def test_decoupling_preserves_results_and_metadata():
    x = da.ones((10, 8), chunks=(5, 4))
    y = da.ones((10, 8), chunks=(5, 4))
    z = (x + y * 2).map_blocks(lambda b: b + 1, dtype="float64")[2:9, 1:7]
    assert z.dtype == np.float64
    assert z.shape == (7, 6)
    assert z.ndim == 2
    expected = (np.ones((10, 8)) * 3 + 1)[2:9, 1:7]
    assert_eq(z, expected)


def test_explicit_dtype_changes_identity():
    # The token drops the *inferred* dtype but keeps the raw dtype operand,
    # so an explicit dtype still yields a distinct name (and result dtype).
    x = da.ones((10, 8), chunks=(5, 4))
    y = da.ones((10, 8), chunks=(5, 4))
    default = da.add(x, y)
    cast = da.add(x, y, dtype="float32")
    assert default._name != cast._name
    assert cast.dtype == np.float32
