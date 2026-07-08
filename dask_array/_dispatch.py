"""
Dispatch registries for dask_array.

This module provides Dispatch objects for array operations that need to be
dispatched based on array type (numpy, cupy, sparse, etc.).

concatenate_lookup and tensordot_lookup are defined in _core_utils.py but
re-exported here for convenience.
"""

from __future__ import annotations

import inspect
import sys
import types

import numpy as np

from dask.tokenize import (
    _normalize_pickle,
    normalize_bound_method,
    normalize_object,
    normalize_token,
)
from dask.utils import Dispatch

# Re-export from _core_utils for convenience
from dask_array._core_utils import concatenate_lookup, tensordot_lookup


# dask itself only registers a masked-array tokenize handler when legacy
# dask.array is imported; without it, the generic ndarray normalizer calls
# .view("i1"), which numpy.ma rejects, so tokenizing (e.g. from_array on) a
# masked source crashes. Register the same handler here so this package
# never depends on the legacy import.
@normalize_token.register(np.ma.masked_array)
def _normalize_masked_array(x):
    return (normalize_token(x.data), normalize_token(x.mask), normalize_token(x.fill_value))


# --- Qualname tokenization of importable callables ---------------------------
#
# dask's fallback for a class / function / ufunc / np-dispatcher is
# `_normalize_pickle`, which cloudpickles the object (plus a sys.modules scan).
# During dask-array optimize every rewritten Expr re-tokenizes its block funcs,
# so a tiny set of module-level callables (np.where, operator.mul, Elemwise,
# ...) gets pickled hundreds of thousands of times.
#
# For an object that is importable by reference, its pickle IS just a
# `(module, qualname)` global lookup -- so we normalize it directly to that
# tuple: cheap, and cross-process deterministic (qualname is stable where a
# pickle-bytes hash and its module scan are not). "Importable" is verified
# strictly: import the module, walk the qualname by getattr, and confirm the
# result `is` the original object -- exactly the round-trip pickle-by-reference
# relies on. Anything that fails (closures, lambdas, dynamically built classes,
# objects rebound away from their qualname) falls back to the existing pickle
# path, so it tokenizes exactly as before.


def _importable_ref(o):
    """Return ``("importable", module, qualname)`` if ``o`` round-trips by
    import, else ``None``.

    The ``is o`` identity check is the whole correctness guarantee: only one
    object can live at ``module.qualname``, so two distinct objects can never
    share a reference token, and an object rebound away from its qualname (so
    the path resolves to something else) is rejected and pickled instead.
    """
    module = getattr(o, "__module__", None)
    qualname = getattr(o, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        return None
    # `__main__` round-trips within one interpreter but names a different object
    # in another (a fresh client, or the frisky scheduler re-tokenizing the
    # submitted expression) -- a reference token would then collide or diverge.
    # dask/cloudpickle pickle `__main__` objects by value for the same reason,
    # so fall through to that path.
    if module == "__main__":
        return None
    # `<locals>` (nested defs / closures) and `<lambda>` never round-trip.
    if "<" in qualname:
        return None
    # `sys.modules.get`, not `import_module`: tokenizing must never trigger an
    # import as a side effect, and a not-yet-loaded module correctly falls
    # through to pickle-by-value. It also avoids import_module's ~5-10us
    # per-call overhead -- about the cost of the pickle-by-reference it
    # replaces -- which otherwise erases the win on hot re-tokenization.
    obj = sys.modules.get(module)
    if obj is None:
        return None
    try:
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        return None
    if obj is o:
        return "importable", module, qualname
    return None


@normalize_token.register(type)
def _normalize_type(o):
    # Non-importable classes keep dask's object handler (slotnames / dataclass
    # / pickle), not bare pickle -- it is the correct fallback for a type.
    return _importable_ref(o) or normalize_object(o)


@normalize_token.register(types.FunctionType)
def _normalize_function(func):
    return _importable_ref(func) or _normalize_pickle(func)


@normalize_token.register(types.BuiltinFunctionType)
def _normalize_builtin(func):
    # BuiltinMethodType is BuiltinFunctionType; a builtin bound to a non-module
    # instance (e.g. an ndarray method) must tokenize by its receiver, as dask
    # does -- only module-level builtins (operator.mul, ...) are importable.
    self = getattr(func, "__self__", None)
    if self is not None and not inspect.ismodule(self):
        return normalize_bound_method(func)
    return _importable_ref(func) or _normalize_pickle(func)


# dask registers its numpy tokenize handlers (np.ndarray, np.ufunc, ...)
# lazily -- the first time a numpy object is tokenized (see
# `normalize_token.register_lazy("numpy")`). Force that to happen now, so the
# np.ufunc handler below registers AFTER dask's and wins; otherwise a later
# numpy tokenize fires the lazy hook and clobbers ours. Idempotent: if the hook
# already fired, this just returns the cached handler.
normalize_token.dispatch(np.ufunc)


@normalize_token.register(np.ufunc)
def _normalize_ufunc(func):
    return _importable_ref(func) or _normalize_pickle(func)


# np.where / np.sum / ... are `numpy._ArrayFunctionDispatcher` instances.
@normalize_token.register(type(np.where))
def _normalize_array_function_dispatcher(func):
    return _importable_ref(func) or _normalize_pickle(func)


# Dispatch registries for array operations
take_lookup = Dispatch("take")
einsum_lookup = Dispatch("einsum")
empty_lookup = Dispatch("empty")
divide_lookup = Dispatch("divide")
percentile_lookup = Dispatch("percentile")
numel_lookup = Dispatch("numel")
nannumel_lookup = Dispatch("nannumel")


# --- numpy implementations ---


def _divide(x1, x2, out=None, dtype=None):
    """Implementation of numpy.divide that works with dtype kwarg."""
    x = np.divide(x1, x2, out)
    if dtype is not None:
        x = x.astype(dtype)
    return x


def _percentile(a, q, method="linear"):
    """
    Chunk-level percentile calculation.

    Returns (percentile_values, n) tuple where n is the number of elements.
    Used for combining percentiles from multiple chunks.
    """
    from collections.abc import Iterator

    n = len(a)
    if not len(a):
        return None, n
    if isinstance(q, Iterator):
        q = list(q)
    if a.dtype.name == "category":
        result = np.percentile(a.cat.codes, q, method=method)
        import pandas as pd

        return (
            pd.Categorical.from_codes(result, a.dtype.categories, a.dtype.ordered),
            n,
        )
    if type(a.dtype).__name__ == "DatetimeTZDtype":
        import pandas as pd

        if isinstance(a, (pd.Series, pd.Index)):
            a = a.values

    if np.issubdtype(a.dtype, np.datetime64):
        values = a
        if type(a).__name__ in ("Series", "Index"):
            a2 = values.astype("i8")
        else:
            a2 = values.view("i8")
        result = np.percentile(a2, q, method=method).astype(values.dtype)
        if q[0] == 0:
            # https://github.com/dask/dask/issues/6864
            result[0] = min(result[0], values.min())
        return result, n
    if not np.issubdtype(a.dtype, np.number):
        method = "nearest"
    return np.percentile(a, q, method=method), n


def _numel(x, **kwargs):
    """
    A reduction to count the number of elements.

    Returns ndarray result (coerces to numpy).
    """
    import math

    shape = x.shape
    keepdims = kwargs.get("keepdims", False)
    axis = kwargs.get("axis")
    dtype = kwargs.get("dtype", np.float64)

    if axis is None:
        prod = np.prod(shape, dtype=dtype)
        if keepdims is False:
            return prod

        return np.full(shape=(1,) * len(shape), fill_value=prod, dtype=dtype)

    if not isinstance(axis, (tuple, list)):
        axis = [axis]

    prod = math.prod(shape[dim] for dim in axis)
    if keepdims is True:
        new_shape = tuple(shape[dim] if dim not in axis else 1 for dim in range(len(shape)))
    else:
        new_shape = tuple(shape[dim] for dim in range(len(shape)) if dim not in axis)

    return np.broadcast_to(np.array(prod, dtype=dtype), new_shape)


def _nannumel(x, **kwargs):
    """A reduction to count the number of elements, excluding nans"""
    return np.sum(~(np.isnan(x)), **kwargs)


# --- Register numpy implementations ---

take_lookup.register((object, np.ndarray, np.ma.masked_array), np.take)
einsum_lookup.register((object, np.ndarray), np.einsum)
empty_lookup.register((object, np.ndarray), np.empty)
empty_lookup.register(np.ma.masked_array, np.ma.empty)
divide_lookup.register((object, np.ndarray), _divide)
divide_lookup.register(np.ma.masked_array, np.ma.divide)
percentile_lookup.register(np.ndarray, _percentile)
numel_lookup.register((object, np.ndarray), _numel)
nannumel_lookup.register((object, np.ndarray), _nannumel)


# --- Register masked array numel ---


@numel_lookup.register(np.ma.masked_array)
def _numel_masked(x, **kwargs):
    """Numel implementation for masked arrays."""
    return np.sum(np.ones_like(x), **kwargs)
