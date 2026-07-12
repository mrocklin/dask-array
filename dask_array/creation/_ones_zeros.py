from __future__ import annotations

import functools
from functools import partial

import numpy as np

from dask_array._new_collection import new_collection
from dask._task_spec import Task
from dask_array._collection import asarray
from dask_array._expr import ArrayExpr
from dask_array._utils import meta_from_array

from ._utils import _get_like_function_shapes_chunks, _parse_wrap_args, broadcast_trick


class BroadcastTrick(ArrayExpr):
    _parameters = ["shape", "dtype", "chunks", "meta", "kwargs", "name"]
    _defaults = {"meta": None, "name": None}
    _is_blockwise_fusable = True

    @functools.cached_property
    def _name(self):
        custom_name = self.operand("name")
        if custom_name is not None:
            return custom_name
        return f"{self._funcname}-{self.deterministic_token}"

    @functools.cached_property
    def _meta(self):
        return meta_from_array(self.operand("meta"), ndim=self.ndim, dtype=self.operand("dtype"))

    @functools.cached_property
    def _wrapped_func(self):
        """Cache the wrapped broadcast function."""
        func = broadcast_trick(self.func)
        k = self.kwargs.copy()
        k.pop("meta", None)
        return partial(func, meta=self._meta, dtype=self.dtype, **k)

    def _layer(self) -> dict:
        from itertools import product

        result = {}
        for block_id in product(*[range(len(c)) for c in self.chunks]):
            key = (self._name, *block_id)
            result[key] = self._task(key, block_id)
        return result

    def _frisky_layer(self):
        from dask_array._frisky.creation import CreationLayer

        try:
            chunks = [[int(s) for s in c] for c in self.chunks]
        except (TypeError, ValueError):
            # Unknown (nan) chunk sizes can't be expanded; fall back.
            raise NotImplementedError("non-concrete chunks")
        return CreationLayer(self._name, self._wrapped_func, chunks)

    def _task(self, key, block_id: tuple[int, ...]) -> Task:
        """Generate task for a specific output block."""
        # Compute chunk shape for this block
        chunk_shape = tuple(self.chunks[i][block_id[i]] for i in range(len(block_id)))
        return Task(key, self._wrapped_func, chunk_shape)

    def _input_block_id(self, dep, block_id: tuple[int, ...]) -> tuple[int, ...]:
        """BroadcastTrick has no dependencies, so this is never called."""
        return block_id

    def _simplify_up(self, parent, dependents):
        """Allow slice and shuffle operations to simplify BroadcastTrick."""
        from dask_array._shuffle import Shuffle
        from dask_array.slicing import SliceSlicesIntegers

        if isinstance(parent, SliceSlicesIntegers):
            return self._slice_pushdown(parent, dependents)
        if isinstance(parent, Shuffle):
            return self._shuffle_pushdown(parent, dependents)
        return None

    def _accept_shuffle(self, shuffle_expr):
        """Accept a shuffle - create new BroadcastTrick with shuffled shape.

        Since all values are identical, we don't need to actually shuffle,
        just create a new constant array. The result replaces the shuffle
        node, so it must carry the shuffle's exact output shape and chunks —
        rewrites must never change user-visible metadata.
        """
        return self.substitute_parameters(
            {
                "shape": shuffle_expr.shape,
                "chunks": shuffle_expr.chunks,
                "name": None,
            }
        )

    def _accept_slice(self, slice_expr):
        """Accept a slice by creating a smaller BroadcastTrick.

        A creation expression (ones/zeros/full/empty) has the same constant
        value at every position, so slicing it by *any* index -- contiguous,
        strided, integer, or negative-step -- just yields a smaller creation
        of the same value.  ``SliceSlicesIntegers`` already computes the
        result's output shape and chunks (dropping integer-indexed dims,
        counting strided elements), so we read them straight off the slice
        node rather than recomputing per dimension.  This is what lets a
        single integer or strided index cull the graph down to the reachable
        blocks instead of leaving every original block behind an unreachable
        ``getitem``.

        ``slice_expr`` is always a ``SliceSlicesIntegers`` whose index holds
        only slices and integers; any ``None`` newaxis is peeled into an
        enclosing ``ExpandDims`` before this pushdown runs, so it never
        reaches here.
        """
        return self.substitute_parameters(
            {
                "shape": slice_expr.shape,
                "chunks": slice_expr.chunks,
                "name": None,
            }
        )


class Ones(BroadcastTrick):
    func = staticmethod(np.ones_like)


class Zeros(BroadcastTrick):
    func = staticmethod(np.zeros_like)


class Empty(BroadcastTrick):
    func = staticmethod(np.empty_like)


class Full(BroadcastTrick):
    func = staticmethod(np.full_like)


def wrap_func_shape_as_first_arg(*args, klass, **kwargs):
    """
    Transform np creation function into blocked version
    """
    if "shape" not in kwargs:
        shape, args = args[0], args[1:]
    else:
        shape = kwargs.pop("shape")

    if isinstance(shape, ArrayExpr):
        raise TypeError("Dask array input not supported. Please use tuple, list, or a 1D numpy array instead.")

    name = kwargs.pop("name", None)
    parsed = _parse_wrap_args(klass.func, args, kwargs, shape)
    return new_collection(
        klass(
            parsed["shape"],
            parsed["dtype"],
            parsed["chunks"],
            kwargs.get("meta"),
            kwargs,
            name,
        )
    )


def wrap(func, **kwargs):
    return partial(func, **kwargs)


ones = wrap(wrap_func_shape_as_first_arg, klass=Ones, dtype="f8")
zeros = wrap(wrap_func_shape_as_first_arg, klass=Zeros, dtype="f8")
empty = wrap(wrap_func_shape_as_first_arg, klass=Empty, dtype="f8")
_full = wrap(wrap_func_shape_as_first_arg, klass=Full, dtype="f8")


def empty_like(a, dtype=None, order="C", chunks=None, name=None, shape=None):
    """
    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.
    chunks : sequence of ints
        The number of samples on each block. Note that the last block will have
        fewer samples if ``len(array) % chunks != 0``.
    name : str, optional
        An optional keyname for the array. Defaults to hashing the input
        keyword arguments.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data with the same
        shape and type as `a`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.

    Notes
    -----
    This function does *not* initialize the returned array; to do that use
    `zeros_like` or `ones_like` instead.  It may be marginally faster than
    the functions that do set the array values.
    """

    a = asarray(a, name=False)
    shape, chunks = _get_like_function_shapes_chunks(a, chunks, shape)

    # if shape is nan we cannot rely on regular empty function, we use
    # generic map_blocks.
    if np.isnan(shape).any():
        return a.map_blocks(partial(np.empty_like, dtype=(dtype or a.dtype)))

    return empty(
        shape,
        dtype=(dtype or a.dtype),
        order=order,
        chunks=chunks,
        name=name,
        meta=a._meta,
    )


def ones_like(a, dtype=None, order="C", chunks=None, name=None, shape=None):
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.
    chunks : sequence of ints
        The number of samples on each block. Note that the last block will have
        fewer samples if ``len(array) % chunks != 0``.
    name : str, optional
        An optional keyname for the array. Defaults to hashing the input
        keyword arguments.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result.

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as `a`.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.
    """

    a = asarray(a, name=False)
    shape, chunks = _get_like_function_shapes_chunks(a, chunks, shape)

    # if shape is nan we cannot rely on regular ones function, we use
    # generic map_blocks.
    if np.isnan(shape).any():
        return a.map_blocks(partial(np.ones_like, dtype=(dtype or a.dtype)))

    return ones(
        shape,
        dtype=(dtype or a.dtype),
        order=order,
        chunks=chunks,
        name=name,
        meta=a._meta,
    )


def zeros_like(a, dtype=None, order="C", chunks=None, name=None, shape=None):
    """
    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.
    chunks : sequence of ints
        The number of samples on each block. Note that the last block will have
        fewer samples if ``len(array) % chunks != 0``.
    name : str, optional
        An optional keyname for the array. Defaults to hashing the input
        keyword arguments.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result.

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.
    """

    a = asarray(a, name=False)
    shape, chunks = _get_like_function_shapes_chunks(a, chunks, shape)

    # if shape is nan we cannot rely on regular zeros function, we use
    # generic map_blocks.
    if np.isnan(shape).any():
        return a.map_blocks(partial(np.zeros_like, dtype=(dtype or a.dtype)))

    return zeros(
        shape,
        dtype=(dtype or a.dtype),
        order=order,
        chunks=chunks,
        name=name,
        meta=a._meta,
    )


def full(shape, fill_value, *args, **kwargs):
    # np.isscalar has somewhat strange behavior:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.isscalar.html
    if np.ndim(fill_value) != 0:
        raise ValueError(f"fill_value must be scalar. Received {type(fill_value).__name__} instead.")
    if kwargs.get("dtype") is None:
        if hasattr(fill_value, "dtype"):
            kwargs["dtype"] = fill_value.dtype
        else:
            kwargs["dtype"] = type(fill_value)
    return _full(*args, shape=shape, fill_value=fill_value, **kwargs)


def full_like(a, fill_value, order="C", dtype=None, chunks=None, name=None, shape=None):
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.
    chunks : sequence of ints
        The number of samples on each block. Note that the last block will have
        fewer samples if ``len(array) % chunks != 0``.
    name : str, optional
        An optional keyname for the array. Defaults to hashing the input
        keyword arguments.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result.

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the same shape and type as `a`.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.
    full : Fill a new array.
    """

    a = asarray(a, name=False)
    shape, chunks = _get_like_function_shapes_chunks(a, chunks, shape)

    # if shape is nan we cannot rely on regular full function, we use
    # generic map_blocks.
    if np.isnan(shape).any():
        return a.map_blocks(partial(np.full_like, dtype=(dtype or a.dtype)), fill_value)

    return full(
        shape,
        fill_value,
        dtype=(dtype or a.dtype),
        order=order,
        chunks=chunks,
        name=name,
        meta=a._meta,
    )
