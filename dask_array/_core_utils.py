"""
Core utility functions extracted from dask.array.core.

This module provides helper functions used throughout dask_array that were
previously imported from dask.array.core.
"""

from __future__ import annotations

import functools
import math
import sys
import traceback
import warnings
from collections.abc import Iterable, Iterator
from itertools import product, zip_longest
from numbers import Integral, Number
from typing import TYPE_CHECKING

import numpy as np
from tlz import first
from toolz import frequencies

from dask import config
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.delayed import delayed
from dask.sizeof import sizeof
from dask.utils import (
    Dispatch,
    cached_cumsum,
    cached_max,
    concrete,
    funcname,
    has_keyword,
    is_arraylike,
    is_integer,
    ndimlist,
    parse_bytes,
)

if TYPE_CHECKING:
    pass

# Type definition
T_IntOrNaN = int | float  # Should be int | Literal[np.nan]


# Error message constant
unknown_chunk_message = (
    "\n\n"
    "A possible solution: "
    "https://docs.dask.org/en/latest/array-chunks.html#unknown-chunks\n"
    "Summary: to compute chunks sizes, use\n\n"
    "   x.compute_chunk_sizes()  # for Dask Array `x`\n"
    "   ddf.to_dask_array(lengths=True)  # for Dask DataFrame `ddf`"
)


class PerformanceWarning(Warning):
    """A warning given when bad chunking may cause poor performance"""


# Dispatch registries for array operations
concatenate_lookup = Dispatch("concatenate")
tensordot_lookup = Dispatch("tensordot")


def getter(a, b, asarray=True, lock=None):
    if isinstance(b, tuple) and any(x is None for x in b):
        b2 = tuple(x for x in b if x is not None)
        b3 = tuple(None if x is None else slice(None, None) for x in b if not isinstance(x, Integral))
        return getter(a, b2, asarray=asarray, lock=lock)[b3]

    if lock:
        lock.acquire()
    try:
        c = a[b]
        # Below we special-case `np.matrix` to force a conversion to
        # `np.ndarray` and preserve original Dask behavior for `getter`,
        # as for all purposes `np.matrix` is array-like and thus
        # `is_arraylike` evaluates to `True` in that case.
        if asarray and (not is_arraylike(c) or isinstance(c, np.matrix)):
            c = np.asarray(c)
    finally:
        if lock:
            lock.release()
    return c


def getter_nofancy(a, b, asarray=True, lock=None):
    """A simple wrapper around ``getter``.

    Used to indicate to the optimization passes that the backend doesn't
    support fancy indexing.
    """
    return getter(a, b, asarray=asarray, lock=lock)


def getter_inline(a, b, asarray=True, lock=None):
    """A getter function that optimizations feel comfortable inlining

    Slicing operations with this function may be inlined into a graph, such as
    in the following rewrite

    **Before**

    >>> a = x[:10]  # doctest: +SKIP
    >>> b = a + 1  # doctest: +SKIP
    >>> c = a * 2  # doctest: +SKIP

    **After**

    >>> b = x[:10] + 1  # doctest: +SKIP
    >>> c = x[:10] * 2  # doctest: +SKIP

    This inlining can be relevant to operations when running off of disk.
    """
    return getter(a, b, asarray=asarray, lock=lock)


def slices_from_chunks(chunks):
    """Translate chunks tuple to a set of slices in product order

    >>> slices_from_chunks(((2, 2), (3, 3, 3)))  # doctest: +NORMALIZE_WHITESPACE
     [(slice(0, 2, None), slice(0, 3, None)),
      (slice(0, 2, None), slice(3, 6, None)),
      (slice(0, 2, None), slice(6, 9, None)),
      (slice(2, 4, None), slice(0, 3, None)),
      (slice(2, 4, None), slice(3, 6, None)),
      (slice(2, 4, None), slice(6, 9, None))]
    """
    cumdims = [cached_cumsum(bds, initial_zero=True) for bds in chunks]
    slices = [[slice(s, s + dim) for s, dim in zip(starts, shapes)] for starts, shapes in zip(cumdims, chunks)]
    return list(product(*slices))


def graph_from_arraylike(
    arr,  # Any array-like which supports slicing
    chunks,
    shape,
    name,
    getitem=None,
    lock=False,
    asarray=True,
    dtype=None,
    inline_array=False,
):
    """
    Generate a graph for slicing chunks from an array-like.

    Returns a dict-based graph (not HighLevelGraph) for use with expression system.
    """
    from dask._task_spec import TaskRef

    if getitem is None:
        getitem = getter

    chunks = normalize_chunks(chunks, shape, dtype=dtype)

    if has_keyword(getitem, "asarray") and has_keyword(getitem, "lock") and (not asarray or lock):
        kwargs = {"asarray": asarray, "lock": lock}
    else:
        # Common case, drop extra parameters
        kwargs = {}

    if inline_array:
        # Embed the array directly in each task
        graph = {}
        for idx, slc in zip(product(*[range(len(c)) for c in chunks]), slices_from_chunks(chunks)):
            key = (name,) + idx
            if kwargs:
                graph[key] = (getitem, arr, slc, kwargs.get("asarray", True), kwargs.get("lock", None))
            else:
                graph[key] = (getitem, arr, slc)
        return graph
    else:
        # Store array separately and reference it
        original_name = f"original-{name}"
        graph = {original_name: arr}
        for idx, slc in zip(product(*[range(len(c)) for c in chunks]), slices_from_chunks(chunks)):
            key = (name,) + idx
            if kwargs:
                graph[key] = (
                    getitem,
                    TaskRef(original_name),
                    slc,
                    kwargs.get("asarray", True),
                    kwargs.get("lock", None),
                )
            else:
                graph[key] = (getitem, TaskRef(original_name), slc)
        return graph


def _concatenate2(arrays, axes=None):
    """Recursively concatenate nested lists of arrays along axes

    Each entry in axes corresponds to each level of the nested list.  The
    length of axes should correspond to the level of nesting of arrays.
    If axes is an empty list or tuple, return arrays, or arrays[0] if
    arrays is a list.

    >>> x = np.array([[1, 2], [3, 4]])
    >>> _concatenate2([x, x], axes=[0])
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]])

    >>> _concatenate2([x, x], axes=[1])
    array([[1, 2, 1, 2],
           [3, 4, 3, 4]])

    >>> _concatenate2([[x, x], [x, x]], axes=[0, 1])
    array([[1, 2, 1, 2],
           [3, 4, 3, 4],
           [1, 2, 1, 2],
           [3, 4, 3, 4]])

    Supports Iterators
    >>> _concatenate2(iter([x, x]), axes=[1])
    array([[1, 2, 1, 2],
           [3, 4, 3, 4]])

    Special Case
    >>> _concatenate2([x, x], axes=())
    array([[1, 2],
           [3, 4]])
    """
    if axes is None:
        axes = []

    if axes == ():
        if isinstance(arrays, list):
            return arrays[0]
        else:
            return arrays

    if isinstance(arrays, Iterator):
        arrays = list(arrays)
    if not isinstance(arrays, (list, tuple)):
        return arrays
    if len(axes) > 1:
        arrays = [_concatenate2(a, axes=axes[1:]) for a in arrays]
    concatenate = concatenate_lookup.dispatch(type(max(arrays, key=lambda x: getattr(x, "__array_priority__", 0))))
    if isinstance(arrays[0], dict):
        # Handle concatenation of `dict`s, used as a replacement for structured
        # arrays when that's not supported by the array library (e.g., CuPy).
        keys = list(arrays[0].keys())
        assert all(list(a.keys()) == keys for a in arrays)
        ret = dict()
        for k in keys:
            ret[k] = concatenate(list(a[k] for a in arrays), axis=axes[0])
        return ret
    else:
        return concatenate(arrays, axis=axes[0])


def apply_infer_dtype(func, args, kwargs, funcname, suggest_dtype="dtype", nout=None):
    """
    Tries to infer output dtype of ``func`` for a small set of input arguments.

    Parameters
    ----------
    func: Callable
        Function for which output dtype is to be determined

    args: List of array like
        Arguments to the function, which would usually be used. Only attributes
        ``ndim`` and ``dtype`` are used.

    kwargs: dict
        Additional ``kwargs`` to the ``func``

    funcname: String
        Name of calling function to improve potential error messages

    suggest_dtype: None/False or String
        If not ``None`` adds suggestion to potential error message to specify a dtype
        via the specified kwarg. Defaults to ``'dtype'``.

    nout: None or Int
        ``None`` if function returns single output, integer if many.
        Defaults to ``None``.

    Returns
    -------
    : dtype or List of dtype
        One or many dtypes (depending on ``nout``)
    """
    from dask_array._utils import meta_from_array

    # make sure that every arg is an evaluated array
    args = [
        (np.zeros_like(meta_from_array(x), shape=((1,) * x.ndim), dtype=x.dtype) if is_arraylike(x) else x)
        for x in args
    ]
    try:
        with np.errstate(all="ignore"):
            o = func(*args, **kwargs)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = "".join(traceback.format_tb(exc_traceback))
        suggest = (
            (f"Please specify the dtype explicitly using the `{suggest_dtype}` kwarg.\n\n") if suggest_dtype else ""
        )
        msg = (
            f"`dtype` inference failed in `{funcname}`.\n\n"
            f"{suggest}"
            "Original error is below:\n"
            "------------------------\n"
            f"{e!r}\n\n"
            "Traceback:\n"
            "---------\n"
            f"{tb}"
        )
    else:
        msg = None
    if msg is not None:
        raise ValueError(msg)
    return getattr(o, "dtype", type(o)) if nout is None else tuple(e.dtype for e in o)


def normalize_arg(x):
    """Normalize user provided arguments to blockwise or map_blocks

    We do a few things:

    1.  If they are string literals that might collide with blockwise_token then we
        quote them
    2.  IF they are large (as defined by sizeof) then we put them into the
        graph on their own by using dask.delayed
    """
    import re

    if is_dask_collection(x):
        return x
    elif isinstance(x, str) and re.match(r"_\d+", x):
        return delayed(x)
    elif isinstance(x, list) and len(x) >= 10:
        return delayed(x)
    elif sizeof(x) > 1e6:
        return delayed(x)
    else:
        return x


def _pass_extra_kwargs(func, keys, *args, **kwargs):
    """Helper for :func:`dask.array.map_blocks` to pass `block_info` or `block_id`.

    For each element of `keys`, a corresponding element of args is changed
    to a keyword argument with that key, before all arguments re passed on
    to `func`.
    """
    kwargs.update(zip(keys, args))
    return func(*args[len(keys) :], **kwargs)


def apply_and_enforce(*args, **kwargs):
    """Apply a function, and enforce the output.ndim to match expected_ndim

    Ensures the output has the expected dimensionality."""
    func = kwargs.pop("_func")
    expected_ndim = kwargs.pop("expected_ndim")
    out = func(*args, **kwargs)
    if getattr(out, "ndim", 0) != expected_ndim:
        out_ndim = getattr(out, "ndim", 0)
        raise ValueError(
            f"Dimension mismatch: expected output of {func} to have dims = {expected_ndim}.  Got {out_ndim} instead."
        )
    return out


def broadcast_chunks(*chunkss):
    """Construct a chunks tuple that broadcasts many chunks tuples

    >>> a = ((5, 5),)
    >>> b = ((5, 5),)
    >>> broadcast_chunks(a, b)
    ((5, 5),)

    >>> a = ((10, 10, 10), (5, 5),)
    >>> b = ((5, 5),)
    >>> broadcast_chunks(a, b)
    ((10, 10, 10), (5, 5))

    >>> a = ((10, 10, 10), (5, 5),)
    >>> b = ((1,), (5, 5),)
    >>> broadcast_chunks(a, b)
    ((10, 10, 10), (5, 5))

    >>> a = ((10, 10, 10), (5, 5),)
    >>> b = ((3, 3,), (5, 5),)
    >>> broadcast_chunks(a, b)
    Traceback (most recent call last):
        ...
    ValueError: Chunks do not align: [(10, 10, 10), (3, 3)]
    """
    if not chunkss:
        return ()
    elif len(chunkss) == 1:
        return chunkss[0]
    n = max(map(len, chunkss))
    chunkss2 = [((1,),) * (n - len(c)) + c for c in chunkss]
    result = []
    for i in range(n):
        step1 = [c[i] for c in chunkss2]
        if all(c == (1,) for c in step1):
            step2 = step1
        else:
            step2 = [c for c in step1 if c != (1,)]
        if len(set(step2)) != 1:
            raise ValueError(f"Chunks do not align: {step2}")
        result.append(step2[0])
    return tuple(result)


CHUNKS_NONE_ERROR_MESSAGE = """
You must specify a chunks= keyword argument.
This specifies the chunksize of your array blocks.

See the following documentation page for details:
  https://docs.dask.org/en/latest/array-creation.html#chunks
""".strip()


def blockdims_from_blockshape(shape, chunks):
    """
    >>> blockdims_from_blockshape((10, 10), (4, 3))
    ((4, 4, 2), (3, 3, 3, 1))
    >>> blockdims_from_blockshape((10, 0), (4, 0))
    ((4, 4, 2), (0,))
    """
    if chunks is None:
        raise TypeError("Must supply chunks= keyword argument")
    if shape is None:
        raise TypeError("Must supply shape= keyword argument")
    if np.isnan(sum(shape)) or np.isnan(sum(chunks)):
        raise ValueError(f"Array chunk sizes are unknown. shape: {shape}, chunks: {chunks}{unknown_chunk_message}")
    if not all(map(is_integer, chunks)):
        raise ValueError("chunks can only contain integers.")
    if not all(map(is_integer, shape)):
        raise ValueError("shape can only contain integers.")
    shape = tuple(map(int, shape))
    chunks = tuple(map(int, chunks))
    return tuple(((bd,) * (d // bd) + ((d % bd,) if d % bd else ()) if d else (0,)) for d, bd in zip(shape, chunks))


def _convert_int_chunk_to_tuple(shape, chunks):
    return sum(
        (
            (blockdims_from_blockshape((s,), (c,)) if not isinstance(c, (tuple, list)) else (c,))
            for s, c in zip(shape, chunks)
        ),
        (),
    )


def _compute_multiplier(limit: int, dtype, largest_block: int, result):
    """
    Utility function for auto_chunk, to find how much larger or smaller the ideal
    chunk size is relative to what we have now.
    """
    return (
        limit
        / dtype.itemsize
        / largest_block
        / math.prod(max(r) if isinstance(r, tuple) else r for r in result.values() if r)
    )


def round_to(c, s):
    """Return a chunk dimension that is close to an even multiple or factor

    We want values for c that are nicely aligned with s.

    If c is smaller than s we use the original chunk size and accept an
    uneven chunk at the end.

    If c is larger than s then we want the largest multiple of s that is still
    smaller than c.
    """
    if c <= s:
        return max(1, int(c))
    else:
        return c // s * s


def auto_chunks(chunks, shape, limit, dtype, previous_chunks=None):
    """Determine automatic chunks

    This takes in a chunks value that contains ``"auto"`` values in certain
    dimensions and replaces those values with concrete dimension sizes that try
    to get chunks to be of a certain size in bytes, provided by the ``limit=``
    keyword.  If multiple dimensions are marked as ``"auto"`` then they will
    all respond to meet the desired byte limit, trying to respect the aspect
    ratio of their dimensions in ``previous_chunks=``, if given.

    Parameters
    ----------
    chunks: Tuple
        A tuple of either dimensions or tuples of explicit chunk dimensions
        Some entries should be "auto"
    shape: Tuple[int]
    limit: int, str
        The maximum allowable size of a chunk in bytes
    previous_chunks: Tuple[Tuple[int]]

    See also
    --------
    normalize_chunks: for full docstring and parameters
    """
    if previous_chunks is not None:
        # rioxarray is passing ((1, ), (x,)) for shapes like (100, 5x),
        # so add this compat code for now
        # https://github.com/corteva/rioxarray/pull/820
        previous_chunks = (c[0] if isinstance(c, tuple) and len(c) == 1 else c for c in previous_chunks)
        previous_chunks = _convert_int_chunk_to_tuple(shape, previous_chunks)
    chunks = list(chunks)

    autos = {i for i, c in enumerate(chunks) if c == "auto"}
    if not autos:
        return tuple(chunks)

    if limit is None:
        limit = config.get("array.chunk-size")
    if isinstance(limit, str):
        limit = parse_bytes(limit)

    if dtype is None:
        raise TypeError("dtype must be known for auto-chunking")

    if dtype.hasobject:
        raise NotImplementedError(
            "Can not use auto rechunking with object dtype. We are unable to estimate the size in bytes of object data"
        )

    for x in tuple(chunks) + tuple(shape):
        if isinstance(x, Number) and np.isnan(x) or isinstance(x, tuple) and np.isnan(x).any():
            raise ValueError(
                f"Can not perform automatic rechunking with unknown (nan) chunk sizes.{unknown_chunk_message}"
            )

    limit = max(1, limit)
    chunksize_tolerance = config.get("array.chunk-size-tolerance")

    largest_block = math.prod(cs if isinstance(cs, Number) else max(cs) for cs in chunks if cs != "auto")

    if previous_chunks:
        # Base ideal ratio on the median chunk size of the previous chunks
        median_chunks = {a: np.median(previous_chunks[a]) for a in autos}
        result = {}

        # How much larger or smaller the ideal chunk size is relative to what we have now
        multiplier = _compute_multiplier(limit, dtype, largest_block, median_chunks)
        if multiplier < 1:
            # we want to update inplace, algorithm relies on it in this case
            result = median_chunks

        ideal_shape = []
        for i, s in enumerate(shape):
            chunk_frequencies = frequencies(previous_chunks[i])
            mode, count = max(chunk_frequencies.items(), key=lambda kv: kv[1])
            if mode > 1 and count >= len(previous_chunks[i]) / 2:
                ideal_shape.append(mode)
            else:
                ideal_shape.append(s)

        def _trivial_aggregate(a):
            autos.remove(a)
            del median_chunks[a]
            return True

        multiplier_remaining = True
        reduce_case = multiplier < 1
        while multiplier_remaining:  # while things change
            last_autos = set(autos)  # record previous values
            multiplier_remaining = False

            # Expand or contract each of the dimensions appropriately
            for a in sorted(autos):
                this_multiplier = multiplier ** (1 / len(last_autos))

                proposed = median_chunks[a] * this_multiplier
                this_chunksize_tolerance = chunksize_tolerance ** (1 / len(last_autos))
                max_chunk_size = proposed * this_chunksize_tolerance

                if proposed > shape[a]:  # we've hit the shape boundary
                    chunks[a] = shape[a]
                    multiplier_remaining = _trivial_aggregate(a)
                    largest_block *= shape[a]
                    result[a] = (shape[a],)
                    continue
                elif reduce_case or max(previous_chunks[a]) > max_chunk_size:
                    result[a] = round_to(proposed, ideal_shape[a])
                    if proposed < 1:
                        multiplier_remaining = True
                        autos.discard(a)
                    continue
                else:
                    dimension_result, new_chunk = [], 0
                    for c in previous_chunks[a]:
                        if c + new_chunk <= proposed:
                            # keep increasing the chunk
                            new_chunk += c
                        else:
                            # We reach the boundary so start a new chunk
                            if new_chunk > 0:
                                dimension_result.append(new_chunk)
                            new_chunk = c
                    if new_chunk > 0:
                        dimension_result.append(new_chunk)

                result[a] = tuple(dimension_result)

            # recompute how much multiplier we have left, repeat
            if multiplier_remaining or reduce_case:
                last_multiplier = multiplier
                multiplier = _compute_multiplier(limit, dtype, largest_block, median_chunks)
                if multiplier != last_multiplier:
                    multiplier_remaining = True

        for k, v in result.items():
            chunks[k] = v if v else 0
        return tuple(chunks)

    else:
        # Check if dtype.itemsize is greater than 0
        if dtype.itemsize == 0:
            raise ValueError(
                "auto-chunking with dtype.itemsize == 0 is not supported, please pass in `chunks` explicitly"
            )
        size = (limit / dtype.itemsize / largest_block) ** (1 / len(autos))
        small = [i for i in autos if shape[i] < size]
        if small:
            for i in small:
                chunks[i] = (shape[i],)
            return auto_chunks(chunks, shape, limit, dtype)

        for i in autos:
            chunks[i] = round_to(size, shape[i])

        return tuple(chunks)


@functools.lru_cache
def normalize_chunks_cached(chunks, shape=None, limit=None, dtype=None, previous_chunks=None):
    """Cached version of normalize_chunks.

    .. note::

        chunks and previous_chunks are expected to be hashable. Dicts and lists aren't
        allowed for this function.

    See :func:`normalize_chunks` for further documentation.
    """
    return normalize_chunks(chunks, shape=shape, limit=limit, dtype=dtype, previous_chunks=previous_chunks)


def normalize_chunks(chunks, shape=None, limit=None, dtype=None, previous_chunks=None):
    """Normalize chunks to tuple of tuples

    This takes in a variety of input types and information and produces a full
    tuple-of-tuples result for chunks, suitable to be passed to Array or
    rechunk or any other operation that creates a Dask array.

    Parameters
    ----------
    chunks: tuple, int, dict, or string
        The chunks to be normalized.  See examples below for more details
    shape: Tuple[int]
        The shape of the array
    limit: int (optional)
        The maximum block size to target in bytes,
        if freedom is given to choose
    dtype: np.dtype
    previous_chunks: Tuple[Tuple[int]] optional
        Chunks from a previous array that we should use for inspiration when
        rechunking auto dimensions.  If not provided but auto-chunking exists
        then auto-dimensions will prefer square-like chunk shapes.

    Examples
    --------
    Fully explicit tuple-of-tuples

    >>> normalize_chunks(((2, 2, 1), (2, 2, 2)), shape=(5, 6))
    ((2, 2, 1), (2, 2, 2))

    Specify uniform chunk sizes

    >>> normalize_chunks((2, 2), shape=(5, 6))
    ((2, 2, 1), (2, 2, 2))

    Cleans up missing outer tuple

    >>> normalize_chunks((3, 2), (5,))
    ((3, 2),)

    Cleans up lists to tuples

    >>> normalize_chunks([[2, 2], [3, 3]])
    ((2, 2), (3, 3))

    Expands integer inputs 10 -> (10, 10)

    >>> normalize_chunks(10, shape=(30, 5))
    ((10, 10, 10), (5,))

    Expands dict inputs

    >>> normalize_chunks({0: 2, 1: 3}, shape=(6, 6))
    ((2, 2, 2), (3, 3))

    The values -1 and None get mapped to full size

    >>> normalize_chunks((5, -1), shape=(10, 10))
    ((5, 5), (10,))
    >>> normalize_chunks((5, None), shape=(10, 10))
    ((5, 5), (10,))

    Use the value "auto" to automatically determine chunk sizes along certain
    dimensions.  This uses the ``limit=`` and ``dtype=`` keywords to
    determine how large to make the chunks.  The term "auto" can be used
    anywhere an integer can be used.  See array chunking documentation for more
    information.

    >>> normalize_chunks(("auto",), shape=(20,), limit=5, dtype='uint8')
    ((5, 5, 5, 5),)
    >>> normalize_chunks("auto", (2, 3), dtype=np.int32)
    ((2,), (3,))

    You can also use byte sizes (see :func:`dask.utils.parse_bytes`) in place of
    "auto" to ask for a particular size

    >>> normalize_chunks("1kiB", shape=(2000,), dtype='float32')
    ((256, 256, 256, 256, 256, 256, 256, 208),)

    Respects null dimensions

    >>> normalize_chunks(())
    ()
    >>> normalize_chunks((), ())
    ()
    >>> normalize_chunks((1,), ())
    ()
    >>> normalize_chunks((), shape=(0, 0))
    ((0,), (0,))

    Handles NaNs

    >>> normalize_chunks((1, (np.nan,)), (1, np.nan))
    ((1,), (nan,))
    """
    if dtype and not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    if chunks is None:
        raise ValueError(CHUNKS_NONE_ERROR_MESSAGE)
    if isinstance(chunks, list):
        chunks = tuple(chunks)
    if isinstance(chunks, (Number, str)):
        chunks = (chunks,) * len(shape)
    if isinstance(chunks, dict):
        chunks = tuple(chunks.get(i, None) for i in range(len(shape)))
    if isinstance(chunks, np.ndarray):
        chunks = chunks.tolist()
    if not chunks and shape and all(s == 0 for s in shape):
        chunks = ((0,),) * len(shape)

    if shape and len(shape) == 1 and len(chunks) > 1 and all(isinstance(c, (Number, str)) for c in chunks):
        chunks = (chunks,)

    if shape and len(chunks) != len(shape):
        raise ValueError(f"Chunks and shape must be of the same length/dimension. Got chunks={chunks}, shape={shape}")
    if -1 in chunks or None in chunks:
        chunks = tuple(s if c == -1 or c is None else c for c, s in zip(chunks, shape))

    # If specifying chunk size in bytes, use that value to set the limit.
    # Verify there is only one consistent value of limit or chunk-bytes used.
    for c in chunks:
        if isinstance(c, str) and c != "auto":
            parsed = parse_bytes(c)
            if limit is None:
                limit = parsed
            elif parsed != limit:
                raise ValueError(f"Only one consistent value of limit or chunk is allowed.Used {parsed} != {limit}")
    # Substitute byte limits with 'auto' now that limit is set.
    chunks = tuple("auto" if isinstance(c, str) and c != "auto" else c for c in chunks)

    if any(c == "auto" for c in chunks):
        chunks = auto_chunks(chunks, shape, limit, dtype, previous_chunks)

    allints = None
    if chunks and shape is not None:
        # allints: did we start with chunks as a simple tuple of ints?
        allints = all(isinstance(c, int) for c in chunks)
        chunks = _convert_int_chunk_to_tuple(shape, chunks)
    for c in chunks:
        if not c:
            raise ValueError(
                "Empty tuples are not allowed in chunks. Express zero length dimensions with 0(s) in chunks"
            )

    if not allints and shape is not None:
        if not all(c == s or (math.isnan(c) or math.isnan(s)) for c, s in zip(map(sum, chunks), shape)):
            raise ValueError(f"Chunks do not add up to shape. Got chunks={chunks}, shape={shape}")
    if allints or isinstance(sum(sum(_) for _ in chunks), int):
        # Fastpath for when we already know chunks contains only integers
        return tuple(tuple(ch) for ch in chunks)
    return tuple(tuple(int(x) if not math.isnan(x) else np.nan for x in c) for c in chunks)


def common_blockdim(blockdims):
    """Find the common block dimensions from the list of block dimensions

    Currently only implements the simplest possible heuristic: the common
    block-dimension is the only one that does not span fully span a dimension.
    This is a conservative choice that allows us to avoid potentially very
    expensive rechunking.

    Assumes that each element of the input block dimensions has all the same
    sum (i.e., that they correspond to dimensions of the same size).

    Examples
    --------
    >>> common_blockdim([(3,), (2, 1)])
    (2, 1)
    >>> common_blockdim([(1, 2), (2, 1)])
    (1, 1, 1)
    >>> common_blockdim([(2, 2), (3, 1)])  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Chunks do not align
    """
    if not any(blockdims):
        return ()
    non_trivial_dims = {d for d in blockdims if len(d) > 1}
    if len(non_trivial_dims) == 1:
        return first(non_trivial_dims)
    if len(non_trivial_dims) == 0:
        return max(blockdims, key=first)

    if np.isnan(sum(map(sum, blockdims))):
        raise ValueError(
            f"Arrays' chunk sizes ({blockdims}) are unknown.\n\nA possible solution:\n  x.compute_chunk_sizes()"
        )

    if len(set(map(sum, non_trivial_dims))) > 1:
        raise ValueError("Chunks do not add up to same value", blockdims)

    # We have multiple non-trivial chunks on this axis
    # e.g. (5, 2) and (4, 3)

    # We create a single chunk tuple with the same total length
    # that evenly divides both, e.g. (4, 1, 2)

    # To accomplish this we walk down all chunk tuples together, finding the
    # smallest element, adding it to the output, and subtracting it from all
    # other elements and remove the element itself.  We stop once we have
    # burned through all of the chunk tuples.
    # For efficiency's sake we reverse the lists so that we can pop off the end
    rchunks = [list(ntd)[::-1] for ntd in non_trivial_dims]
    total = sum(first(non_trivial_dims))
    i = 0

    out = []
    while i < total:
        m = min(c[-1] for c in rchunks)
        out.append(m)
        for c in rchunks:
            c[-1] -= m
            if c[-1] == 0:
                c.pop()
        i += m

    return tuple(out)


def is_scalar_for_elemwise(arg):
    """
    >>> is_scalar_for_elemwise(42)
    True
    >>> is_scalar_for_elemwise('foo')
    True
    >>> is_scalar_for_elemwise(True)
    True
    >>> is_scalar_for_elemwise(np.array(42))
    True
    >>> is_scalar_for_elemwise([1, 2, 3])
    True
    >>> is_scalar_for_elemwise(np.array([1, 2, 3]))
    False
    """
    # the second half of shape_condition is essentially just to ensure that
    # dask series / frame are treated as scalars in elemwise.
    maybe_shape = getattr(arg, "shape", None)
    shape_condition = not isinstance(maybe_shape, Iterable) or any(is_dask_collection(x) for x in maybe_shape)

    return (
        np.isscalar(arg)
        or shape_condition
        or isinstance(arg, np.dtype)
        or (isinstance(arg, np.ndarray) and arg.ndim == 0)
    )


def broadcast_shapes(*shapes):
    """
    Determines output shape from broadcasting arrays.

    Parameters
    ----------
    shapes : tuples
        The shapes of the arguments.

    Returns
    -------
    output_shape : tuple

    Raises
    ------
    ValueError
        If the input shapes cannot be successfully broadcast together.
    """
    if len(shapes) == 1:
        return shapes[0]
    out = []
    for sizes in zip_longest(*map(reversed, shapes), fillvalue=-1):
        has_nan = np.isnan(sizes).any()
        # Filter out -1 (missing dims), 0 and 1 (broadcastable), and nan
        non_trivial = [s for s in sizes if s not in (-1, 0, 1) and not np.isnan(s)]

        if has_nan:
            # If any nan, output is nan but we still validate non-nan values
            dim = np.nan
            # All non-trivial sizes must match each other
            if len(set(non_trivial)) > 1:
                raise ValueError(
                    "operands could not be broadcast together with shapes {}".format(" ".join(map(str, shapes)))
                )
        else:
            dim = 0 if 0 in sizes else np.max(sizes).item()
            if any(i not in [-1, 0, 1, dim] for i in sizes):
                raise ValueError(
                    "operands could not be broadcast together with shapes {}".format(" ".join(map(str, shapes)))
                )
        out.append(dim)
    return tuple(reversed(out))


def _elemwise_handle_where(*args, **kwargs):
    function = kwargs.pop("elemwise_where_function")
    *args, where, out = args
    if hasattr(out, "copy"):
        out = out.copy()
    return function(*args, where=where, out=out, **kwargs)


def handle_out(out, result):
    """Handle out parameters

    If out is a dask.array then this overwrites the contents of that array with
    the result
    """
    from dask_array._collection import Array

    if isinstance(out, tuple):
        if len(out) == 1:
            out = out[0]
        elif len(out) > 1:
            raise NotImplementedError("The out parameter is not fully supported")
        else:
            out = None
    if not (out is None or isinstance(out, Array)):
        raise NotImplementedError(
            f"The out parameter is not fully supported. Received type {type(out).__name__}, expected Dask Array"
        )
    if isinstance(out, Array):
        if out.shape != result.shape:
            raise ValueError(
                f"Mismatched shapes between result and out parameter. out={out.shape}, result={result.shape}"
            )
        # For expression-based arrays, we need to update the expression
        out._expr = result._expr
        return out
    else:
        return result


def _enforce_dtype(*args, **kwargs):
    """Calls a function and converts its result to the given dtype.

    The parameters have deliberately been given unwieldy names to avoid
    clashes with keyword arguments consumed by blockwise

    A dtype of `object` is treated as a special case and not enforced,
    because it is used as a dummy value in some places when the result will
    not be a block in an Array.

    Parameters
    ----------
    enforce_dtype : dtype
        Result dtype
    enforce_dtype_function : callable
        The wrapped function, which will be passed the remaining arguments
    """
    dtype = kwargs.pop("enforce_dtype")
    function = kwargs.pop("enforce_dtype_function")

    result = function(*args, **kwargs)
    if hasattr(result, "dtype") and dtype != result.dtype and dtype != object:
        if not np.can_cast(result, dtype, casting="same_kind"):
            raise ValueError(
                f"Inferred dtype from function {funcname(function)!r} was {str(dtype)!r} "
                f"but got {str(result.dtype)!r}, which can't be cast using "
                "casting='same_kind'"
            )
        if np.isscalar(result):
            # scalar astype method doesn't take the keyword arguments, so
            # have to convert via 0-dimensional array and back.
            result = result.astype(dtype)
        else:
            try:
                result = result.astype(dtype, copy=False)
            except TypeError:
                # Missing copy kwarg
                result = result.astype(dtype)
    return result


def unpack_singleton(x):
    """
    >>> unpack_singleton([[[[1]]]])
    1
    >>> unpack_singleton(np.array(np.datetime64('2000-01-01')))
    array('2000-01-01', dtype='datetime64[D]')
    """
    while isinstance(x, (list, tuple)):
        try:
            x = x[0]
        except (IndexError, TypeError, KeyError):
            break
    return x


def deepfirst(seq):
    """First element in a nested list

    >>> deepfirst([[[1, 2], [3, 4]], [5, 6], [7, 8]])
    1
    """
    if not isinstance(seq, (list, tuple)):
        return seq
    else:
        return deepfirst(seq[0])


def chunks_from_arrays(arrays):
    """Chunks tuple from nested list of arrays

    >>> x = np.array([1, 2])
    >>> chunks_from_arrays([x, x])
    ((2, 2),)

    >>> x = np.array([[1, 2]])
    >>> chunks_from_arrays([[x], [x]])
    ((1, 1), (2,))

    >>> x = np.array([[1, 2]])
    >>> chunks_from_arrays([[x, x]])
    ((1,), (2, 2))

    >>> chunks_from_arrays([1, 1])
    ((1, 1),)
    """
    if not arrays:
        return ()
    result = []
    dim = 0

    def shape(x):
        try:
            return x.shape if x.shape else (1,)
        except AttributeError:
            return (1,)

    while isinstance(arrays, (list, tuple)):
        result.append(tuple(shape(deepfirst(a))[dim] for a in arrays))
        arrays = arrays[0]
        dim += 1
    return tuple(result)


def concatenate3(arrays):
    """Recursive np.concatenate

    Input should be a nested list of numpy arrays arranged in the order they
    should appear in the array itself.  Each array should have the same number
    of dimensions as the desired output and the nesting of the lists.

    >>> x = np.array([[1, 2]])
    >>> concatenate3([[x, x, x], [x, x, x]])
    array([[1, 2, 1, 2, 1, 2],
           [1, 2, 1, 2, 1, 2]])

    >>> concatenate3([[x, x], [x, x], [x, x]])
    array([[1, 2, 1, 2],
           [1, 2, 1, 2],
           [1, 2, 1, 2]])
    """
    from dask import core as dask_core

    # We need this as __array_function__ may not exist on older NumPy versions.
    # And to reduce verbosity.
    NDARRAY_ARRAY_FUNCTION = getattr(np.ndarray, "__array_function__", None)

    arrays = concrete(arrays)
    if not arrays or all(el is None for el in flatten(arrays)):
        return np.empty(0)

    advanced = max(
        dask_core.flatten(arrays, container=(list, tuple)),
        key=lambda x: getattr(x, "__array_priority__", 0),
    )

    if not all(
        NDARRAY_ARRAY_FUNCTION is getattr(type(arr), "__array_function__", NDARRAY_ARRAY_FUNCTION)
        for arr in dask_core.flatten(arrays, container=(list, tuple))
    ):
        try:
            x = unpack_singleton(arrays)
            return _concatenate2(arrays, axes=tuple(range(x.ndim)))
        except TypeError:
            pass

    if concatenate_lookup.dispatch(type(advanced)) is not np.concatenate:
        x = unpack_singleton(arrays)
        return _concatenate2(arrays, axes=list(range(x.ndim)))

    ndim = ndimlist(arrays)
    if not ndim:
        return arrays
    chunks = chunks_from_arrays(arrays)
    shape = tuple(map(sum, chunks))

    def dtype(x):
        try:
            return x.dtype
        except AttributeError:
            return type(x)

    result = np.empty(shape=shape, dtype=dtype(deepfirst(arrays)))

    for idx, arr in zip(slices_from_chunks(chunks), dask_core.flatten(arrays, container=(list, tuple))):
        if hasattr(arr, "ndim"):
            while arr.ndim < ndim:
                arr = arr[None, ...]
        result[idx] = arr

    return result


# Register numpy concatenate as default
concatenate_lookup.register(np.ndarray, np.concatenate)
concatenate_lookup.register(object, np.concatenate)

# Register numpy tensordot as default
tensordot_lookup.register(np.ndarray, np.tensordot)
tensordot_lookup.register(object, np.tensordot)


# Vindex helper functions


def _get_axis(indexes):
    """Get axis along which point-wise slicing results lie

    This is mostly a hack because I can't figure out NumPy's rule on this and
    can't be bothered to go reading.

    >>> _get_axis([[1, 2], None, [1, 2], None])
    0
    >>> _get_axis([None, [1, 2], [1, 2], None])
    1
    >>> _get_axis([None, None, [1, 2], [1, 2]])
    2
    """
    ndim = len(indexes)
    indexes = [slice(None, None) if i is None else [0] for i in indexes]
    x = np.empty((2,) * ndim)
    x2 = x[tuple(indexes)]
    return x2.shape.index(1)


def _vindex_merge(locations, values):
    """

    >>> locations = [0], [2, 1]
    >>> values = [np.array([[1, 2, 3]]),
    ...           np.array([[10, 20, 30], [40, 50, 60]])]

    >>> _vindex_merge(locations, values)
    array([[ 1,  2,  3],
           [40, 50, 60],
           [10, 20, 30]])
    """
    locations = list(map(list, locations))
    values = list(values)

    n = sum(map(len, locations))

    shape = list(values[0].shape)
    shape[0] = n
    shape = tuple(shape)

    dtype = values[0].dtype

    x = np.empty_like(values[0], dtype=dtype, shape=shape)

    ind = [slice(None, None) for i in range(x.ndim)]
    for loc, val in zip(locations, values):
        ind[0] = loc
        x[tuple(ind)] = val

    return x


def _vindex_slice_and_transpose(block, points, axis):
    """Pull out point-wise slices from block and rotate block so that
    points are on the first dimension"""
    points = [p if isinstance(p, slice) else list(p) for p in points]
    block = block[tuple(points)]
    axes = [axis] + list(range(axis)) + list(range(axis + 1, block.ndim))
    return block.transpose(axes)


def interleave_none(a, b):
    """

    >>> interleave_none([0, None, 2, None], [1, 3])
    (0, 1, 2, 3)
    """
    result = []
    i = j = 0
    n = len(a) + len(b)
    while i + j < n:
        if a[i] is not None:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            i += 1
            j += 1
    return tuple(result)


def keyname(name, i, okey):
    """

    >>> keyname('x', 3, [None, None, 0, 2])
    ('x', 3, 0, 2)
    """
    return (name, i) + tuple(k for k in okey if k is not None)


# __array_function__ dict for mapping aliases and mismatching names
_HANDLED_FUNCTIONS = {}


def implements(*numpy_functions):
    """Register an __array_function__ implementation for dask.array.Array

    Register that a function implements the API of a NumPy function (or several
    NumPy functions in case of aliases) which is handled with
    ``__array_function__``.

    Parameters
    ----------
    \\*numpy_functions : callables
        One or more NumPy functions that are handled by ``__array_function__``
        and will be mapped by `implements` to a `dask.array` function.
    """

    def decorator(dask_func):
        for numpy_function in numpy_functions:
            _HANDLED_FUNCTIONS[numpy_function] = dask_func

        return dask_func

    return decorator


def _should_delegate(self, other) -> bool:
    """Check whether Dask should delegate to the other.
    This implementation follows NEP-13:
    https://numpy.org/neps/nep-0013-ufunc-overrides.html#behavior-in-combination-with-python-s-binary-operations
    """
    from dask_array._chunk_types import is_valid_array_chunk

    if hasattr(other, "__array_ufunc__") and other.__array_ufunc__ is None:
        return True
    elif (
        hasattr(other, "__array_ufunc__")
        and not is_valid_array_chunk(other)
        # don't delegate to our own parent classes
        and not isinstance(self, type(other))
        and type(self) is not type(other)
    ):
        return True
    elif (
        not hasattr(other, "__array_ufunc__")
        and hasattr(other, "__array_priority__")
        and other.__array_priority__ > self.__array_priority__
    ):
        return True
    return False


def check_if_handled_given_other(f):
    """Check if method is handled by Dask given type of other

    Ensures proper deferral to upcast types in dunder operations without
    assuming unknown types are automatically downcast types.
    """
    from functools import wraps

    @wraps(f)
    def wrapper(self, other):
        if _should_delegate(self, other):
            return NotImplemented
        else:
            return f(self, other)

    return wrapper


def finalize(results):
    """Finalize results from a dask array computation.

    Concatenates results if multiple chunks, otherwise returns a copy.
    """
    if not results:
        return concatenate3(results)
    results2 = results
    while isinstance(results2, (tuple, list)):
        if len(results2) > 1:
            return concatenate3(results)
        else:
            results2 = results2[0]

    results = unpack_singleton(results)
    # Single chunk. There is a risk that the result holds a buffer stored in the
    # graph or on a process-local Worker. Deep copy to make sure that nothing can
    # accidentally write back to it.
    try:
        return results.copy()  # numpy, sparse, scipy.sparse (any version)
    except AttributeError:
        # Not an Array API object
        return results


def _get_chunk_shape(a):
    """Get chunk shape as an array suitable for stacking."""
    s = np.asarray(a.shape, dtype=int)
    return s[len(s) * (None,) + (slice(None),)]
