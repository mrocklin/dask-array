"""Slicing utility functions copied from dask.array.slicing.

These are local copies to reduce dependency on dask.array.slicing.
"""

from __future__ import annotations

import bisect
import functools
import math
import warnings
from numbers import Integral, Number

import numpy as np
from tlz import memoize

from dask.base import is_dask_collection
from dask.utils import cached_cumsum, is_arraylike

colon = slice(None, None, None)


class SlicingNoop(Exception):
    """This indicates that a slicing operation is a no-op. The caller has to handle this"""

    pass


# ============================================================================
# Helper functions
# ============================================================================


def _array_like_safe(np_func, da_func, a, like, **kwargs):
    """Helper for asanyarray_safe."""
    from dask_array._collection import Array

    if like is a and hasattr(a, "__array_function__"):
        return a

    if isinstance(like, Array):
        return da_func(a, **kwargs)

    # Handle dask arrays backed by cupy
    if isinstance(a, Array):
        try:
            from dask.utils import is_cupy_type

            if is_cupy_type(a._meta):
                a = a.compute(scheduler="sync")
        except ImportError:
            pass

    if hasattr(like, "__array_function__"):
        return np_func(a, like=like, **kwargs)

    if type(like).__module__.startswith("scipy.sparse"):
        kwargs.pop("order", None)
        if np.isscalar(a):
            a = np.array([[a]])
        return type(like)(a, **kwargs)

    return np_func(a, **kwargs)


def asanyarray_safe(a, like, **kwargs):
    """Convert to array using np.asanyarray with proper dispatching.

    If a is dask.array, return dask.array.asanyarray(a, **kwargs),
    otherwise return np.asanyarray(a, like=like, **kwargs), dispatching
    the call to the library that implements the like array.
    """
    from dask_array.core._conversion import asanyarray

    return _array_like_safe(np.asanyarray, asanyarray, a, like, **kwargs)


# ============================================================================
# sanitize_index and helpers
# ============================================================================


def _sanitize_index_element(ind):
    """Sanitize a one-element index."""
    if isinstance(ind, Number):
        ind2 = int(ind)
        if ind2 != ind:
            raise IndexError(f"Bad index.  Must be integer-like: {ind}")
        else:
            return ind2
    elif ind is None:
        return None
    elif is_dask_collection(ind):
        if ind.dtype.kind != "i" or ind.size != 1:
            raise IndexError(f"Bad index. Must be integer-like: {ind}")
        return ind
    else:
        raise TypeError("Invalid index type", type(ind), ind)


def sanitize_index(ind):
    """Sanitize the elements for indexing along one axis.

    Examples
    --------
    >>> sanitize_index([2, 3, 5])
    array([2, 3, 5])
    >>> sanitize_index([True, False, True, False])
    array([0, 2])
    >>> sanitize_index(np.array([1, 2, 3]))
    array([1, 2, 3])
    >>> sanitize_index(np.array([False, True, True]))
    array([1, 2])
    >>> type(sanitize_index(np.int32(0)))
    <class 'int'>
    >>> sanitize_index(1.0)
    1
    >>> sanitize_index(0.5)
    Traceback (most recent call last):
    ...
    IndexError: Bad index.  Must be integer-like: 0.5
    """
    if ind is None:
        return None
    elif isinstance(ind, slice):
        return slice(
            _sanitize_index_element(ind.start),
            _sanitize_index_element(ind.stop),
            _sanitize_index_element(ind.step),
        )
    elif isinstance(ind, Number):
        return _sanitize_index_element(ind)
    elif is_dask_collection(ind):
        return ind
    index_array = asanyarray_safe(ind, like=ind)
    if index_array.dtype == bool:
        nonzero = np.nonzero(index_array)
        if len(nonzero) == 1:
            nonzero = nonzero[0]
        if is_arraylike(nonzero):
            return nonzero
        else:
            return np.asanyarray(nonzero)
    elif np.issubdtype(index_array.dtype, np.integer):
        return index_array
    elif np.issubdtype(index_array.dtype, np.floating):
        int_index = index_array.astype(np.intp)
        if np.allclose(index_array, int_index):
            return int_index
        else:
            check_int = np.isclose(index_array, int_index)
            first_err = index_array.ravel()[np.flatnonzero(~check_int)[0]]
            raise IndexError(f"Bad index.  Must be integer-like: {first_err}")
    else:
        raise TypeError("Invalid index type", type(ind), ind)


# ============================================================================
# replace_ellipsis
# ============================================================================


def replace_ellipsis(n, index):
    """Replace ... with slices, :, :, :

    Examples
    --------
    >>> replace_ellipsis(4, (3, Ellipsis, 2))
    (3, slice(None, None, None), slice(None, None, None), 2)

    >>> replace_ellipsis(2, (Ellipsis, None))
    (slice(None, None, None), slice(None, None, None), None)
    """
    # Careful about using in or index because index may contain arrays
    isellipsis = [i for i, ind in enumerate(index) if ind is Ellipsis]
    if not isellipsis:
        return index
    else:
        loc = isellipsis[0]
    extra_dimensions = n - (len(index) - sum(i is None for i in index) - 1)
    return (
        index[:loc] + (slice(None, None, None),) * extra_dimensions + index[loc + 1 :]
    )


# ============================================================================
# normalize_slice and helpers
# ============================================================================


def normalize_slice(idx, dim):
    """Normalize slices to canonical form.

    Parameters
    ----------
    idx: slice or other index
    dim: dimension length

    Examples
    --------
    >>> normalize_slice(slice(0, 10, 1), 10)
    slice(None, None, None)
    """
    if isinstance(idx, slice):
        if math.isnan(dim):
            return idx
        start, stop, step = idx.indices(dim)
        if step > 0:
            if start == 0:
                start = None
            if stop >= dim:
                stop = None
            if step == 1:
                step = None
            if stop is not None and start is not None and stop < start:
                stop = start
        elif step < 0:
            if start >= dim - 1:
                start = None
            if stop < 0:
                stop = None
        return slice(start, stop, step)
    return idx


# ============================================================================
# posify_index
# ============================================================================


def posify_index(shape, ind):
    """Flip negative indices around to positive ones.

    Examples
    --------
    >>> posify_index(10, 3)
    3
    >>> posify_index(10, -3)
    7
    >>> posify_index(10, [3, -3])
    array([3, 7])

    >>> posify_index((10, 20), (3, -3))
    (3, 17)
    >>> posify_index((10, 20), (3, [3, 4, -3]))  # doctest: +NORMALIZE_WHITESPACE
    (3, array([ 3,  4, 17]))
    """
    if isinstance(ind, tuple):
        return tuple(map(posify_index, shape, ind))
    if isinstance(ind, Integral):
        if ind < 0 and not math.isnan(shape):
            return ind + shape
        else:
            return ind
    if isinstance(ind, (np.ndarray, list)) and not math.isnan(shape):
        ind = np.asanyarray(ind)
        return np.where(ind < 0, ind + shape, ind)
    return ind


# ============================================================================
# check_index
# ============================================================================


def check_index(axis, ind, dimension):
    """Check validity of index for a given dimension.

    Examples
    --------
    >>> check_index(0, 3, 5)
    >>> check_index(0, 5, 5)
    Traceback (most recent call last):
    ...
    IndexError: Index 5 is out of bounds for axis 0 with size 5

    >>> check_index(1, 6, 5)
    Traceback (most recent call last):
    ...
    IndexError: Index 6 is out of bounds for axis 1 with size 5

    >>> check_index(1, -1, 5)
    >>> check_index(1, -6, 5)
    Traceback (most recent call last):
    ...
    IndexError: Index -6 is out of bounds for axis 1 with size 5

    >>> check_index(0, [1, 2], 5)
    >>> check_index(0, [6, 3], 5)
    Traceback (most recent call last):
    ...
    IndexError: Index is out of bounds for axis 0 with size 5

    >>> check_index(1, slice(0, 3), 5)

    >>> check_index(0, [True], 1)
    >>> check_index(0, [True, True], 3)
    Traceback (most recent call last):
    ...
    IndexError: Boolean array with size 2 is not long enough for axis 0 with size 3
    >>> check_index(0, [True, True, True], 1)
    Traceback (most recent call last):
    ...
    IndexError: Boolean array with size 3 is not long enough for axis 0 with size 1
    """
    if isinstance(ind, list):
        ind = np.asanyarray(ind)

    # unknown dimension, assumed to be in bounds
    if np.isnan(dimension):
        return
    elif is_dask_collection(ind):
        return
    elif is_arraylike(ind):
        if ind.dtype == bool:
            if ind.size != dimension:
                raise IndexError(
                    f"Boolean array with size {ind.size} is not long enough "
                    f"for axis {axis} with size {dimension}"
                )
        elif (ind >= dimension).any() or (ind < -dimension).any():
            raise IndexError(
                f"Index is out of bounds for axis {axis} with size {dimension}"
            )
    elif isinstance(ind, slice):
        return
    elif ind is None:
        return

    elif ind >= dimension or ind < -dimension:
        raise IndexError(
            f"Index {ind} is out of bounds for axis {axis} with size {dimension}"
        )


# ============================================================================
# _slice_1d
# ============================================================================


def _slice_1d(dim_shape, lengths, index):
    """Returns a dict of {blocknum: slice}.

    This function figures out where each slice should start in each
    block for a single dimension. If the slice won't return any elements
    in the block, that block will not be in the output.

    Parameters
    ----------
    dim_shape - the number of elements in this dimension.
      This should be a positive, non-zero integer
    lengths - the number of elements per block in this dimension
      This should be a positive, non-zero integer
    index - a description of the elements in this dimension that we want
      This might be an integer, a slice(), or an Ellipsis

    Returns
    -------
    dictionary where the keys are the integer index of the blocks that
      should be sliced and the values are the slices

    Examples
    --------
    Trivial slicing

    >>> _slice_1d(100, [60, 40], slice(None, None, None))
    {0: slice(None, None, None), 1: slice(None, None, None)}

    100 length array cut into length 20 pieces, slice 0:35

    >>> _slice_1d(100, [20, 20, 20, 20, 20], slice(0, 35))
    {0: slice(None, None, None), 1: slice(0, 15, 1)}

    Support irregular blocks and various slices

    >>> _slice_1d(100, [20, 10, 10, 10, 25, 25], slice(10, 35))
    {0: slice(10, 20, 1), 1: slice(None, None, None), 2: slice(0, 5, 1)}

    Support step sizes

    >>> _slice_1d(100, [15, 14, 13], slice(10, 41, 3))
    {0: slice(10, 15, 3), 1: slice(1, 14, 3), 2: slice(2, 12, 3)}

    >>> _slice_1d(100, [20, 20, 20, 20, 20], slice(0, 100, 40))  # step > blocksize
    {0: slice(0, 20, 40), 2: slice(0, 20, 40), 4: slice(0, 20, 40)}

    Also support indexing single elements

    >>> _slice_1d(100, [20, 20, 20, 20, 20], 25)
    {1: 5}

    And negative slicing

    >>> _slice_1d(100, [20, 20, 20, 20, 20], slice(100, 0, -3)) # doctest: +NORMALIZE_WHITESPACE
    {4: slice(-1, -21, -3),
     3: slice(-2, -21, -3),
     2: slice(-3, -21, -3),
     1: slice(-1, -21, -3),
     0: slice(-2, -20, -3)}

    >>> _slice_1d(100, [20, 20, 20, 20, 20], slice(100, 12, -3)) # doctest: +NORMALIZE_WHITESPACE
    {4: slice(-1, -21, -3),
     3: slice(-2, -21, -3),
     2: slice(-3, -21, -3),
     1: slice(-1, -21, -3),
     0: slice(-2, -8, -3)}

    >>> _slice_1d(100, [20, 20, 20, 20, 20], slice(100, -12, -3))
    {4: slice(-1, -12, -3)}
    """
    chunk_boundaries = cached_cumsum(lengths)

    if isinstance(index, Integral):
        # use right-side search to be consistent with previous result
        i = bisect.bisect_right(chunk_boundaries, index)
        if i > 0:
            # the very first chunk has no relative shift
            ind = index - chunk_boundaries[i - 1]
        else:
            ind = index
        return {int(i): int(ind)}

    assert isinstance(index, slice)

    if index == colon:
        return dict.fromkeys(range(len(lengths)), colon)

    step = index.step or 1
    if step > 0:
        start = index.start or 0
        stop = index.stop if index.stop is not None else dim_shape
    else:
        start = index.start if index.start is not None else dim_shape - 1
        start = dim_shape - 1 if start >= dim_shape else start
        stop = -(dim_shape + 1) if index.stop is None else index.stop

    # posify start and stop
    if start < 0:
        start += dim_shape
    if stop < 0:
        stop += dim_shape

    d = dict()
    if step > 0:
        istart = bisect.bisect_right(chunk_boundaries, start)
        istop = bisect.bisect_left(chunk_boundaries, stop)

        # the bound is not exactly tight; make it tighter?
        istop = min(istop + 1, len(lengths))

        # jump directly to istart
        if istart > 0:
            start = start - chunk_boundaries[istart - 1]
            stop = stop - chunk_boundaries[istart - 1]

        for i in range(istart, istop):
            length = lengths[i]
            if start < length and stop > 0:
                d[i] = slice(start, min(stop, length), step)
                start = (start - length) % step
            else:
                start = start - length
            stop -= length
    else:
        rstart = start  # running start

        istart = bisect.bisect_left(chunk_boundaries, start)
        istop = bisect.bisect_right(chunk_boundaries, stop)

        # the bound is not exactly tight; make it tighter?
        istart = min(istart + 1, len(chunk_boundaries) - 1)
        istop = max(istop - 1, -1)

        for i in range(istart, istop, -1):
            chunk_stop = chunk_boundaries[i]
            # create a chunk start and stop
            if i == 0:
                chunk_start = 0
            else:
                chunk_start = chunk_boundaries[i - 1]

            # if our slice is in this chunk
            if (chunk_start <= rstart < chunk_stop) and (rstart > stop):
                d[i] = slice(
                    rstart - chunk_stop,
                    max(chunk_start - chunk_stop - 1, stop - chunk_stop),
                    step,
                )

                # compute the next running start point,
                offset = (rstart - (chunk_start - 1)) % step
                rstart = chunk_start + offset - 1

    # replace 0:20:1 with : if appropriate
    for k, v in d.items():
        if v == slice(0, lengths[k], 1):
            d[k] = slice(None, None, None)

    if not d:  # special case x[:0]
        d[0] = slice(0, 0, 1)

    return d


# ============================================================================
# new_blockdim
# ============================================================================


def new_blockdim(dim_shape, lengths, index):
    """Compute new block dimension sizes after slicing.

    Examples
    --------
    >>> new_blockdim(100, [20, 10, 20, 10, 40], slice(0, 90, 2))
    [10, 5, 10, 5, 15]

    >>> new_blockdim(100, [20, 10, 20, 10, 40], [5, 1, 30, 22])
    [4]

    >>> new_blockdim(100, [20, 10, 20, 10, 40], slice(90, 10, -2))
    [16, 5, 10, 5, 4]
    """
    from operator import itemgetter

    if index == slice(None, None, None):
        return lengths
    if isinstance(index, list):
        return [len(index)]
    assert not isinstance(index, Integral)
    pairs = sorted(_slice_1d(dim_shape, lengths, index).items(), key=itemgetter(0))
    slices = [
        slice(0, lengths[i], 1) if slc == slice(None, None, None) else slc
        for i, slc in pairs
    ]
    if isinstance(index, slice) and index.step and index.step < 0:
        slices.reverse()
    return [int(math.ceil((1.0 * slc.stop - slc.start) / slc.step)) for slc in slices]


# ============================================================================
# normalize_index (uses many helpers above)
# ============================================================================


def normalize_index(idx, shape):
    """Normalize slicing indexes.

    1.  Replaces ellipses with many full slices
    2.  Adds full slices to end of index
    3.  Checks bounding conditions
    4.  Replace multidimensional numpy arrays with dask arrays
    5.  Replaces numpy arrays with lists
    6.  Posify's integers and lists
    7.  Normalizes slices to canonical form

    Examples
    --------
    >>> normalize_index(1, (10,))
    (1,)
    >>> normalize_index(-1, (10,))
    (9,)
    >>> normalize_index([-1], (10,))
    (array([9]),)
    >>> normalize_index(slice(-3, 10, 1), (10,))
    (slice(7, None, None),)
    >>> normalize_index((Ellipsis, None), (10,))
    (slice(None, None, None), None)
    """
    from dask_array._collection import Array
    from dask_array.core import from_array

    if not isinstance(idx, tuple):
        idx = (idx,)

    # if a > 1D numpy.array is provided, cast it to a dask array
    if len(idx) > 0 and len(shape) > 1:
        i = idx[0]
        if is_arraylike(i) and not isinstance(i, Array) and i.shape == shape:
            idx = (from_array(i), *idx[1:])

    idx = replace_ellipsis(len(shape), idx)
    n_sliced_dims = 0
    for i in idx:
        if hasattr(i, "ndim") and i.ndim >= 1:
            n_sliced_dims += i.ndim
        elif i is None:
            continue
        else:
            n_sliced_dims += 1

    idx = idx + (slice(None),) * (len(shape) - n_sliced_dims)
    if len([i for i in idx if i is not None]) > len(shape):
        raise IndexError("Too many indices for array")

    none_shape = []
    i = 0
    for ind in idx:
        if ind is not None:
            none_shape.append(shape[i])
            i += 1
        else:
            none_shape.append(None)

    for axis, (i, d) in enumerate(zip(idx, none_shape)):
        if d is not None:
            check_index(axis, i, d)
    idx = tuple(map(sanitize_index, idx))
    idx = tuple(map(normalize_slice, idx, none_shape))
    idx = posify_index(none_shape, idx)
    return idx


# ============================================================================
# parse_assignment_indices
# ============================================================================


def parse_assignment_indices(indices, shape):
    """Reformat the indices for assignment.

    The aim of this is to convert the indices to a standardised form
    so that it is easier to ascertain which chunks are touched by the
    indices.

    This function is intended to be called by `setitem_array`.

    A slice object that is decreasing (i.e. with a negative step), is
    recast as an increasing slice (i.e. with a positive step. For
    example ``slice(7,3,-1)`` would be cast as ``slice(4,8,1)``. This
    is to facilitate finding which blocks are touched by the
    index. The dimensions for which this has occurred are returned by
    the function.

    Parameters
    ----------
    indices : numpy-style indices
        Indices to array defining the elements to be assigned.
    shape : sequence of `int`
        The shape of the array.

    Returns
    -------
    parsed_indices : `list`
        The reformatted indices that are equivalent to the input
        indices.
    implied_shape : `list`
        The shape implied by the parsed indices. For instance, indices
        of ``(slice(0,2), 5, [4,1,-1])`` will have implied shape
        ``[2,3]``.
    reverse : `list`
        The positions of the dimensions whose indices in the
        parsed_indices output are reversed slices.
    implied_shape_positions: `list`
        The positions of the dimensions whose indices contribute to
        the implied_shape. For instance, indices of ``(slice(0,2), 5,
        [4,1,-1])`` will have implied_shape ``[2,3]`` and
        implied_shape_positions ``[0,2]``.

    Examples
    --------
    >>> parse_assignment_indices((slice(1, -1),), (8,))
    ([slice(1, 7, 1)], [6], [], [0])

    >>> parse_assignment_indices(([1, 2, 6, 5],), (8,))
    ([array([1, 2, 6, 5])], [4], [], [0])

    >>> parse_assignment_indices((3, slice(-1, 2, -1)), (7, 8))
    ([3, slice(3, 8, 1)], [5], [1], [1])

    >>> parse_assignment_indices((slice(-1, 2, -1), 3, [1, 2]), (7, 8, 9))
    ([slice(3, 7, 1), 3, array([1, 2])], [4, 2], [0], [0, 2])

    >>> parse_assignment_indices((slice(0, 5), slice(3, None, 2)), (5, 4))
    ([slice(0, 5, 1), slice(3, 4, 2)], [5, 1], [], [0, 1])

    >>> parse_assignment_indices((slice(0, 5), slice(3, 3, 2)), (5, 4))
    ([slice(0, 5, 1), slice(3, 3, 2)], [5, 0], [], [0])

    """
    if not isinstance(indices, tuple):
        indices = (indices,)

    # Disallow scalar boolean indexing, and also indexing by scalar
    # numpy or dask array.
    for index in indices:
        if index is True or index is False:
            raise NotImplementedError(
                "dask does not yet implement assignment to a scalar "
                f"boolean index: {index!r}"
            )

        if (is_arraylike(index) or is_dask_collection(index)) and not index.ndim:
            raise NotImplementedError(
                "dask does not yet implement assignment to a scalar "
                f"numpy or dask array index: {index!r}"
            )

    # Initialize output variables
    implied_shape = []
    implied_shape_positions = []
    reverse = []
    parsed_indices = list(normalize_index(indices, shape))

    n_lists = 0

    for i, (index, size) in enumerate(zip(parsed_indices, shape)):
        is_slice = isinstance(index, slice)
        if is_slice:
            # Index is a slice
            start, stop, step = index.indices(size)
            if step < 0 and stop == -1:
                stop = None

            index = slice(start, stop, step)

            if step < 0:
                # When the slice step is negative, transform the
                # original slice to a new slice with a positive step
                # such that the result of the new slice is the reverse
                # of the result of the original slice.
                start, stop, step = index.indices(size)
                step *= -1
                div, mod = divmod(start - stop - 1, step)
                div_step = div * step
                start -= div_step
                stop = start + div_step + 1

                index = slice(start, stop, step)
                reverse.append(i)

            start, stop, step = index.indices(size)

            # Note: We now have stop >= start and step >= 0

            div, mod = divmod(stop - start, step)
            if not div and not mod:
                # stop equals start => zero-sized slice for this
                # dimension
                implied_shape.append(0)
            else:
                if mod != 0:
                    div += 1

                implied_shape.append(div)
                implied_shape_positions.append(i)

        elif isinstance(index, (int, np.integer)):
            # Index is an integer
            index = int(index)

        elif is_arraylike(index) or is_dask_collection(index):
            # Index is 1-d array
            n_lists += 1
            if n_lists > 1:
                raise NotImplementedError(
                    "dask is currently limited to at most one "
                    "dimension's assignment index being a "
                    "1-d array of integers or booleans. "
                    f"Got: {indices}"
                )

            if index.ndim != 1:
                raise IndexError(
                    f"Incorrect shape ({index.shape}) of integer "
                    f"indices for dimension with size {size}"
                )

            index_size = index.size
            if (
                index.dtype == bool
                and not math.isnan(index_size)
                and index_size != size
            ):
                raise IndexError(
                    "boolean index did not match indexed array along "
                    f"dimension {i}; dimension is {size} but "
                    f"corresponding boolean dimension is {index_size}"
                )

            # Posify an integer dask array (integer numpy arrays were
            # posified in `normalize_index`)
            if is_dask_collection(index):
                if index.dtype == bool:
                    index_size = np.nan
                else:
                    index = np.where(index < 0, index + size, index)

            implied_shape.append(index_size)
            implied_shape_positions.append(i)

        parsed_indices[i] = index

    return parsed_indices, implied_shape, reverse, implied_shape_positions


# ============================================================================
# setitem (chunk function)
# ============================================================================


def setitem(x, v, indices):
    """Chunk function for array assignment.

    Assign v to indices of x.

    Parameters
    ----------
    x : numpy/cupy/etc. array
        The array to be assigned to.
    v : numpy/cupy/etc. array
        The values which will be assigned.
    indices : list of `slice`, `int`, or numpy array
        The indices describing the elements of x to be assigned from
        v. One index per axis.

        Note that an individual index can not be a `list`, use a 1-d
        numpy array instead.

        If a 1-d numpy array index contains the non-valid value of the
        size of the corresponding dimension of x, then those index
        elements will be removed prior to the assignment (see
        `block_index_from_1d_index` function).

    Returns
    -------
    numpy/cupy/etc. array
        A new independent array with assigned elements, unless v is
        empty (i.e. has zero size) in which case then the input array
        is returned and the indices are ignored.

    Examples
    --------
    >>> x = np.arange(8).reshape(2, 4)
    >>> setitem(x, np.array(-99), [np.array([False, True])])
    array([[  0,   1,   2,   3],
           [-99, -99, -99, -99]])
    >>> x
    array([[0, 1, 2, 3],
           [4, 5, 6, 7]])
    >>> setitem(x, np.array([-88, -99]), [slice(None), np.array([1, 3])])
    array([[  0, -88,   2, -99],
           [  4, -88,   6, -99]])
    >>> setitem(x, -x, [slice(None)])
    array([[ 0, -1, -2, -3],
           [-4, -5, -6, -7]])
    >>> x
    array([[0, 1, 2, 3],
           [4, 5, 6, 7]])
    >>> setitem(x, np.array([-88, -99]), [slice(None), np.array([4, 4, 3, 4, 1, 4])])
    array([[  0, -99,   2, -88],
           [  4, -99,   6, -88]])
    >>> value = np.where(x < 0)[0]
    >>> value.size
    0
    >>> y = setitem(x, value, [Ellipsis])
    >>> y is x
    True
    """
    if not math.prod(v.shape):
        return x

    # Normalize integer array indices
    for i, (index, block_size) in enumerate(zip(indices, x.shape)):
        if isinstance(index, np.ndarray) and index.dtype != bool:
            # Strip out any non-valid place-holder values
            index = index[np.where(index < block_size)[0]]
            indices[i] = index

    # If x is not masked but v is, then turn the x into a masked
    # array.
    if not np.ma.isMA(x) and np.ma.isMA(v):
        x = x.view(np.ma.MaskedArray)

    # Copy the array to guarantee no other objects are corrupted.
    # When x is the output of a scalar __getitem__ call, it is a
    # np.generic, which is read-only. Convert it to a (writeable)
    # 0-d array. x could also be a cupy array etc.
    x = np.asarray(x) if isinstance(x, np.generic) else x.copy()

    # Do the assignment
    try:
        x[tuple(indices)] = v
    except ValueError as e:
        raise ValueError(
            "shape mismatch: value array could not be broadcast to indexing result"
        ) from e

    return x


# ============================================================================
# shuffle_slice and helpers
# ============================================================================


def make_block_sorted_slices(index, chunks):
    """Generate blockwise-sorted index pairs for shuffling an array.

    Parameters
    ----------
    index : ndarray
        An array of index positions.
    chunks : tuple
        Chunks from the original dask array

    Returns
    -------
    index2 : ndarray
        Same values as `index`, but each block has been sorted
    index3 : ndarray
        The location of the values of `index` in `index2`

    Examples
    --------
    >>> index = np.array([6, 0, 4, 2, 7, 1, 5, 3])
    >>> chunks = ((4, 4),)
    >>> a, b = make_block_sorted_slices(index, chunks)

    Notice that the first set of 4 items are sorted, and the
    second set of 4 items are sorted.

    >>> a
    array([0, 2, 4, 6, 1, 3, 5, 7])
    >>> b
    array([3, 0, 2, 1, 7, 4, 6, 5])
    """
    from dask_array._core_utils import slices_from_chunks

    slices = slices_from_chunks(chunks)

    if len(slices[0]) > 1:
        slices = [slice_[0] for slice_ in slices]

    offsets = np.roll(np.cumsum(chunks[0]), 1)
    offsets[0] = 0

    index2 = np.empty_like(index)
    index3 = np.empty_like(index)

    for slice_, offset in zip(slices, offsets):
        a = index[slice_]
        b = np.sort(a)
        c = offset + np.argsort(b.take(np.argsort(a)))
        index2[slice_] = b
        index3[slice_] = c

    return index2, index3


def shuffle_slice(x, index):
    """A relatively efficient way to shuffle `x` according to `index`.

    Parameters
    ----------
    x : Array
    index : ndarray
        This should be an ndarray the same length as `x` containing
        each index position in ``range(0, len(x))``.

    Returns
    -------
    Array
    """
    chunks1 = chunks2 = x.chunks
    if x.ndim > 1:
        chunks1 = (chunks1[0],)
    index2, index3 = make_block_sorted_slices(index, chunks1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        return x[index2].rechunk(chunks2)[index3]


# ============================================================================
# expander (used by some slice operations)
# ============================================================================


@memoize
def _expander(where):
    if not where:

        def expand(seq, val):
            return seq

        return expand
    else:
        decl = """def expand(seq, val):
            return ({left}) + tuple({right})
        """
        left = []
        j = 0
        for i in range(max(where) + 1):
            if i in where:
                left.append("val, ")
            else:
                left.append(f"seq[{j}], ")
                j += 1
        right = f"seq[{j}:]"
        left = "".join(left)
        decl = decl.format(**locals())
        ns = {}
        exec(compile(decl, "<dynamic>", "exec"), ns, ns)
        return ns["expand"]


def expander(where):
    """Create a function to insert value at many locations in sequence.

    Examples
    --------
    >>> expander([0, 2])(['a', 'b', 'c'], 'z')
    ('z', 'a', 'z', 'b', 'c')
    """
    return _expander(tuple(where))


# ============================================================================
# fuse_slice and helpers
# ============================================================================


def _normalize_slice_for_fusion(s):
    """Replace Nones in slices with integers for fusion.

    Unlike normalize_slice which takes a dimension parameter, this version
    is used for slice fusion where we don't need the dimension.

    >>> _normalize_slice_for_fusion(slice(None, None, None))
    slice(0, None, 1)
    """
    start, stop, step = s.start, s.stop, s.step
    if start is None:
        start = 0
    if step is None:
        step = 1
    if start < 0 or step < 0 or stop is not None and stop < 0:
        raise NotImplementedError()
    return slice(start, stop, step)


def _check_for_nonfusible_fancy_indexing(fancy, normal):
    """Check for fancy indexing and normal indexing conflicts.

    Disallow things like:
    x[:, [1, 2], :][0, :, :] -> x[0, [1, 2], :] or
    x[0, :, :][:, [1, 2], :] -> x[0, [1, 2], :]
    """
    from itertools import zip_longest

    for f, n in zip_longest(fancy, normal, fillvalue=slice(None)):
        if type(f) is not list and isinstance(n, Integral):
            raise NotImplementedError(
                "Can't handle normal indexing with "
                "integers and fancy indexing if the "
                "integers and fancy indices don't "
                "align with the same dimensions."
            )


def fuse_slice(a, b):
    """Fuse stacked slices together.

    Fuse a pair of repeated slices into a single slice:

    >>> fuse_slice(slice(1000, 2000), slice(10, 15))
    slice(1010, 1015, None)

    This also works for tuples of slices

    >>> fuse_slice((slice(100, 200), slice(100, 200, 10)),
    ...            (slice(10, 15), [5, 2]))
    (slice(110, 115, None), [150, 120])

    And a variety of other interesting cases

    >>> fuse_slice(slice(1000, 2000), 10)  # integers
    1010

    >>> fuse_slice(slice(1000, 2000, 5), slice(10, 20, 2))
    slice(1050, 1100, 10)

    >>> fuse_slice(slice(1000, 2000, 5), [1, 2, 3])  # lists
    [1005, 1010, 1015]

    >>> fuse_slice(None, slice(None, None))  # doctest: +SKIP
    None
    """
    # None only works if the second side is a full slice
    if a is None and isinstance(b, slice) and b == slice(None, None):
        return None

    # Replace None with 0 and one in start and step
    if isinstance(a, slice):
        a = _normalize_slice_for_fusion(a)
    if isinstance(b, slice):
        b = _normalize_slice_for_fusion(b)

    if isinstance(a, slice) and isinstance(b, Integral):
        if b < 0:
            raise NotImplementedError()
        return a.start + b * a.step

    if isinstance(a, slice) and isinstance(b, slice):
        start = a.start + a.step * b.start
        if b.stop is not None:
            stop = a.start + a.step * b.stop
        else:
            stop = None
        if a.stop is not None:
            if stop is not None:
                stop = min(a.stop, stop)
            else:
                stop = a.stop
        step = a.step * b.step
        if step == 1:
            step = None
        return slice(start, stop, step)

    if isinstance(b, list):
        return [fuse_slice(a, bb) for bb in b]
    if isinstance(a, list) and isinstance(b, (Integral, slice)):
        return a[b]

    if isinstance(a, tuple) and not isinstance(b, tuple):
        b = (b,)

    # If given two tuples walk through both, being mindful of uneven sizes
    # and newaxes
    if isinstance(a, tuple) and isinstance(b, tuple):
        # Check for non-fusible cases with fancy-indexing
        a_has_lists = any(isinstance(item, list) for item in a)
        b_has_lists = any(isinstance(item, list) for item in b)
        if a_has_lists and b_has_lists:
            raise NotImplementedError("Can't handle multiple list indexing")
        elif a_has_lists:
            _check_for_nonfusible_fancy_indexing(a, b)
        elif b_has_lists:
            _check_for_nonfusible_fancy_indexing(b, a)

        j = 0
        result = list()
        for i in range(len(a)):
            #  axis ceased to exist  or we're out of b
            if isinstance(a[i], Integral) or j == len(b):
                result.append(a[i])
                continue
            while b[j] is None:  # insert any Nones on the rhs
                result.append(None)
                j += 1
            result.append(fuse_slice(a[i], b[j]))  # Common case
            j += 1
        while j < len(b):  # anything leftover on the right?
            result.append(b[j])
            j += 1
        return tuple(result)
    raise NotImplementedError()
