from __future__ import annotations

import functools
import math
from functools import reduce
from itertools import product
from operator import mul

import numpy as np

from dask._task_spec import Task, TaskRef
from dask_array._expr import ArrayExpr
from dask_array.slicing._utils import sanitize_index
from dask_array._utils import meta_from_array
from dask.utils import M


# --------------------------------------------------------------------------
# reshape_rechunk and helper functions (copied from dask.array.reshape)
# --------------------------------------------------------------------------

_not_implemented_message = """
Dask's reshape only supports operations that merge or split existing dimensions
evenly. For example:

>>> x = da.ones((6, 5, 4), chunks=(3, 2, 2))
>>> x.reshape((3, 2, 5, 4))  # supported, splits 6 into 3 & 2
>>> x.reshape((30, 4))       # supported, merges 6 & 5 into 30
>>> x.reshape((4, 5, 6))     # unsupported, existing dimensions split unevenly

To work around this you may call reshape in multiple passes, or (if your data
is small enough) call ``compute`` first and handle reshaping in ``numpy``
directly.
"""


def reshape_rechunk(inshape, outshape, inchunks, disallow_dimension_expansion=False):
    assert all(isinstance(c, tuple) for c in inchunks)
    ii = len(inshape) - 1
    oi = len(outshape) - 1
    result_inchunks = [None for i in range(len(inshape))]
    result_outchunks = [None for i in range(len(outshape))]
    mapper_in, one_dimensions = {}, []

    while ii >= 0 or oi >= 0:
        if inshape[ii] == outshape[oi]:
            result_inchunks[ii] = inchunks[ii]
            result_outchunks[oi] = inchunks[ii]
            mapper_in[ii] = oi
            ii -= 1
            oi -= 1
            continue
        din = inshape[ii]
        dout = outshape[oi]
        if din == 1:
            result_inchunks[ii] = (1,)
            ii -= 1
        elif dout == 1:
            result_outchunks[oi] = (1,)
            one_dimensions.append(oi)
            oi -= 1
        elif din < dout:  # (4, 4, 4) -> (64,)
            ileft = ii - 1
            mapper_in[ii] = oi
            while (
                ileft >= 0 and reduce(mul, inshape[ileft : ii + 1]) < dout
            ):  # 4 < 64, 4*4 < 64, 4*4*4 == 64
                mapper_in[ileft] = oi
                ileft -= 1

            mapper_in[ileft] = oi
            if reduce(mul, inshape[ileft : ii + 1]) != dout:
                raise NotImplementedError(_not_implemented_message)
            # Special case to avoid intermediate rechunking:
            # When all the lower axis are completely chunked (chunksize=1) then
            # we're simply moving around blocks.
            if all(len(inchunks[i]) == inshape[i] for i in range(ii)):
                for i in range(ii + 1):
                    result_inchunks[i] = inchunks[i]
                result_outchunks[oi] = inchunks[ii] * math.prod(
                    map(len, inchunks[ileft:ii])
                )
            else:
                for i in range(ileft + 1, ii + 1):  # need single-shape dimensions
                    result_inchunks[i] = (inshape[i],)  # chunks[i] = (4,)

                chunk_reduction = reduce(mul, map(len, inchunks[ileft + 1 : ii + 1]))
                result_inchunks[ileft] = expand_tuple(inchunks[ileft], chunk_reduction)

                max_in_chunk = _cal_max_chunk_size(inchunks, ileft, ii)
                result_inchunks = _smooth_chunks(
                    ileft, ii, max_in_chunk, result_inchunks
                )
                # Build cross product of result_inchunks[ileft:ii+1]
                result_outchunks[oi] = _calc_lower_dimension_chunks(
                    result_inchunks, ileft, ii
                )

            oi -= 1
            ii = ileft - 1
        elif din > dout:  # (64,) -> (4, 4, 4)
            if disallow_dimension_expansion:
                raise NotImplementedError(
                    "reshape_blockwise not implemented for expanding dimensions without passing chunk hints."
                )
            oleft = oi - 1
            while oleft >= 0 and reduce(mul, outshape[oleft : oi + 1]) < din:
                oleft -= 1
            if reduce(mul, outshape[oleft : oi + 1]) != din:
                raise NotImplementedError(_not_implemented_message)
            # TODO: don't coalesce shapes unnecessarily
            cs = reduce(mul, outshape[oleft + 1 : oi + 1])

            result_inchunks[ii] = contract_tuple(inchunks[ii], cs)  # (16, 16, 16, 16)

            for i in range(oleft + 1, oi + 1):
                result_outchunks[i] = (outshape[i],)

            result_outchunks[oleft] = tuple(c // cs for c in result_inchunks[ii])

            max_in_chunk = _cal_max_chunk_size(inchunks, ii, ii)
            result_outchunks = _smooth_chunks(oleft, oi, max_in_chunk, result_outchunks)
            # Build cross product of result_outchunks[oleft:oi+1]
            result_inchunks[ii] = _calc_lower_dimension_chunks(
                result_outchunks, oleft, oi
            )
            oi = oleft - 1
            ii -= 1

    return tuple(result_inchunks), tuple(result_outchunks), mapper_in, one_dimensions


def _calc_lower_dimension_chunks(chunks, start, stop):
    # We need the lower dimension chunks to match what the higher dimension chunks
    # can be combined to, i.e. multiply the different dimensions
    return tuple(
        map(
            lambda x: reduce(mul, x),
            product(*chunks[start : stop + 1]),
        )
    )


def _smooth_chunks(ileft, ii, max_in_chunk, result_inchunks):
    # The previous step squashed the whole dimension into a single
    # chunk for ileft + 1 (and potentially combined too many elements
    # into a single chunk for ileft as well). We split up the single
    # chunk into multiple chunks to match the max_in_chunk to keep
    # chunksizes consistent:
    # ((1, 1), (200)) -> ((1, 1), (20, ) * 10) for max_in_chunk = 20
    # It's important to ensure that all dimensions before the dimension
    # we adjust have all-1 chunks to respect C contiguous arrays
    # during the reshaping

    ileft_orig = ileft
    max_result_in_chunk = _cal_max_chunk_size(result_inchunks, ileft, ii)
    if max_in_chunk == max_result_in_chunk:
        # reshaping doesn't mess up
        return result_inchunks

    while all(x == 1 for x in result_inchunks[ileft]):
        # Find the first dimension where we can split chunks
        ileft += 1

    if ileft < ii + 1:
        factor = math.ceil(max_result_in_chunk / max_in_chunk)
        result_in_chunk = result_inchunks[ileft]

        if len(result_in_chunk) == 1:
            # This is a trivial case, when we arrive here is the chunk we are
            # splitting the same length as the whole dimension and all previous
            # chunks that are reshaped into the same dimension are all-one.
            # So we can split this dimension.
            elem = result_in_chunk[0]
            factor = min(factor, elem)
            ceil_elem = math.ceil(elem / factor)
            new_inchunk = [ceil_elem] * factor
            for i in range(ceil_elem * factor - elem):
                new_inchunk[i] -= 1
            result_inchunks[ileft] = tuple(new_inchunk)

            if all(x == 1 for x in new_inchunk) and ileft < ii:
                # might have to do another round
                return _smooth_chunks(ileft_orig, ii, max_in_chunk, result_inchunks)
        else:
            # We are now in the more complicated case. The first dimension in the set
            # of dimensions to squash has non-ones and our max chunk is bigger than
            # what we want. We need to split the non-ones into multiple chunks along
            # this axis.
            other_max_chunk = max_result_in_chunk // max(result_inchunks[ileft])
            result_in = []

            for elem_in in result_in_chunk:
                if elem_in * other_max_chunk <= max_in_chunk:
                    result_in.append(elem_in)
                    continue

                factor = math.ceil(elem_in * other_max_chunk / max_in_chunk)
                ceil_elem = math.ceil(elem_in / factor)
                new_in_chunk = [ceil_elem] * math.ceil(factor)
                for i in range(ceil_elem * factor - elem_in):
                    new_in_chunk[i] -= 1
                result_in.extend(new_in_chunk)

            result_inchunks[ileft] = tuple(result_in)
    return result_inchunks


def _cal_max_chunk_size(chunks, start, stop):
    return int(
        reduce(
            mul,
            [max(chunks[axis]) for axis in range(start, stop + 1)],
        )
    )


def expand_tuple(chunks, factor):
    """
    >>> expand_tuple((2, 4), 2)
    (1, 1, 2, 2)

    >>> expand_tuple((2, 4), 3)
    (1, 1, 1, 1, 2)

    >>> expand_tuple((3, 4), 2)
    (1, 2, 2, 2)

    >>> expand_tuple((7, 4), 3)
    (2, 2, 3, 1, 1, 2)
    """
    if factor == 1:
        return chunks

    out = []
    for c in chunks:
        x = c
        part = max(x / factor, 1)
        while x >= 2 * part:
            out.append(int(part))
            x -= int(part)
        if x:
            out.append(x)
    assert sum(chunks) == sum(out)
    return tuple(out)


def contract_tuple(chunks, factor):
    """Return simple chunks tuple such that factor divides all elements

    Examples
    --------
    >>> contract_tuple((2, 2, 8, 4), 4)
    (4, 8, 4)
    """
    assert sum(chunks) % factor == 0

    out = []
    residual = 0
    for chunk in chunks:
        chunk += residual
        div = chunk // factor
        residual = chunk % factor
        good = factor * div
        if good:
            out.append(good)
    return tuple(out)


# --------------------------------------------------------------------------
# End of reshape_rechunk helpers
# --------------------------------------------------------------------------


class Reshape(ArrayExpr):
    """Reshape array to new shape.

    This is the high-level expression that gets lowered to ReshapeLowered.
    The lowering step computes the required rechunking.
    """

    _parameters = ["array", "_shape"]

    def __new__(cls, *args, **kwargs):
        # Call parent __new__ to create the instance
        instance = super().__new__(cls, *args, **kwargs)
        # Eagerly validate by computing chunks (which calls reshape_rechunk)
        # This ensures NotImplementedError is raised at creation time
        _ = instance.chunks
        return instance

    @functools.cached_property
    def _meta(self):
        return meta_from_array(self.array._meta, ndim=len(self._shape))

    @functools.cached_property
    def _reshape_chunks(self):
        """Compute input and output chunks for reshape."""
        inchunks, outchunks, _, _ = reshape_rechunk(
            self.array.shape, self._shape, self.array.chunks
        )
        return inchunks, outchunks

    @property
    def _inchunks(self):
        return self._reshape_chunks[0]

    @property
    def _outchunks(self):
        return self._reshape_chunks[1]

    @functools.cached_property
    def chunks(self):
        return self._outchunks

    def _lower(self):
        """Lower to ReshapeLowered with the rechunked array as an operand."""
        if self._inchunks == self.array.chunks:
            rechunked = self.array
        else:
            rechunked = self.array.rechunk(self._inchunks)
        return ReshapeLowered(rechunked, self._shape, self._outchunks)

    def _simplify_up(self, parent, dependents):
        """Allow slice operations to push through Reshape."""
        from dask_array.slicing import SliceSlicesIntegers

        if isinstance(parent, SliceSlicesIntegers):
            return self._accept_slice(parent)
        return None

    def _accept_slice(self, slice_expr):
        """Accept a slice being pushed through Reshape.

        Reshape can be pushed through when the slice only affects dimensions
        that have the same size in both input and output shapes (preserved dims).

        For example:
            x.reshape((10, 2, 3))[:5]  # (10, 6) -> (10, 2, 3), first dim preserved
            becomes: x[:5].reshape((5, 2, 3))
        """
        from numbers import Integral

        from dask._collections import new_collection

        in_shape = self.array.shape
        out_shape = self._shape
        index = slice_expr.index

        # Separate None (newaxis) from real indices
        # None insertions don't interact with reshape and can be re-applied after
        none_positions = []  # positions where None appears in original index
        stripped_index = []  # index without Nones
        for i, idx in enumerate(index):
            if idx is None:
                none_positions.append(i)
            else:
                stripped_index.append(idx)

        # Pad stripped index to output ndim
        out_ndim = len(out_shape)
        full_index = list(stripped_index) + [slice(None)] * (
            out_ndim - len(stripped_index)
        )

        # Find how many leading dimensions are preserved (same size in both shapes)
        preserved_dims = 0
        for in_size, out_size in zip(in_shape, out_shape):
            if in_size == out_size:
                preserved_dims += 1
            else:
                break

        if preserved_dims == 0:
            return None  # No preserved dimensions, can't push through

        # Check if slice only affects preserved dimensions
        # (non-preserved dims must all be slice(None))
        if any(
            isinstance(idx, Integral) or idx != slice(None)
            for idx in full_index[preserved_dims:]
        ):
            return None

        # Build the input slice (only on preserved dims, same indices)
        in_ndim = len(in_shape)
        input_index = list(full_index[:preserved_dims])
        input_index += [slice(None)] * (in_ndim - preserved_dims)

        # Compute new output shape after slicing
        new_out_shape = []
        for idx, size in zip(full_index, out_shape):
            if isinstance(idx, Integral):
                # Integer index removes dimension
                continue
            elif idx == slice(None):
                new_out_shape.append(size)
            else:
                # Normalize slice
                start, stop, step = idx.indices(size)
                if step != 1:
                    return None  # Don't handle non-unit steps
                new_out_shape.append(stop - start)

        new_out_shape = tuple(new_out_shape)

        # Apply slice to input, then reshape
        sliced_input = new_collection(self.array)[tuple(input_index)]
        result = Reshape(sliced_input.expr, new_out_shape)

        # Re-apply None insertions if any using expand_dims
        if none_positions:
            from dask_array.manipulation._expand import expand_dims

            # Compute where Nones should be inserted in the OUTPUT of reshape
            # Account for integer indices that remove dimensions
            axes = []
            for pos in none_positions:
                # Count how many real (non-None) indices come before this position
                real_before = sum(1 for idx in index[:pos] if idx is not None)
                # Account for integer indices that removed dimensions
                ints_before = sum(
                    1
                    for idx in stripped_index[:real_before]
                    if isinstance(idx, Integral)
                )
                axes.append(
                    pos - len([p for p in none_positions if p < pos]) - ints_before
                )

            return expand_dims(new_collection(result), axis=tuple(axes)).expr

        return result


class ReshapeLowered(ArrayExpr):
    """Lowered reshape expression with rechunked input as operand."""

    _parameters = ["array", "_shape", "_outchunks"]

    @functools.cached_property
    def _name(self):
        return f"reshape-{self.deterministic_token}"

    @functools.cached_property
    def _meta(self):
        return meta_from_array(self.array._meta, ndim=len(self._shape))

    @functools.cached_property
    def chunks(self):
        return self._outchunks

    def _layer(self) -> dict:
        inchunks = self.array.chunks
        outchunks = self._outchunks

        in_keys = list(product([self.array._name], *[range(len(c)) for c in inchunks]))
        out_keys = list(product([self._name], *[range(len(c)) for c in outchunks]))
        shapes = list(product(*outchunks))

        dsk = {
            out_key: Task(out_key, M.reshape, TaskRef(in_key), shape)
            for out_key, in_key, shape in zip(out_keys, in_keys, shapes)
        }
        return dsk


def reshape(x, shape, merge_chunks=True, limit=None):
    """Reshape array to new shape.

    Parameters
    ----------
    x : Array
        Input array
    shape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.
    merge_chunks : bool, default True
        Whether to merge chunks using the logic in :meth:`dask.array.rechunk`
        when communication is necessary given the input array chunking and
        the output shape.
    limit : int (optional)
        The maximum block size to target in bytes.

    Returns
    -------
    reshaped : Array
    """
    from dask._collections import new_collection

    # Normalize shape
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(map(sanitize_index, shape))

    # Handle -1 in shape
    known_sizes = [s for s in shape if s != -1]
    if len(known_sizes) < len(shape):
        if len(shape) - len(known_sizes) > 1:
            raise ValueError("can only specify one unknown dimension")
        # Fastpath for x.reshape(-1) on 1D arrays
        if len(shape) == 1 and x.ndim == 1:
            return new_collection(x.expr)
        missing_size = sanitize_index(x.size / reduce(mul, known_sizes, 1))
        shape = tuple(missing_size if s == -1 else s for s in shape)

    # Sanity checks
    if np.isnan(sum(x.shape)):
        raise ValueError(
            f"Array chunk size or shape is unknown. shape: {x.shape}\n\n"
            "Possible solution with x.compute_chunk_sizes()"
        )
    if reduce(mul, shape, 1) != x.size:
        raise ValueError("total size of new array must be unchanged")

    # Identity reshape - return input unchanged
    if x.shape == shape:
        return x

    # Single partition case: use simple blockwise reshape
    expr = x.expr
    npartitions = reduce(mul, (len(c) for c in expr.chunks), 1)
    if npartitions == 1:
        return new_collection(ReshapeLowered(expr, shape, tuple((d,) for d in shape)))

    # Handle merge_chunks=False: pre-rechunk to size-1 chunks in early dimensions
    if not merge_chunks and x.ndim > len(shape):
        pre_rechunk = dict.fromkeys(range(x.ndim - len(shape)), 1)
        expr = expr.rechunk(pre_rechunk)

    return new_collection(Reshape(expr, shape))


class ReshapeBlockwise(ArrayExpr):
    """Blockwise reshape - each block reshaped independently.

    Unlike regular Reshape, this doesn't rechunk. Each block is independently
    reshaped and the results are concatenated. The output may have different
    element ordering than NumPy's reshape.
    """

    _parameters = ["array", "_shape", "_chunks"]
    _defaults = {"_chunks": None}

    @functools.cached_property
    def _meta(self):
        return meta_from_array(self.array._meta, ndim=len(self._shape))

    @functools.cached_property
    def _reshape_info(self):
        """Compute reshape mapping info (cached to avoid recomputation)."""
        if len(self._shape) > self.array.ndim:
            return None  # Expansion case uses provided chunks directly

        _, _, mapper_in, one_dimensions = reshape_rechunk(
            self.array.shape,
            self._shape,
            self.array.chunks,
            disallow_dimension_expansion=True,
        )
        return mapper_in, one_dimensions

    @functools.cached_property
    def _out_shapes(self):
        """Per-block output shapes."""
        if len(self._shape) > self.array.ndim:
            return list(product(*(c for c in self._chunks)))

        mapper_in, one_dims = self._reshape_info
        return [
            self._convert_to_shape(c, mapper_in, one_dims)
            for c in product(*(c for c in self.array.chunks))
        ]

    @functools.cached_property
    def chunks(self):
        if len(self._shape) > self.array.ndim:
            if self._chunks is None:
                raise TypeError("Need to specify chunks if expanding dimensions.")
            return self._chunks

        mapper_in, one_dims = self._reshape_info
        nr_out_chunks = self._convert_to_shape(
            tuple(map(len, self.array.chunks)), mapper_in, one_dims
        )

        # Build output chunks from per-block shapes
        output_chunks = []
        ctr = 1
        for i, nr_chunks_dim in enumerate(reversed(nr_out_chunks)):
            dim_chunks = [
                self._out_shapes[elem * ctr][len(nr_out_chunks) - i - 1]
                for elem in range(nr_chunks_dim)
            ]
            output_chunks.append(tuple(dim_chunks))
            ctr *= nr_chunks_dim

        return tuple(reversed(output_chunks))

    @staticmethod
    def _convert_to_shape(shape, mapper_in, one_dims):
        """Map input dimensions to output dimensions."""
        output_shape = [[] for _ in range(len(set(mapper_in.values())) + len(one_dims))]
        for i in one_dims:
            output_shape[i] = [1]
        for k, v in mapper_in.items():
            output_shape[v].append(shape[k])
        return tuple(reduce(mul, x) for x in output_shape)

    def _layer(self) -> dict:
        in_keys = list(
            product([self.array._name], *[range(len(c)) for c in self.array.chunks])
        )
        out_keys = list(product([self._name], *[range(len(c)) for c in self.chunks]))

        return {
            out_key: Task(out_key, M.reshape, TaskRef(in_key), shape)
            for in_key, out_key, shape in zip(in_keys, out_keys, self._out_shapes)
        }


def reshape_blockwise(x, shape, chunks=None):
    """Blockwise-reshape into a new shape.

    The regular reshape operation in Dask preserves C-ordering in the array
    which requires a rechunking for most reshaping operations, making the
    computation relatively expensive.

    Blockwise-reshape reshapes every block into the new shape and concatenates
    the results. This is a trivial blockwise computation but will return the
    result in a different order than NumPy. This is a good solution for
    subsequent operations that don't rely on the order.

    Parameters
    ----------
    x : Array
        The input array to reshape.
    shape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.
    chunks : tuple of tuples of ints, optional
        The chunk sizes for the output array. Required when expanding
        dimensions (increasing ndim). Ignored when collapsing dimensions.

    Returns
    -------
    reshaped : Array

    Notes
    -----
    This is a parallelized version of ``np.reshape`` with the following
    limitations:

    1. It does not return elements in the same order as NumPy would
    2. It only allows for reshapings that collapse like ``(1, 2, 3, 4) -> (1, 6, 4)``

    Examples
    --------
    >>> import dask_array as da
    >>> x = da.from_array(np.arange(0, 27).reshape(3, 3, 3), chunks=(3, 2, (2, 1)))
    >>> result = reshape_blockwise(x, (3, 9))
    >>> result.chunks
    ((3,), (4, 2, 2, 1))
    """
    import math

    from dask._collections import new_collection
    from dask_array.core import asarray

    x = asarray(x)

    if shape in [-1, (-1,)]:
        shape = (reduce(mul, x.shape),)

    if not isinstance(shape, tuple):
        shape = (shape,)

    # Validate shape
    if np.isnan(sum(x.shape)):
        raise ValueError(
            f"Array chunk size or shape is unknown. shape: {x.shape}\n\n"
            "Possible solution with x.compute_chunk_sizes()"
        )
    if reduce(mul, shape, 1) != x.size:
        raise ValueError("total size of new array must be unchanged")

    # Identity reshape
    if len(shape) == x.ndim and shape == x.shape:
        return x

    # Validate chunks for expansion
    if len(shape) > x.ndim:
        if chunks is None:
            raise TypeError("Need to specify chunks if expanding dimensions.")
        out_shapes = list(product(*(c for c in chunks)))
        in_shapes = list(product(*(c for c in x.chunks)))
        non_matching_chunks = [
            (i, in_c, out_c)
            for i, (in_c, out_c) in enumerate(zip(in_shapes, out_shapes))
            if math.prod(in_c) != math.prod(out_c)
        ]
        if non_matching_chunks:
            raise ValueError(
                f"Chunk sizes do not match for the following chunks: "
                f"{[c[0] for c in non_matching_chunks[:5]]}. \n"
                f"The corresponding chunksizes are: {[c[1:] for c in non_matching_chunks[:5]]}. "
                f"(restricted to first 5 entries)."
            )
    elif chunks is not None:
        raise ValueError(
            "Setting chunks is not allowed when reducing the number of dimensions."
        )

    return new_collection(ReshapeBlockwise(x.expr, shape, chunks))


def ravel(array_like):
    """Return a flattened array.

    Parameters
    ----------
    array_like : array_like
        Input array. Non-array inputs are converted to arrays.

    Returns
    -------
    raveled : Array
        A 1-D array containing the elements of the input.

    See Also
    --------
    numpy.ravel

    Examples
    --------
    >>> import dask_array as da
    >>> x = da.ones((2, 3), chunks=2)
    >>> da.ravel(x).compute()
    array([1., 1., 1., 1., 1., 1.])
    """
    from dask_array.core import asanyarray

    return asanyarray(array_like).reshape((-1,))
