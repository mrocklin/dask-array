from __future__ import annotations

import heapq
import itertools
import math
import operator
from functools import reduce
from itertools import chain, product
from operator import add, itemgetter, mul
from warnings import warn

import numpy as np
import toolz
from tlz import accumulate

from dask import config
from dask._task_spec import Alias, List, Task, TaskRef
from dask.base import tokenize
from dask.utils import cached_property, parse_bytes

from dask_array._expr import ArrayExpr
from dask_array._core_utils import concatenate3, normalize_chunks
from dask_array._utils import validate_axis


# ============================================================================
# Rechunk planning utilities (copied from dask.array.rechunk)
# ============================================================================


def cumdims_label(chunks, const):
    """Internal utility for cumulative sum with label.

    >>> cumdims_label(((5, 3, 3), (2, 2, 1)), 'n')  # doctest: +NORMALIZE_WHITESPACE
    [(('n', 0), ('n', 5), ('n', 8), ('n', 11)),
     (('n', 0), ('n', 2), ('n', 4), ('n', 5))]
    """
    return [
        tuple(zip((const,) * (1 + len(bds)), accumulate(add, (0,) + bds)))
        for bds in chunks
    ]


def _breakpoints(cumold, cumnew):
    """
    >>> new = cumdims_label(((2, 3), (2, 2, 1)), 'n')
    >>> old = cumdims_label(((2, 2, 1), (5,)), 'o')

    >>> _breakpoints(new[0], old[0])
    (('n', 0), ('o', 0), ('n', 2), ('o', 2), ('o', 4), ('n', 5), ('o', 5))
    >>> _breakpoints(new[1], old[1])
    (('n', 0), ('o', 0), ('n', 2), ('n', 4), ('n', 5), ('o', 5))
    """
    return tuple(sorted(cumold + cumnew, key=itemgetter(1)))


def _intersect_1d(breaks):
    """
    Internal utility to intersect chunks for 1d after preprocessing.

    >>> new = cumdims_label(((2, 3), (2, 2, 1)), 'n')
    >>> old = cumdims_label(((2, 2, 1), (5,)), 'o')

    >>> _intersect_1d(_breakpoints(old[0], new[0]))  # doctest: +NORMALIZE_WHITESPACE
    [[(0, slice(0, 2, None))],
     [(1, slice(0, 2, None)), (2, slice(0, 1, None))]]
    >>> _intersect_1d(_breakpoints(old[1], new[1]))  # doctest: +NORMALIZE_WHITESPACE
    [[(0, slice(0, 2, None))],
     [(0, slice(2, 4, None))],
     [(0, slice(4, 5, None))]]

    Parameters
    ----------
    breaks: list of tuples
        Each tuple is ('o', 8) or ('n', 8)
        These are pairs of 'o' old or new 'n'
        indicator with a corresponding cumulative sum,
        or breakpoint (a position along the chunking axis).
        The list of pairs is already ordered by breakpoint.
        Note that an 'o' pair always occurs BEFORE
        an 'n' pair if both share the same breakpoint.
    Uses 'o' and 'n' to make new tuples of slices for
    the new block crosswalk to old blocks.
    """
    o_pairs = [pair for pair in breaks if pair[0] == "o"]
    last_old_chunk_idx = len(o_pairs) - 2
    last_o_br = o_pairs[-1][1]

    start = 0
    last_end = 0
    old_idx = 0
    last_o_end = 0
    ret = []
    ret_next = []
    for idx in range(1, len(breaks)):
        label, br = breaks[idx]
        last_label, last_br = breaks[idx - 1]
        if last_label == "n":
            start = last_end
            if ret_next:
                ret.append(ret_next)
                ret_next = []
        else:
            start = 0
        end = br - last_br + start
        last_end = end
        if br == last_br:
            if label == "o":
                old_idx += 1
                last_o_end = end
            if label == "n" and last_label == "n":
                if br == last_o_br:
                    slc = slice(last_o_end, last_o_end)
                    ret_next.append((last_old_chunk_idx, slc))
                    continue
            else:
                continue
        ret_next.append((old_idx, slice(start, end)))
        if label == "o":
            old_idx += 1
            start = 0
            last_o_end = end

    if ret_next:
        ret.append(ret_next)

    return ret


def old_to_new(old_chunks, new_chunks):
    """Helper to build old_chunks to new_chunks.

    Handles missing values, as long as the dimension with the missing chunk values
    is unchanged.

    Examples
    --------
    >>> old = ((10, 10, 10, 10, 10), )
    >>> new = ((25, 5, 20), )
    >>> old_to_new(old, new)  # doctest: +NORMALIZE_WHITESPACE
    [[[(0, slice(0, 10, None)), (1, slice(0, 10, None)), (2, slice(0, 5, None))],
      [(2, slice(5, 10, None))],
      [(3, slice(0, 10, None)), (4, slice(0, 10, None))]]]
    """

    def is_unknown(dim):
        return any(math.isnan(chunk) for chunk in dim)

    dims_unknown = [is_unknown(dim) for dim in old_chunks]

    known_indices = []
    unknown_indices = []
    for i, unknown in enumerate(dims_unknown):
        if unknown:
            unknown_indices.append(i)
        else:
            known_indices.append(i)

    old_known = [old_chunks[i] for i in known_indices]
    new_known = [new_chunks[i] for i in known_indices]

    cmos = cumdims_label(old_known, "o")
    cmns = cumdims_label(new_known, "n")

    sliced = [None] * len(old_chunks)
    for i, cmo, cmn in zip(known_indices, cmos, cmns):
        sliced[i] = _intersect_1d(_breakpoints(cmo, cmn))

    for i in unknown_indices:
        dim = old_chunks[i]
        extra = [
            [(j, slice(0, size if not math.isnan(size) else None))]
            for j, size in enumerate(dim)
        ]
        sliced[i] = extra
    assert all(x is not None for x in sliced)
    return sliced


def intersect_chunks(old_chunks, new_chunks):
    """
    Make dask.array slices as intersection of old and new chunks.

    >>> intersections = intersect_chunks(((4, 4), (2,)),
    ...                                  ((8,), (1, 1)))
    >>> list(intersections)  # doctest: +NORMALIZE_WHITESPACE
    [(((0, slice(0, 4, None)), (0, slice(0, 1, None))),
      ((1, slice(0, 4, None)), (0, slice(0, 1, None)))),
     (((0, slice(0, 4, None)), (0, slice(1, 2, None))),
      ((1, slice(0, 4, None)), (0, slice(1, 2, None))))]

    Parameters
    ----------
    old_chunks : iterable of tuples
        block sizes along each dimension (convert from old_chunks)
    new_chunks: iterable of tuples
        block sizes along each dimension (converts to new_chunks)
    """
    cross1 = product(*old_to_new(old_chunks, new_chunks))
    cross = chain(tuple(product(*cr)) for cr in cross1)
    return cross


def _validate_rechunk(old_chunks, new_chunks):
    """Validates that rechunking an array from ``old_chunks`` to ``new_chunks``
    is possible, raises an error if otherwise.
    """
    assert len(old_chunks) == len(new_chunks)

    old_shapes = tuple(map(sum, old_chunks))
    new_shapes = tuple(map(sum, new_chunks))

    for old_shape, old_dim, new_shape, new_dim in zip(
        old_shapes, old_chunks, new_shapes, new_chunks
    ):
        if old_shape != new_shape:
            if not (
                math.isnan(old_shape) and math.isnan(new_shape)
            ) or not np.array_equal(old_dim, new_dim, equal_nan=True):
                raise ValueError(
                    "Chunks must be unchanging along dimensions with missing values.\n\n"
                    "A possible solution:\n  x.compute_chunk_sizes()"
                )


def _number_of_blocks(chunks):
    return reduce(mul, map(len, chunks))


def _largest_block_size(chunks):
    return reduce(mul, map(max, chunks))


def estimate_graph_size(old_chunks, new_chunks):
    """Estimate the graph size during a rechunk computation."""
    crossed_size = reduce(
        mul,
        (
            (len(oc) + len(nc) - 1 if oc != nc else len(oc))
            for oc, nc in zip(old_chunks, new_chunks)
        ),
    )
    return crossed_size


def divide_to_width(desired_chunks, max_width):
    """Minimally divide the given chunks so as to make the largest chunk
    width less or equal than *max_width*.
    """
    chunks = []
    for c in desired_chunks:
        nb_divides = int(np.ceil(c / max_width))
        for i in range(nb_divides):
            n = c // (nb_divides - i)
            chunks.append(n)
            c -= n
        assert c == 0
    return tuple(chunks)


def merge_to_number(desired_chunks, max_number):
    """Minimally merge the given chunks so as to drop the number of
    chunks below *max_number*, while minimizing the largest width.
    """
    if len(desired_chunks) <= max_number:
        return desired_chunks

    distinct = set(desired_chunks)
    if len(distinct) == 1:
        w = distinct.pop()
        n = len(desired_chunks)
        total = n * w

        desired_width = total // max_number
        width = w * (desired_width // w)
        adjust = (total - max_number * width) // w

        return (width + w,) * adjust + (width,) * (max_number - adjust)

    desired_width = sum(desired_chunks) // max_number
    nmerges = len(desired_chunks) - max_number

    heap = [
        (desired_chunks[i] + desired_chunks[i + 1], i, i + 1)
        for i in range(len(desired_chunks) - 1)
    ]
    heapq.heapify(heap)

    chunks = list(desired_chunks)

    while nmerges > 0:
        width, i, j = heapq.heappop(heap)
        if chunks[j] == 0:
            j += 1
            while chunks[j] == 0:
                j += 1
            heapq.heappush(heap, (chunks[i] + chunks[j], i, j))
            continue
        elif chunks[i] + chunks[j] != width:
            heapq.heappush(heap, (chunks[i] + chunks[j], i, j))
            continue
        assert chunks[i] != 0
        chunks[i] = 0
        chunks[j] = width
        nmerges -= 1

    return tuple(filter(None, chunks))


def find_merge_rechunk(old_chunks, new_chunks, block_size_limit):
    """
    Find an intermediate rechunk that would merge some adjacent blocks
    together in order to get us nearer the *new_chunks* target, without
    violating the *block_size_limit* (in number of elements).
    """
    ndim = len(old_chunks)

    old_largest_width = [max(c) for c in old_chunks]
    new_largest_width = [max(c) for c in new_chunks]

    graph_size_effect = {
        dim: len(nc) / len(oc)
        for dim, (oc, nc) in enumerate(zip(old_chunks, new_chunks))
    }

    block_size_effect = {
        dim: new_largest_width[dim] / (old_largest_width[dim] or 1)
        for dim in range(ndim)
    }

    merge_candidates = [dim for dim in range(ndim) if graph_size_effect[dim] <= 1.0]

    def key(k):
        gse = graph_size_effect[k]
        bse = block_size_effect[k]
        if bse == 1:
            bse = 1 + 1e-9
        return (np.log(gse) / np.log(bse)) if bse > 0 else 0

    sorted_candidates = sorted(merge_candidates, key=key)

    largest_block_size = reduce(mul, old_largest_width)

    chunks = list(old_chunks)
    memory_limit_hit = False

    for dim in sorted_candidates:
        new_largest_block_size = (
            largest_block_size * new_largest_width[dim] // (old_largest_width[dim] or 1)
        )
        if new_largest_block_size <= block_size_limit:
            chunks[dim] = new_chunks[dim]
            largest_block_size = new_largest_block_size
        else:
            largest_width = old_largest_width[dim]
            chunk_limit = int(block_size_limit * largest_width / largest_block_size)
            c = divide_to_width(new_chunks[dim], chunk_limit)
            if len(c) <= len(old_chunks[dim]):
                chunks[dim] = c
                largest_block_size = largest_block_size * max(c) // largest_width

            memory_limit_hit = True

    assert largest_block_size == _largest_block_size(chunks)
    assert largest_block_size <= block_size_limit
    return tuple(chunks), memory_limit_hit


def find_split_rechunk(old_chunks, new_chunks, graph_size_limit):
    """
    Find an intermediate rechunk that would split some chunks to
    get us nearer *new_chunks*, without violating the *graph_size_limit*.
    """
    ndim = len(old_chunks)

    chunks = list(old_chunks)

    for dim in range(ndim):
        graph_size = estimate_graph_size(chunks, new_chunks)
        if graph_size > graph_size_limit:
            break
        if len(old_chunks[dim]) > len(new_chunks[dim]):
            continue
        max_number = int(len(old_chunks[dim]) * graph_size_limit / graph_size)
        c = merge_to_number(new_chunks[dim], max_number)
        assert len(c) <= max_number
        if len(c) >= len(old_chunks[dim]) and max(c) <= max(old_chunks[dim]):
            chunks[dim] = c

    return tuple(chunks)


def _graph_size_threshold(old_chunks, new_chunks, threshold):
    return threshold * (_number_of_blocks(old_chunks) + _number_of_blocks(new_chunks))


def plan_rechunk(
    old_chunks, new_chunks, itemsize, threshold=None, block_size_limit=None
):
    """Plan an iterative rechunking from *old_chunks* to *new_chunks*.
    The plan aims to minimize the rechunk graph size.

    Parameters
    ----------
    itemsize: int
        The item size of the array
    threshold: int
        The graph growth factor under which we don't bother
        introducing an intermediate step
    block_size_limit: int
        The maximum block size (in bytes) we want to produce during an
        intermediate step
    """
    threshold = threshold or config.get("array.rechunk.threshold")
    block_size_limit = block_size_limit or config.get("array.chunk-size")
    if isinstance(block_size_limit, str):
        block_size_limit = parse_bytes(block_size_limit)

    has_nans = (any(math.isnan(y) for y in x) for x in old_chunks)

    if len(new_chunks) <= 1 or not all(new_chunks) or any(has_nans):
        return [new_chunks]

    block_size_limit /= itemsize

    largest_old_block = _largest_block_size(old_chunks)
    largest_new_block = _largest_block_size(new_chunks)
    block_size_limit = max([block_size_limit, largest_old_block, largest_new_block])

    graph_size_threshold = _graph_size_threshold(old_chunks, new_chunks, threshold)

    current_chunks = old_chunks
    first_pass = True
    steps = []

    while True:
        graph_size = estimate_graph_size(current_chunks, new_chunks)
        if graph_size < graph_size_threshold:
            break

        if first_pass:
            chunks = current_chunks
        else:
            chunks = find_split_rechunk(
                current_chunks, new_chunks, graph_size * threshold
            )
        chunks, memory_limit_hit = find_merge_rechunk(
            chunks, new_chunks, block_size_limit
        )
        if (chunks == current_chunks and not first_pass) or chunks == new_chunks:
            break
        if chunks != current_chunks:
            steps.append(chunks)
        current_chunks = chunks
        if not memory_limit_hit:
            break
        first_pass = False

    return steps + [new_chunks]


def _get_chunks(n, chunksize):
    leftover = n % chunksize
    n_chunks = n // chunksize

    chunks = [chunksize] * n_chunks
    if leftover:
        chunks.append(leftover)
    return tuple(chunks)


def _balance_chunksizes(chunks: tuple[int, ...]) -> tuple[int, ...]:
    """
    Balance the chunk sizes

    Parameters
    ----------
    chunks : tuple[int, ...]
        Chunk sizes for Dask array.

    Returns
    -------
    new_chunks : tuple[int, ...]
        New chunks for Dask array with balanced sizes.
    """
    median_len = np.median(chunks).astype(int)
    n_chunks = len(chunks)
    eps = median_len // 2
    if min(chunks) <= 0.5 * max(chunks):
        n_chunks -= 1

    new_chunks = [
        _get_chunks(sum(chunks), chunk_len)
        for chunk_len in range(median_len - eps, median_len + eps + 1)
    ]
    possible_chunks = [c for c in new_chunks if len(c) == n_chunks]
    if not len(possible_chunks):
        warn(
            "chunk size balancing not possible with given chunks. "
            "Try increasing the chunk size."
        )
        return chunks

    diffs = [max(c) - min(c) for c in possible_chunks]
    best_chunk_size = np.argmin(diffs)
    return possible_chunks[best_chunk_size]


def _choose_rechunk_method(old_chunks, new_chunks, threshold=None):
    if method := config.get("array.rechunk.method", None):
        return method
    try:
        from distributed import default_client

        default_client()
    except (ImportError, ValueError):
        return "tasks"

    _old_to_new = old_to_new(old_chunks, new_chunks)
    graph_size = math.prod(sum(len(ins) for ins in axis) for axis in _old_to_new)
    threshold = threshold or config.get("array.rechunk.threshold")
    graph_size_threshold = _graph_size_threshold(old_chunks, new_chunks, threshold)
    return "tasks" if graph_size < graph_size_threshold else "p2p"


# ============================================================================
# Expression classes
# ============================================================================


class Rechunk(ArrayExpr):
    _parameters = [
        "array",
        "_chunks",
        "threshold",
        "block_size_limit",
        "balance",
        "method",
    ]

    _defaults = {
        "_chunks": "auto",
        "threshold": None,
        "block_size_limit": None,
        "balance": None,
        "method": None,
    }

    @property
    def _meta(self):
        return self.array._meta

    @property
    def _name(self):
        return "rechunk-merge-" + tokenize(*self.operands)

    @cached_property
    def chunks(self):
        x = self.array
        chunks = self.operand("_chunks")

        # don't rechunk if array is empty
        if x.ndim > 0 and all(s == 0 for s in x.shape):
            return x.chunks

        if isinstance(chunks, dict):
            chunks = {validate_axis(c, x.ndim): v for c, v in chunks.items()}
            for i in range(x.ndim):
                if i not in chunks:
                    chunks[i] = x.chunks[i]
                elif chunks[i] is None:
                    chunks[i] = x.chunks[i]
        if isinstance(chunks, (tuple, list)):
            chunks = tuple(
                lc if lc is not None else rc for lc, rc in zip(chunks, x.chunks)
            )
        chunks = normalize_chunks(
            chunks,
            x.shape,
            limit=self.block_size_limit,
            dtype=x.dtype,
            previous_chunks=x.chunks,
        )

        if not len(chunks) == x.ndim:
            raise ValueError("Provided chunks are not consistent with shape")

        if self.balance:
            chunks = tuple(_balance_chunksizes(chunk) for chunk in chunks)

        _validate_rechunk(x.chunks, chunks)

        return chunks

    def _simplify_down(self):
        # No-op rechunk: if chunks already match, return the original array
        if not self.balance and self.chunks == self.array.chunks:
            return self.array

        from dask_array._blockwise import Elemwise
        from dask_array.manipulation._transpose import Transpose

        # Rechunk(Rechunk(x)) -> single Rechunk to final chunks
        # Only match Rechunk, not TasksRechunk (which is already lowered)
        # Don't merge if inner has method='p2p' - preserve explicit p2p semantics
        if type(self.array) is Rechunk and self.array.method != "p2p":
            return Rechunk(
                self.array.array,
                self._chunks,
                self.threshold,
                self.block_size_limit,
                self.balance or self.array.balance,
                self.method,
            )

        # Rechunk(Transpose) -> Transpose(rechunked input)
        if isinstance(self.array, Transpose):
            return self._pushdown_through_transpose()

        # Rechunk(Elemwise) -> Elemwise(rechunked inputs)
        if isinstance(self.array, Elemwise):
            return self._pushdown_through_elemwise()

        # Rechunk(Concatenate) -> Concatenate(rechunked inputs)
        # Only for non-concat axes
        from dask_array._concatenate import Concatenate

        if isinstance(self.array, Concatenate):
            return self._pushdown_through_concatenate()

        # Rechunk(IO) -> IO with new chunks (if IO supports it)
        # Skip if method='p2p' is explicitly requested - user wants distributed shuffle
        if getattr(self.array, "_can_rechunk_pushdown", False) and self.method != "p2p":
            # Keep the same name prefix - the token will change with the new chunks
            return self.array.substitute_parameters({"_chunks": self.chunks})

    def _pushdown_through_transpose(self):
        """Push rechunk through transpose by reordering chunk spec."""
        from dask_array.manipulation._transpose import Transpose

        transpose = self.array
        axes = transpose.axes
        chunks = self._chunks

        if isinstance(chunks, tuple):
            # Map output chunks back through transpose axes to get input chunks
            # axes[i] tells us which input axis becomes output axis i
            # So output axis i has chunks[i], which should go to input axis axes[i]
            # We need to invert the permutation: place chunks[i] at position axes[i]
            new_chunks = [None] * len(axes)
            for i, ax in enumerate(axes):
                new_chunks[ax] = chunks[i]
            new_chunks = tuple(new_chunks)
        elif isinstance(chunks, dict):
            # Map dict keys through axes
            new_chunks = {}
            for out_axis, chunk_spec in chunks.items():
                in_axis = axes[out_axis]
                new_chunks[in_axis] = chunk_spec
        else:
            return None

        rechunked_input = transpose.array.rechunk(new_chunks)
        return Transpose(rechunked_input, axes)

    def _pushdown_through_elemwise(self):
        """Push rechunk through elemwise by rechunking each input."""
        from dask_array._blockwise import Elemwise, is_scalar_for_elemwise
        from dask_array._expr import ArrayExpr

        elemwise = self.array
        out_ind = elemwise.out_ind
        chunks = self._chunks

        # Convert dict chunks to tuple for positional indexing
        if isinstance(chunks, dict):
            chunks = tuple(chunks.get(i, -1) for i in range(elemwise.ndim))

        def rechunk_array_arg(arg):
            """Rechunk an array argument to match target output chunks."""
            if is_scalar_for_elemwise(arg):
                return arg
            if not isinstance(arg, ArrayExpr):
                return arg
            # Map output chunks to this input's dimensions
            # arg has indices tuple(range(arg.ndim)[::-1])
            arg_ind = tuple(range(arg.ndim)[::-1])

            # For each dimension of arg, find where its index appears in out_ind
            arg_chunks = []
            for i, dim_idx in enumerate(arg_ind):
                # Get the arg's dimension size for this position
                arg_dim_size = arg.shape[i]

                # If this dimension is broadcast (size 1), keep its original chunk
                if arg_dim_size == 1:
                    arg_chunks.append((1,))
                    continue

                try:
                    out_pos = out_ind.index(dim_idx)
                    arg_chunks.append(chunks[out_pos])
                except ValueError:
                    # Index not in output (shouldn't happen for elemwise)
                    arg_chunks.append(-1)  # auto

            return arg.rechunk(tuple(arg_chunks))

        new_args = [rechunk_array_arg(arg) for arg in elemwise.elemwise_args]

        # Also rechunk where and out if they are arrays
        new_where = elemwise.where
        if isinstance(new_where, ArrayExpr):
            new_where = rechunk_array_arg(new_where)

        new_out = elemwise.out
        if isinstance(new_out, ArrayExpr):
            new_out = rechunk_array_arg(new_out)

        return Elemwise(
            elemwise.op,
            elemwise.operand("dtype"),
            elemwise.operand("name"),
            new_where,
            new_out,
            elemwise.operand("_user_kwargs"),
            *new_args,
        )

    def _pushdown_through_concatenate(self):
        """Push rechunk through concatenate for non-concat axes."""
        from dask._collections import new_collection

        concat = self.array
        axis = concat.axis
        arrays = concat.args
        chunks = self._chunks

        # Only handle tuple chunks for now
        if not isinstance(chunks, tuple):
            # For dict chunks, check if we're only rechunking non-concat axes
            if isinstance(chunks, dict) and axis not in chunks:
                # Build chunks for each input (same rechunk spec)
                rechunked_arrays = [new_collection(a).rechunk(chunks) for a in arrays]
                return type(concat)(
                    rechunked_arrays[0].expr,
                    axis,
                    concat._meta,
                    *[a.expr for a in rechunked_arrays[1:]],
                )
            return None

        # Only push through if we're not changing the concat axis chunking
        # (redistributing across concat boundaries is too complex)
        if chunks[axis] != concat.chunks[axis]:
            return None

        # Build rechunk spec for each input (excluding concat axis)
        # For the concat axis, each input keeps its original chunks
        rechunked_arrays = []
        for arr in arrays:
            arr_chunks = list(chunks)
            arr_chunks[axis] = arr.chunks[axis]
            rechunked_arrays.append(new_collection(arr).rechunk(tuple(arr_chunks)))

        return type(concat)(
            rechunked_arrays[0].expr,
            axis,
            concat._meta,
            *[a.expr for a in rechunked_arrays[1:]],
        )

    def _lower(self):

        if not self.balance and (self.chunks == self.array.chunks):
            return self.array

        method = self.method or _choose_rechunk_method(
            self.array.chunks, self.chunks, threshold=self.threshold
        )
        if method == "p2p":
            return P2PRechunk(
                self.array,
                self.chunks,
                self.threshold,
                self.block_size_limit,
                self.balance,
            )
        else:
            return TasksRechunk(
                self.array, self.chunks, self.threshold, self.block_size_limit
            )


class TasksRechunk(Rechunk):
    _parameters = ["array", "_chunks", "threshold", "block_size_limit"]

    @cached_property
    def chunks(self):
        return self.operand("_chunks")

    def _simplify_down(self):
        # TasksRechunk is already lowered - don't apply parent's simplifications
        return None

    def _lower(self):
        return

    def _layer(self):
        steps = plan_rechunk(
            self.array.chunks,
            self.chunks,
            self.array.dtype.itemsize,
            self.threshold,
            self.block_size_limit,
        )
        name = self.array.name
        old_chunks = self.array.chunks
        layers = []
        for i, c in enumerate(steps):
            level = len(steps) - i - 1
            name, old_chunks, layer = _compute_rechunk(
                name, old_chunks, c, level, self.name
            )
            layers.append(layer)

        return toolz.merge(*layers)


def _convert_to_task_refs(obj):
    """Recursively convert nested lists of keys to TaskRefs."""
    if isinstance(obj, list):
        return List(*[_convert_to_task_refs(item) for item in obj])
    elif isinstance(obj, tuple):
        # Keys are tuples like (name, i, j, ...)
        return TaskRef(obj)
    else:
        return obj


def _compute_rechunk(old_name, old_chunks, chunks, level, name):
    """Compute the rechunk of *x* to the given *chunks*."""
    ndim = len(old_chunks)
    crossed = intersect_chunks(old_chunks, chunks)
    x2 = {}
    intermediates = {}

    if level != 0:
        merge_name = name.replace("rechunk-merge-", f"rechunk-merge-{level}-")
        split_name = name.replace("rechunk-merge-", f"rechunk-split-{level}-")
    else:
        merge_name = name.replace("rechunk-merge-", "rechunk-merge-")
        split_name = name.replace("rechunk-merge-", "rechunk-split-")
    split_name_suffixes = itertools.count()

    # Pre-allocate old block references
    old_blocks = np.empty([len(c) for c in old_chunks], dtype="O")
    for index in np.ndindex(old_blocks.shape):
        old_blocks[index] = (old_name,) + index

    # Iterate over all new blocks
    new_index = itertools.product(*(range(len(c)) for c in chunks))

    for new_idx, cross1 in zip(new_index, crossed):
        key = (merge_name,) + new_idx
        old_block_indices = [[cr[i][0] for cr in cross1] for i in range(ndim)]
        subdims1 = [len(set(old_block_indices[i])) for i in range(ndim)]

        rec_cat_arg = np.empty(subdims1, dtype="O")
        rec_cat_arg_flat = rec_cat_arg.flat

        # Iterate over the old blocks required to build the new block
        for rec_cat_index, ind_slices in enumerate(cross1):
            old_block_index, slices = zip(*ind_slices)
            intermediate_name = (split_name, next(split_name_suffixes))
            old_index = old_blocks[old_block_index][1:]
            if all(
                slc.start == 0 and slc.stop == old_chunks[i][ind]
                for i, (slc, ind) in enumerate(zip(slices, old_index))
            ):
                # No slicing needed - use old block directly
                rec_cat_arg_flat[rec_cat_index] = old_blocks[old_block_index]
            else:
                # Need to slice the old block
                intermediates[intermediate_name] = Task(
                    intermediate_name,
                    operator.getitem,
                    TaskRef(old_blocks[old_block_index]),
                    slices,
                )
                rec_cat_arg_flat[rec_cat_index] = intermediate_name

        assert rec_cat_index == rec_cat_arg.size - 1

        # New block is formed by concatenation of sliced old blocks
        if all(d == 1 for d in rec_cat_arg.shape):
            # Single source block - alias to it
            source_key = rec_cat_arg.flat[0]
            x2[key] = Alias(key, source_key)
        else:
            # Multiple source blocks - concatenate
            x2[key] = Task(
                key, concatenate3, _convert_to_task_refs(rec_cat_arg.tolist())
            )

    del old_blocks, new_index

    return merge_name, chunks, {**x2, **intermediates}


class P2PRechunk(ArrayExpr):
    """P2P rechunk expression using distributed shuffle."""

    _parameters = ["array", "_chunks", "threshold", "block_size_limit", "balance"]
    _defaults = {
        "threshold": None,
        "block_size_limit": None,
        "balance": False,
    }

    @property
    def _meta(self):
        return self.array._meta

    @property
    def _name(self):
        return "rechunk-p2p-" + tokenize(*self.operands)

    @cached_property
    def chunks(self):
        return self.operand("_chunks")

    @cached_property
    def _prechunked_chunks(self):
        """Calculate chunks needed before the p2p shuffle."""
        from distributed.shuffle._rechunk import _calculate_prechunking

        return _calculate_prechunking(
            self.array.chunks,
            self.chunks,
            self.array.dtype,
            self.block_size_limit,
        )

    @cached_property
    def _prechunked_array(self):
        """Return the input array, potentially prechunked."""
        prechunked = self._prechunked_chunks
        if prechunked != self.array.chunks:
            return TasksRechunk(
                self.array,
                prechunked,
                self.threshold,
                self.block_size_limit,
            )
        return self.array

    def _simplify_down(self):
        # P2PRechunk is a lowered form - don't apply further simplifications
        return None

    def _lower(self):
        return None

    def _layer(self):
        from distributed.shuffle._rechunk import (
            _split_partials,
            partial_concatenate,
            partial_rechunk,
        )

        import dask

        input_name = self._prechunked_array.name
        input_chunks = self._prechunked_chunks
        chunks = self.chunks
        token = tokenize(*self.operands)
        disk = dask.config.get("distributed.p2p.storage.disk")

        _old_to_new = old_to_new(input_chunks, chunks)

        # Create keepmap (all True - no culling at expression level)
        shape = tuple(len(axis) for axis in chunks)
        keepmap = np.ones(shape, dtype=bool)

        dsk = {}
        for ndpartial in _split_partials(_old_to_new):
            partial_keepmap = keepmap[ndpartial.new]
            output_count = np.sum(partial_keepmap)
            if output_count == 0:
                continue
            elif output_count == 1:
                # Single output chunk - use simple concatenation
                dsk.update(
                    partial_concatenate(
                        input_name=input_name,
                        input_chunks=input_chunks,
                        ndpartial=ndpartial,
                        token=token,
                        keepmap=keepmap,
                        old_to_new=_old_to_new,
                    )
                )
            else:
                # Multiple output chunks - use p2p shuffle
                dsk.update(
                    partial_rechunk(
                        input_name=input_name,
                        input_chunks=input_chunks,
                        chunks=chunks,
                        ndpartial=ndpartial,
                        token=token,
                        disk=disk,
                        keepmap=keepmap,
                    )
                )
        return dsk

    def dependencies(self):
        return [self._prechunked_array]


def rechunk(
    x,
    chunks="auto",
    threshold=None,
    block_size_limit=None,
    balance=False,
    method=None,
):
    """
    Convert blocks in dask array x for new chunks.

    Parameters
    ----------
    x: dask array
        Array to be rechunked.
    chunks:  int, tuple, dict or str, optional
        The new block dimensions to create. -1 indicates the full size of the
        corresponding dimension. Default is "auto" which automatically
        determines chunk sizes.
    threshold: int, optional
        The graph growth factor under which we don't bother introducing an
        intermediate step.
    block_size_limit: int, optional
        The maximum block size (in bytes) we want to produce
        Defaults to the configuration value ``array.chunk-size``
    balance : bool, default False
        If True, try to make each chunk to be the same size.

        This means ``balance=True`` will remove any small leftover chunks, so
        using ``x.rechunk(chunks=len(x) // N, balance=True)``
        will almost certainly result in ``N`` chunks.
    method: {'tasks', 'p2p'}, optional.
        Rechunking method to use.


    Examples
    --------
    >>> import dask.array as da
    >>> x = da.ones((1000, 1000), chunks=(100, 100))

    Specify uniform chunk sizes with a tuple

    >>> y = x.rechunk((1000, 10))

    Or chunk only specific dimensions with a dictionary

    >>> y = x.rechunk({0: 1000})

    Use the value ``-1`` to specify that you want a single chunk along a
    dimension or the value ``"auto"`` to specify that dask can freely rechunk a
    dimension to attain blocks of a uniform block size

    >>> y = x.rechunk({0: -1, 1: 'auto'}, block_size_limit=1e8)

    If a chunk size does not divide the dimension then rechunk will leave any
    unevenness to the last chunk.

    >>> x.rechunk(chunks=(400, -1)).chunks
    ((400, 400, 200), (1000,))

    However if you want more balanced chunks, and don't mind Dask choosing a
    different chunksize for you then you can use the ``balance=True`` option.

    >>> x.rechunk(chunks=(400, -1), balance=True).chunks
    ((500, 500), (1000,))
    """
    import dask
    from dask._collections import new_collection

    # Capture config value at creation time, not during lowering
    if method is None:
        method = dask.config.get("array.rechunk.method", None)

    return new_collection(
        x.expr.rechunk(chunks, threshold, block_size_limit, balance, method)
    )
