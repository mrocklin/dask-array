from __future__ import annotations

import functools
from itertools import product

import numpy as np

from dask._task_spec import Task
from dask_array.io._base import IO
from dask_array._utils import meta_from_array


def _apply_call(bundle):
    """Run one packed call ``(func, args, kwargs)``. This is the shared ``func``
    of every ``FromMap`` produced by normalizing a single-call ``FromDelayed``:
    the per-block *value* carries the call, so heterogeneous funcs across blocks
    still share one execution func -- which is what lets two normalized
    ``FromMap``s merge (their ``func`` matches by identity)."""
    func, args, kwargs = bundle
    return func(*args, **kwargs)


def _merge_from_maps(children, axis, new_axis, dtype, meta):
    """Merge sibling ``FromMap``s along ``axis`` into one ``FromMap``.

    ``new_axis=True`` is the ``stack`` case (insert a unit axis at ``axis``);
    ``False`` is ``concatenate``. The block grids combine by exactly
    ``np.stack`` / ``np.concatenate`` on the children's ``values`` arrays, and
    the chunk tuples combine the same way. Returns ``None`` (decline) unless
    every child is a ``FromMap`` sharing one ``func`` and ``kwargs`` over known
    (non-nan) chunks.

    Note the merge is *constructor-scoped*: every normalized-from-delayed map
    shares ``func=_apply_call`` (the call rides in the per-block value), while a
    direct ``from_map`` carries the user's own ``func``. So an all-delayed or
    all-direct sibling set collapses, but a mix of the two declines (still
    correct -- it just stays a ``Concatenate``/``Stack`` of two ``FromMap``s)."""
    if not children or not all(isinstance(c, FromMap) for c in children):
        return None
    func = children[0].operand("func")
    kwargs = children[0].operand("kwargs")
    for c in children:
        if c.operand("func") != func or c.operand("kwargs") != kwargs:
            return None
        if any(b != b for dim in c.chunks for b in dim):  # unknown (nan) chunks
            return None

    values_list = [c.operand("values") for c in children]
    base = children[0].chunks

    if new_axis:
        # stack: every child has the same chunks; add a length-n unit axis.
        if any(c.chunks != base for c in children):
            return None
        n = len(children)
        values = np.stack(values_list, axis=axis)
        chunks = base[:axis] + ((1,) * n,) + base[axis:]
    else:
        # concatenate: non-axis chunks must match; concat the axis chunk list.
        for c in children:
            if c.chunks[:axis] != base[:axis] or c.chunks[axis + 1 :] != base[axis + 1 :]:
                return None
        values = np.concatenate(values_list, axis=axis)
        merged_axis = sum((c.chunks[axis] for c in children), ())
        chunks = base[:axis] + (merged_axis,) + base[axis + 1 :]

    return FromMap(func, values, chunks, dtype, meta, kwargs, None)


def _map_block(func, value, shape, kwargs):
    """Run one block: ``func(value, **kwargs)``, matched to the block's declared
    chunk ``shape``. In the common case the func already returns the chunk shape
    and this is a no-op. The only reshape the rewrites need is inserting unit
    axes (a ``stack`` new axis, e.g. ``(5,) -> (1, 5)``), which never reorders
    data; anything else -- a same-size-but-permuted shape -- would silently
    corrupt the block, so it is rejected. Kept a single task so the block never
    splits into an ungrouped sub-task."""
    x = func(value, **kwargs)
    xshape = getattr(x, "shape", None)
    if xshape is None:
        # A bare scalar (or list) return -- natural for a 0-d block. Coerce to an
        # array so it has a shape to match against the (0-d) chunk.
        x = np.asarray(x)
        xshape = x.shape
    if xshape == shape:
        return x
    if tuple(d for d in xshape if d != 1) != tuple(d for d in shape if d != 1):
        raise ValueError(
            f"from_map block function returned shape {xshape}, which is incompatible "
            f"with the declared chunk shape {shape}"
        )
    return x.reshape(shape)


class FromMap(IO):
    """Build an array by calling ``func`` once per block.

    ``values`` is an object ndarray whose shape *is* the block grid: block
    ``idx`` is ``func(values[idx], **kwargs)``. Because each block is a single
    task under a tuple key ``(name, *idx)``, the whole layer is one clean group
    for Frisky (no opaque bodies), and merging two ``FromMap``s is just
    ``np.stack`` / ``np.concatenate`` on their ``values`` grids (see the
    Stack/Concatenate ``_simplify_down`` rules)."""

    _parameters = ["func", "values", "chunks", "dtype", "_meta", "kwargs", "_name_prefix"]
    _defaults = {"dtype": None, "_meta": None, "kwargs": None, "_name_prefix": None}

    @functools.cached_property
    def _meta(self):
        meta = self.operand("_meta")
        dtype = self.operand("dtype")
        ndim = len(self.operand("chunks"))
        if meta is not None:
            if dtype is None:
                dtype = getattr(meta, "dtype", None)
            return meta_from_array(meta, dtype=dtype)
        if dtype is not None:
            return np.empty((0,) * ndim, dtype=dtype)
        return np.empty((0,) * ndim)

    @functools.cached_property
    def _name(self):
        prefix = self.operand("_name_prefix")
        if prefix:
            return prefix
        return "from-map-" + self.deterministic_token

    def _layer(self):
        func = self.operand("func")
        values = self.operand("values")
        kwargs = self.operand("kwargs") or {}
        chunks = self.chunks

        dsk = {}
        for idx in product(*(range(len(c)) for c in chunks)):
            key = (self._name,) + idx
            block_shape = tuple(chunks[d][idx[d]] for d in range(len(chunks)))
            dsk[key] = Task(key, _map_block, func, values[idx], block_shape, kwargs)
        return dsk


def from_map(func, values, chunks=None, dtype=None, meta=None, name=None, **kwargs):
    """Create a dask array by mapping ``func`` over a grid of per-block values.

    Each block of the result is ``func(values[idx], **kwargs)``, where ``idx`` is
    the block's position in the chunk grid. ``values`` is an array whose shape is
    the block grid (one cell per block); it is coerced to ``object`` dtype so each
    cell can hold an arbitrary Python argument.

    Parameters
    ----------
    func : callable
        Called once per block as ``func(values[idx], **kwargs)``. Should return an
        array whose shape matches that block's chunk shape.
    values : array-like
        Per-block arguments; ``values.shape`` must equal the number of blocks per
        dimension implied by ``chunks``. Coerced to ``object`` dtype. If each cell
        is itself a sequence (e.g. a tuple of args), build the object array
        explicitly (``np.empty(grid, dtype=object)`` then fill) -- ``np.asarray``
        would otherwise absorb the sequence into extra array dimensions.
    chunks : tuple of tuples of int
        The block structure, e.g. ``((5, 5, 5),)`` for a 1-D array of three
        length-5 blocks.
    dtype : np.dtype, optional
    meta : array-like, optional
    name : str, optional
        Output key prefix; defaults to a deterministic ``from-map-<token>``.
    **kwargs :
        Constant keyword arguments passed to ``func`` for every block.

    Examples
    --------
    >>> import numpy as np
    >>> import dask_array as da
    >>> values = np.array([1, 2, 3], dtype=object)
    >>> a = da.from_map(lambda v: np.full(5, v), values, chunks=((5, 5, 5),), dtype=int)
    >>> a.compute()
    array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])

    See Also
    --------
    dask.array.from_delayed
    """
    from dask_array._new_collection import new_collection

    if chunks is None:
        raise ValueError("from_map requires `chunks` (a tuple of tuples of block sizes)")

    values = np.asarray(values, dtype=object)
    chunks = tuple(tuple(int(b) for b in c) for c in chunks)
    numblocks = tuple(len(c) for c in chunks)
    if values.shape != numblocks:
        raise ValueError(f"values.shape {values.shape} must equal the block grid {numblocks} implied by chunks")

    return new_collection(FromMap(func, values, chunks, dtype, meta, kwargs or None, name))
