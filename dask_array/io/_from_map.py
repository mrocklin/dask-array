from __future__ import annotations

import functools
import logging
import pickle
from itertools import product

import cloudpickle
import numpy as np

from dask._task_spec import Task
from dask.hashing import hash_buffer_hex
from dask.tokenize import _tokenize_deterministic
from dask_array.io._base import IO
from dask_array._utils import meta_from_array

logger = logging.getLogger(__name__)


def _dumps5(obj):
    """One pickle of ``obj`` (protocol 5), for feeding a content hash.

    Mirrors dask's ``_normalize_pickle``: try the stdlib pickle first, and only
    fall back to cloudpickle when the payload references ``__main__`` (an
    interactively/script-defined object stdlib pickle can't reproduce across
    processes). Unlike ``normalize_token``, this does NOT round-trip
    (unpickle) the result as a determinism self-check -- callers must guard or
    restrict its use to operands they can accept as deterministic."""
    out = pickle.dumps(obj, protocol=5)
    if b"__main__" in out:
        out = cloudpickle.dumps(obj, protocol=5)
    return out


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


def _match_block_shape(x, shape):
    """Coerce a block func's return ``x`` to the declared chunk ``shape``. In the
    common case ``x`` already IS that shape and this is a no-op. The only reshape
    the rewrites need is inserting unit axes (a ``stack`` new axis, e.g.
    ``(5,) -> (1, 5)``), which never reorders data; anything else -- a
    same-size-but-permuted shape -- would silently corrupt the block, so it is
    rejected. Shared by both from_map block bodies (``_map_block`` and
    ``_apply_args_template``) so the contract can't drift between them."""
    shape = tuple(shape)
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


def _map_block(func, value, shape, kwargs):
    """Run one block: ``func(value, **kwargs)``, matched to the block's declared
    chunk ``shape``. Kept a single task so the block never splits into an
    ungrouped sub-task."""
    return _match_block_shape(func(value, **kwargs), shape)


def _apply_args_template(varying, shape, *, _fn, _fkwargs, _template, _positions):
    """Binary-path block body. Everything constant across the layer is baked into
    the shared partial (pickled once): the hoisted function ``_fn``, its shared
    ``_fkwargs``, and ``_template`` -- the full positional-arg list with the args
    that are UNIFORM across blocks already in place and the varying positions left
    as placeholders. ``varying`` carries this block's values for the varying
    positions (``_positions``, in order); splice them in and call ``_fn``, then
    match the declared chunk ``shape``. Only ``varying`` / ``shape`` differ per
    block, and both are binary-grammar values -- which is what keeps this layer
    pure-Rust even when some args (a storage_options dict, a config object) aren't
    themselves expressible in the grammar."""
    args = list(_template)
    for pos, val in zip(_positions, varying):
        args[pos] = val
    return _match_block_shape(_fn(*args, **(_fkwargs or {})), shape)


# Largest / smallest integer the binary grammar's i64 scalar/int-tuple slots hold.
_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1


def _encode_arg(a):
    """Classify one call argument into a ``(tag, payload)`` slot descriptor the
    Rust ``FromMapBinaryLayer`` slots directly, or ``None`` if it isn't a value the
    binary grammar can carry *without changing what the block function sees*.

    Only EXACT Python ``int`` / ``float`` / ``str`` (a filename), tuples of exact
    ``int`` (rebuilt as a tuple), and lists thereof (rebuilt as a list) are
    accepted. A numpy scalar -- or any ``int``/``float`` subclass -- is declined
    rather than coerced: the legacy ``_apply_call`` path passes the *original*
    object to the block function, so silently turning ``np.int64(5)`` into a plain
    ``int`` could change the function's result or the block's inferred dtype. Those
    args fall back to the faithful Python-tuple layer instead (``type(a) is int``
    also excludes ``bool``, which is an ``int`` subclass but semantically
    distinct)."""
    t = type(a)
    if t is str:
        return ("s", a)
    if t is int:
        return ("i", a) if _I64_MIN <= a <= _I64_MAX else None
    if t is float:
        return ("f", a)
    if t is tuple:
        if all(type(x) is int for x in a):
            return ("t", list(a)) if all(_I64_MIN <= x <= _I64_MAX for x in a) else None
        return None
    if t is list:
        inner = []
        for x in a:
            d = _encode_arg(x)
            if d is None:
                return None
            inner.append(d)
        return ("l", inner)
    return None


def _all_equal(col):
    """True iff every element of ``col`` equals the first -- used to decide which
    positional args are UNIFORM across blocks (bake into the shared partial) vs
    VARYING (slot per block). Safe for values whose ``==`` doesn't return a plain
    bool (ndarrays): identity short-circuits first, then a guarded ``==`` treats a
    non-bool or raising result as 'not equal', so the arg is handled as varying."""
    first = col[0]
    for x in col[1:]:
        if x is first:
            continue
        try:
            if not bool(x == first):
                return False
        except Exception:
            return False
    return True


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

    def __dask_tokenize__(self):
        """Name this FromMap by a content hash of its operands.

        The stock tokenize (``_tokenize_deterministic(type(self),
        *self.operands)``) pickle-*round-trips* every operand as a determinism
        self-check -- the dominant cost of naming a multi-million-task graph.
        Only the coalesced-from_delayed shape (``func is _apply_call``) tries
        the shortcut: its per-block values are ``(fn, args, kwargs)`` call
        bundles built by our own normalization. We accept the fast token only
        when two pickles produce identical bytes and the payload does not refer
        to ``__main__``; otherwise, fall through to stock tokenization. The
        ``"fm1"`` tag namespaces the scheme so it can't collide a stock 32-hex
        token, and a direct ``da.from_map`` (user ``func``, arbitrary per-block
        objects) keeps the stock round-trip its safety net is meant to police."""
        if not self._determ_token:
            if self.operand("func") is _apply_call:
                try:
                    payload = ("FromMap", type(self).__qualname__, *self.operands)
                    first = _dumps5(payload)
                    second = _dumps5(payload)
                    if first == second and b"__main__" not in first:
                        self._determ_token = "fm1" + hash_buffer_hex(first)
                        return self._determ_token
                except Exception:
                    pass  # fall through to the stock round-trip tokenize
            self._determ_token = _tokenize_deterministic(type(self), *self.operands)
        return self._determ_token

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

    def _binary_frisky_layer(self):
        """The pure-Rust ``FromMapBinaryLayer`` when this is a coalesced-delayed
        FromMap whose blocks share ONE function; otherwise ``None``.

        Only the coalesced-from_delayed shape qualifies: ``func is _apply_call``,
        so each per-block value is a ``(fn, args, kwargs)`` call bundle. We hoist
        everything constant across the layer -- the shared ``fn``, its ``kwargs``,
        and every positional arg that is UNIFORM across blocks -- into the shared
        partial (pickled once). Only the positional args that VARY per block go
        through the slot grammar, so they must be binary-expressible; an arg that
        both varies AND isn't expressible has nowhere to go -> decline. A direct
        ``da.from_map`` (user ``func``) stays on the general ``FromMapLayer``."""
        if self.operand("func") is not _apply_call or self.operand("kwargs"):
            return None
        values = self.operand("values")
        chunks = self.chunks
        idxs = list(product(*(range(len(c)) for c in chunks)))  # C order == Rust iteration
        bundles = [values[idx] for idx in idxs]
        if not bundles or not all(isinstance(b, tuple) and len(b) == 3 for b in bundles):
            return None
        fn, fkwargs = bundles[0][0], bundles[0][2]
        for f, _args, kw in bundles:
            if f is not fn:
                return None  # blocks call different functions -> can't hoist one
            try:
                if kw != fkwargs:
                    return None  # differing per-call kwargs across blocks
            except Exception:
                return None  # non-comparable kwargs (e.g. an ndarray value)
        # Positional args must share arity to classify uniform-vs-varying by position.
        arity = len(bundles[0][1])
        if any(len(b[1]) != arity for b in bundles):
            return None
        # Each position is UNIFORM (bake into the template, pickled once) or VARYING
        # (slot per block -- must be binary-expressible). ``template`` holds the
        # baked args in place with ``None`` placeholders at the varying positions.
        template = [None] * arity
        positions = []
        block_args = [[] for _ in bundles]
        for i in range(arity):
            col = [b[1][i] for b in bundles]
            if _all_equal(col):
                template[i] = col[0]  # uniform -> baked once
                continue
            descs = [_encode_arg(v) for v in col]
            if any(d is None for d in descs):
                return None  # varies AND not expressible -> decline
            positions.append(i)
            for slots, d in zip(block_args, descs):
                slots.append(d)

        from dask_array._frisky.from_map import FromMapBinaryLayer

        return FromMapBinaryLayer(
            self._name,
            _apply_args_template,
            {"_fn": fn, "_fkwargs": dict(fkwargs), "_template": template, "_positions": positions},
            block_args,
            [list(c) for c in chunks],
        )

    def _frisky_layer(self):
        """Native records layer. Prefer the pure-Rust ``FromMapBinaryLayer`` (one
        shared function + binary-expressible per-block args -> a binary
        ``to_records_chunk``); otherwise the general ``FromMapLayer``, whose block
        values are arbitrary Python objects and so ship as plain task records.

        Values are flattened C-order to match the Rust layer's row-major block
        iteration (and ``_layer``'s ``itertools.product``)."""
        binary = self._binary_frisky_layer()
        if binary is not None:
            return binary
        if self.operand("func") is _apply_call:
            # A coalesced-from_delayed FromMap that could have been pure-Rust but
            # whose blocks don't share one function with binary-only args. Correct
            # (plain records below), but flag the missed fast path.
            logger.warning(
                "FromMap %s stays on Python-tuple records: its blocks don't share "
                "one function with binary-expressible args (scalars/strings/"
                "int-tuples), so Frisky ships one Python task per block.",
                self._name,
            )
        from dask_array._frisky.from_map import FromMapLayer

        return FromMapLayer(
            self._name,
            _map_block,
            self.operand("func"),
            self.operand("kwargs") or {},
            list(self.operand("values").ravel(order="C")),
            [list(c) for c in self.chunks],
        )


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
