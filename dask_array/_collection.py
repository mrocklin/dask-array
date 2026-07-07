from __future__ import annotations

import math
import operator
import warnings
import weakref
from functools import cached_property

import numpy as np

from dask import config
from dask_array import _chunk as chunk
from dask_array._new_collection import new_collection
from dask_array._expr import ArrayExpr, RootAlias
from dask_array.manipulation._transpose import Transpose
from dask_array._chunk_types import is_valid_chunk_type
from dask.base import DaskMethodsMixin, is_dask_collection, named_schedulers
from dask.utils import format_bytes, has_keyword, typename

from dask_array._templates import get_template

try:
    ARRAY_TEMPLATE = get_template("array.html.j2")
except ImportError:
    ARRAY_TEMPLATE = None

# Import blockwise functions from their module
# Import broadcast
from dask_array._broadcast import broadcast_to

# Import concatenate and stacking
from dask_array._concatenate import concatenate
from dask_array._stack import stack
from dask_array.core._blockwise_funcs import blockwise, elemwise

# Import core conversion functions from their module
from dask_array.core._conversion import (
    array,
    asanyarray,
    asarray,
    from_array,
)
from dask_array.core._from_graph import from_graph

# Import manipulation functions
from dask_array.manipulation._expand import (
    atleast_1d,
    atleast_2d,
    atleast_3d,
    expand_dims,
)
from dask_array.manipulation._flip import flip, fliplr, flipud, rot90
from dask_array.manipulation._roll import roll
from dask_array.manipulation._transpose import (
    moveaxis,
    rollaxis,
    transpose,
)
from dask_array.stacking._block import block
from dask_array.stacking._simple import dstack, hstack, vstack

# Type imports
from dask_array._core_utils import (
    _HANDLED_FUNCTIONS,
    T_IntOrNaN,
    _get_chunk_shape,
    _should_delegate,
    cached_max,
    check_if_handled_given_other,
    finalize,
)

__all__ = [
    "Array",
    "array",
    "asanyarray",
    "asarray",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "block",
    "blockwise",
    "broadcast_to",
    "concatenate",
    "dstack",
    "elemwise",
    "expand_dims",
    "flip",
    "fliplr",
    "flipud",
    "from_array",
    "from_graph",
    "hstack",
    "moveaxis",
    "ravel",
    "rechunk",
    "reshape",
    "reshape_blockwise",
    "roll",
    "rollaxis",
    "rot90",
    "squeeze",
    "stack",
    "swapaxes",
    "transpose",
    "vstack",
]


# Process-wide, ``_name``-keyed cache of one-pass lowering results, shared across
# every ``_lower`` call.  Lowering is a deterministic, context-free function of a
# node's structure (captured by its ``_name``), so memoizing by ``_name`` is safe.
# This relies on every ``_lower`` override depending only on ``self`` — not on
# config, parent context, or randomness (unlike ``_simplify_up``, or the ``_layer``
# methods that legitimately read config at graph-build time).  A ``_lower`` that
# read config would make this cache serve stale results across config changes.
# The payoff is cross-collection sharing: the 666 quantities of one model Dataset
# share a deep ancestry, and without this each collection re-lowers (and so
# re-tokenizes) that shared subtree from scratch — O(N^2) over the Dataset.  With
# the shared cache a shared subtree lowers once.  Weak values bound the cache to
# lowered expressions that are still live (each is reachable from some collection's
# ``_lowered_expr``), so it self-evicts as collections are dropped.
_LOWER_CACHE: weakref.WeakValueDictionary[str, ArrayExpr] = weakref.WeakValueDictionary()


def _lower(expr, optimize_graph):
    """Simplify (when optimizing) and lower ``expr`` to a concrete form.

    Equivalent to ``expr.simplify().lower_completely()`` but lowering goes
    through the shared ``_LOWER_CACHE`` (see above), so subtrees shared by
    many collections lower once.
    """
    if optimize_graph:
        expr = expr.simplify()
    while True:
        new = expr.lower_once(_LOWER_CACHE)
        if new._name == expr._name:
            return expr
        expr = new


def _materialize(expr, optimize_graph=None):
    """Optimize an expression fully (simplify → lower → fuse) and pin its
    output keys back to the raw root name.

    This is the single place where an expression becomes a task graph — used
    by ``__dask_graph__``, ``__dask_keys__``'s contract, and the Frisky
    records walk. Optimization renames every node it rewrites, but a
    collection's advertised keys are its *raw* root expression's name
    (``Array._name``), assigned at construction and stable forever. So when
    optimization renamed the root, the result is wrapped in a ``RootAlias``
    whose layer aliases ``(raw name, i, ...)`` to the optimized root's keys.
    Results therefore always come back under the keys that were advertised —
    ``persist`` round-trips without any name reconciliation.

    When ``array.optimize-graph`` is True (the default) we ``simplify()`` and
    ``fuse()``; set it False to observe the raw, un-simplified graph (e.g.
    structural tests asserting on task counts). Lowering always runs (a graph
    cannot be built from un-lowered nodes) and goes through the shared
    ``_LOWER_CACHE`` (see above) so a subtree shared by many collections is
    lowered once rather than once per collection.
    """
    if optimize_graph is None:
        optimize_graph = config.get("array.optimize-graph", True)
    if isinstance(expr, RootAlias):
        return expr  # only ever built here, over an already-materialized tree
    name = expr._name
    chunks = expr.chunks

    expr = _lower(expr, optimize_graph)
    if optimize_graph:
        expr = expr.fuse()
    if expr._name != name:
        if not _chunks_match(expr.chunks, chunks):
            # A rewrite chose a different output chunking (e.g. the
            # sliding-window reductions avoid a padding rechunk by emitting
            # coarser blocks). The collection's advertised chunks are a
            # promise, so bridge back to them. Cheap when the rewrite's grid
            # is coarser (pure splits); the real fix is for such rewrites to
            # advertise their chunking at construction time.
            if any(math.isnan(s) for dim in chunks for s in dim):
                raise RuntimeError(
                    f"optimization changed the block structure of {name} "
                    f"({chunks} -> {expr.chunks}) and the advertised chunks "
                    "are unknown, so they cannot be restored"
                )
            expr = _lower(expr.rechunk(chunks), optimize_graph=False)
        if any(node._name == name for node in expr.walk()):
            # The pin's alias keys would collide with that node's own layer.
            # No rewrite embeds its input root today; fail loudly if one ever
            # does rather than silently merging two layers under one name.
            raise RuntimeError(
                f"optimization embedded the original root {name} inside its rewrite; cannot pin output keys"
            )
        expr = RootAlias(expr, name)
    return expr


def _chunks_match(a, b):
    """Chunk equality, treating unknown (nan) sizes as matching."""
    if len(a) != len(b):
        return False
    return all(
        len(da) == len(db) and all(sa == sb or (math.isnan(sa) and math.isnan(sb)) for sa, sb in zip(da, db))
        for da, db in zip(a, b)
    )


class Array(DaskMethodsMixin):
    __dask_scheduler__ = staticmethod(named_schedulers.get("threads", named_schedulers["sync"]))
    __dask_optimize__ = staticmethod(lambda dsk, keys, **kwargs: dsk)
    __array_priority__ = 11  # higher than numpy.ndarray and numpy.matrix

    def __init__(self, expr):
        self._expr = expr

    def __getstate__(self):
        state = self.__dict__.copy()
        # Derived caches: _lowered_expr can be huge (a materialized graph) and
        # _cached_dask_keys holds one tuple per block. Both rebuild cheaply.
        state.pop("_lowered_expr", None)
        state.pop("_cached_dask_keys", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def expr(self) -> ArrayExpr:
        return self._expr

    def _replace_expr(self, expr):
        """Swap this collection's expression in place (``out=``, in-place ops,
        ``compute_chunk_sizes``). Name, keys, and graph all derive from the
        expression, so cached derivations must go with it."""
        self._expr = expr
        for cached in ("_lowered_expr", "_lowered_expr_optimize_graph", "_cached_dask_keys"):
            self.__dict__.pop(cached, None)

    @cached_property
    def _lowered_expr_optimize_graph(self):
        return config.get("array.optimize-graph", True)

    @cached_property
    def _lowered_expr(self):
        return _materialize(self.expr, optimize_graph=self._lowered_expr_optimize_graph)

    @property
    def _name(self):
        # The raw root expression's name, assigned at construction. Stable:
        # optimization renames internal nodes freely, but materialization pins
        # the graph's output keys back to this name (see ``_materialize``).
        # Never triggers lowering — keep it that way; a ``.name`` access that
        # lowers makes construction loops O(tree^2).
        return self.expr._name

    def __dask_postcompute__(self):
        return finalize, ()

    def __dask_postpersist__(self):
        # Persist is name-preserving: the rebuilt collection keeps this
        # collection's name and keys. The rebuild layer's keys per block are
        # found at rebuild time (see ``FromGraph._layer``): our own keys when
        # the layer came from a pinned graph (``Array.persist``, frisky,
        # ``dask.optimize`` via the pinned ``ArrayExpr.__dask_graph__``), or
        # located by block id when a raw expression went through dask's
        # generic optimizer (``dask.persist``). No keys are predicted here —
        # predicting meant lowering, and stale predictions were the old bug
        # class.
        meta = self._meta
        if meta is None:
            # Fallback to synthetic meta if original is also None
            meta = np.empty((0,) * self.ndim, dtype=self.dtype)
        # Use self.chunks to preserve nan chunks for unknown-sized operations
        return from_graph, (
            meta,
            self.chunks,
            [],
            self._name,
        )

    @property
    def dask(self):
        return self.__dask_graph__()

    def __dask_graph__(self):
        from dask._expr import Expr

        out = self._lowered_expr
        return Expr.__dask_graph__(out)

    @cached_property
    def _cached_dask_keys(self):
        # Built from the raw name and block structure only — no lowering.
        # ``_materialize`` guarantees the graph produces these keys.
        name, chunks, numblocks = self._name, self.chunks, self.numblocks

        def keys(*args):
            if not chunks:
                return [(name,)]
            ind = len(args)
            if ind + 1 == len(numblocks):
                return [(name,) + args + (i,) for i in range(numblocks[ind])]
            return [keys(*(args + (i,))) for i in range(numblocks[ind])]

        return keys()

    def __dask_keys__(self):
        return self._cached_dask_keys

    def _check_frisky_supported(self):
        if isinstance(self._meta, np.ma.MaskedArray):
            raise NotImplementedError("masked arrays use the regular dask graph path")
        # Unknown (nan) chunk *sizes* are fine on the records path: records are
        # keyed by block coordinate, and numblocks is known even when sizes are
        # not, so the graph is fully static (boolean masks, unique, argwhere).
        # Only tasks that need concrete sizes fail, and those decline per-layer.
        #
        # We previously also bailed the whole submission when a ReshapeLowered
        # sat under a CumReduction ("flattened cumulative reductions"), out of a
        # concern that the reshape-backed sequential-scan graph could not be
        # ordered safely on the records path.  Direct testing showed the records
        # generate cleanly (no dangling deps) and both the records and
        # expression paths compute the correct, deterministic result for the
        # axis=None flatten and reshape-fed EW-scan shapes -- the guard only
        # forced correct graphs down the slow materialized-graph path.  The
        # reshape and cumulative are ordinary producer/consumer layers, so let
        # them ride the records path.

    def __frisky_graph__(self, seen=None):
        """Frisky submission protocol (duck-typed; Frisky never imports
        dask_array). Returns a flat Frisky graph as ``(key, func, args, kwargs,
        deps)`` task records, or raises ``NotImplementedError`` if the graph
        can't be represented (so the caller falls back to the materialized-graph
        path).

        ``seen`` is an optional set of already-walked expr ``_name``s, threaded
        by Frisky across the collections of one ``dask.compute(x, y)`` so a
        shared subgraph is expanded once; in that mode completeness is checked by
        the caller over the combined records (see ``collect_task_records``)."""
        from dask_array._frisky.collect import collect_task_records

        self._check_frisky_supported()
        return collect_task_records(self, seen=seen)

    def __frisky_records_chunks__(self, seen=None):
        """Frisky binary-records protocol (duck-typed). Returns
        ``(chunks, records, chunk_groups)``: a list of binary records LAYER chunks
        (bytes) for the layers that support the fast Rust-to-Rust path, plus plain
        ``(key, func, args, kwargs, deps)`` records for the layers that don't
        (from_array source, generic fallback), plus ``chunk_groups`` — parallel to
        ``chunks``, each the producing expr's ``(_name, metadata_json)`` so Frisky
        groups a layer's tasks by their true identity and can display the layer's
        shape/chunks/dtype. Frisky decodes both task sources and unions them under
        one ``dask.order`` pass. Raises ``NotImplementedError`` if the graph can't
        be represented at all (caller falls back to ``__frisky_graph__`` or the
        materialized-graph path). ``seen`` threads like ``__frisky_graph__``."""
        from dask_array._frisky.collect import collect_record_chunks

        self._check_frisky_supported()
        return collect_record_chunks(self, seen=seen)

    def __frisky_output_keys__(self):
        """Frisky expression-submission protocol (duck-typed). The flat,
        stringified output keys this collection wants results for. Both the
        client (to build futures) and the scheduler (to register the client's
        desire when it expands the submitted expression) derive these from the
        small expression — they are never sent over the wire. Matches the
        records path's output-key derivation exactly, so the keys line up with
        the record graph the expander produces."""
        from dask.core import flatten

        self._check_frisky_supported()
        return list(dict.fromkeys(str(k) for k in flatten(self.__dask_keys__())))

    def __dask_tokenize__(self):
        return "Array", self._name

    def compute(self, **kwargs):
        return DaskMethodsMixin.compute(self._pinned(), **kwargs)

    def persist(self, **kwargs):
        # The pinned collection's keys are this collection's keys, so the
        # persisted result keeps our name: x.persist().name == x.name.
        return DaskMethodsMixin.persist(self._pinned(), **kwargs)

    def _pinned(self):
        """This collection over its materialized (optimized + key-pinned) expression.

        ``dask.base`` re-derives keys by running its generic optimizer over
        ``collection.expr`` — handing it the raw expression means unfused
        graphs and keys that drift from ours. The materialized expression is
        already fused and pins its output keys to this collection's name, so
        it passes through dask's optimizer unchanged: what gets scheduled is
        exactly ``__dask_graph__``/``__dask_keys__``, on every entry point.
        """
        return new_collection(self._lowered_expr)

    def optimize(self):
        if self.__dict__.get("_optimized", False):
            return self
        expr = _lower(self.expr, optimize_graph=True).fuse()
        out = new_collection(expr)
        out.__dict__["_optimized"] = True
        out.__dict__["_lowered_expr_optimize_graph"] = True
        out.__dict__["_lowered_expr"] = expr
        return out

    def simplify(self):
        return new_collection(self.expr.simplify())

    def pprint(self):
        """Pretty print the expression tree.

        Uses rich table format if rich is installed, otherwise falls back
        to the basic tree representation.
        """
        self.expr.pprint()

    def visualize(self, tasks: bool = False, **kwargs):  # type: ignore[override]
        """Visualize the expression or task graph.

        Parameters
        ----------
        tasks : bool
            Whether to visualize the task graph. By default
            the expression graph will be visualized instead.
        **kwargs
            Additional arguments passed to the visualizer.
        """
        # color= options require task graph visualization
        if tasks or kwargs.get("color"):
            return super().visualize(**kwargs)
        return self.expr.visualize(**kwargs)

    @property
    def _meta(self):
        return self.expr._meta

    @property
    def dtype(self):
        return self.expr.dtype

    @property
    def shape(self):
        return self.expr.shape

    @property
    def chunks(self):
        return self.expr.chunks

    @chunks.setter
    def chunks(self, chunks):
        raise TypeError(
            "Can not set chunks directly\n\n"
            "Please use the rechunk method instead:\n"
            f"  x.rechunk({chunks})\n\n"
            "Documentation\n"
            "-------------\n"
            "https://docs.dask.org/en/latest/generated/dask.array.rechunk.html"
        )

    @property
    def _chunks(self):
        """Internal access to chunks (for compatibility with tests)."""
        return self.expr.chunks

    @_chunks.setter
    def _chunks(self, chunks):
        """Set chunks by wrapping the expression with ChunksOverride.

        This is primarily for internal use and testing when simulating
        arrays with unknown chunk sizes.
        """
        from dask_array._expr import ChunksOverride

        self._replace_expr(ChunksOverride(self._expr, chunks))

    @property
    def chunksize(self) -> tuple:
        return tuple(cached_max(c) for c in self.chunks)

    @property
    def ndim(self):
        return self.expr.ndim

    @property
    def numblocks(self):
        return self.expr.numblocks

    @property
    def npartitions(self):
        from math import prod

        return prod(self.numblocks)

    def compute_chunk_sizes(self):
        """
        Compute the chunk sizes for a Dask array. This is especially useful
        when the chunk sizes are unknown (e.g., when indexing one Dask array
        with another).

        Notes
        -----
        This function modifies the Dask array in-place.

        Examples
        --------
        >>> import dask_array as da
        >>> import numpy as np
        >>> x = da.from_array([-2, -1, 0, 1, 2], chunks=2)
        >>> x.chunks
        ((2, 2, 1),)
        >>> y = x[x <= 0]
        >>> y.chunks
        ((nan, nan, nan),)
        >>> y.compute_chunk_sizes()  # in-place computation
        dask.array<getitem, shape=(3,), dtype=int64, chunksize=(2,), chunktype=numpy.ndarray>
        >>> y.chunks
        ((2, 1, 0),)
        """
        from dask.base import compute

        chunk_shapes = self.map_blocks(
            _get_chunk_shape,
            dtype=int,
            chunks=tuple(len(c) * (1,) for c in self.chunks) + ((self.ndim,),),
            new_axis=self.ndim,
        )

        c = []
        for i in range(self.ndim):
            s = self.ndim * [0] + [i]
            s[i] = slice(None)
            s = tuple(s)

            c.append(tuple(chunk_shapes[s]))

        # `map_blocks` assigns numpy dtypes
        # cast chunk dimensions back to python int before returning
        new_chunks = tuple(tuple(int(chunk) for chunk in chunks) for chunks in compute(tuple(c))[0])

        # In the expression system, wrap with ChunksOverride to set the new chunks
        from dask_array._expr import ChunksOverride

        self._replace_expr(ChunksOverride(self._expr, new_chunks))

        return self

    @property
    def _key_array(self):
        return np.array(self.__dask_keys__(), dtype=object)

    @property
    def blocks(self):
        from dask_array.slicing._blocks import BlockView

        return BlockView(self)

    @property
    def partitions(self):
        """Slice an array by partitions. Alias of dask array .blocks attribute."""
        return self.blocks

    @property
    def size(self) -> T_IntOrNaN:
        return self.expr.size

    @property
    def nbytes(self) -> T_IntOrNaN:
        """Number of bytes in array"""
        return self.size * self.dtype.itemsize

    @property
    def itemsize(self) -> int:
        """Length of one array element in bytes"""
        return self.dtype.itemsize

    @property
    def transfer_bytes(self):
        """Estimated (min, max) inter-worker bytes moved by this node.

        See ArrayExpr.transfer_bytes.  Counts only the root expression's
        incoming edges; walk the optimized expression graph and sum for a
        whole-computation estimate.
        """
        return self.expr.transfer_bytes

    @property
    def name(self):
        return self._name

    def __len__(self):
        return self.expr.__len__()

    def __repr__(self):
        name = self.name.rsplit("-", 1)[0]
        return "dask.array<{}, shape={}, dtype={}, chunksize={}, chunktype={}.{}>".format(
            name,
            self.shape,
            self.dtype,
            self.chunksize,
            type(self._meta).__module__.split(".")[0],
            type(self._meta).__name__,
        )

    def _repr_html_(self):
        if ARRAY_TEMPLATE is None:
            return repr(self)

        try:
            grid = self.to_svg(size=config.get("array.svg.size", 120))
        except NotImplementedError:
            grid = ""

        if "sparse" in typename(type(self._meta)):
            nbytes = None
            cbytes = None
        elif not math.isnan(self.nbytes):
            nbytes = format_bytes(self.nbytes)
            cbytes = format_bytes(math.prod(self.chunksize) * self.dtype.itemsize)
        else:
            nbytes = "unknown"
            cbytes = "unknown"

        # Expression flow summary and diagram
        from dask_array._expr_flow import build_flow_graph, render_flow_svg, count_operations

        try:
            nodes, edges = build_flow_graph(self._expr)
            n_expr = count_operations(self._expr)
            expr_flow = render_flow_svg(self._expr) if nodes else ""
        except Exception:
            n_expr = 1
            expr_flow = ""

        return ARRAY_TEMPLATE.render(
            array=self,
            grid=grid,
            nbytes=nbytes,
            cbytes=cbytes,
            n_expr=n_expr,
            expr_flow=expr_flow,
        )

    def __bool__(self):
        if self.size > 1:
            raise ValueError(f"The truth value of a {self.__class__.__name__} is ambiguous. Use a.any() or a.all().")
        return bool(self.compute())

    def _scalarfunc(self, cast_type):
        if self.size > 1:
            raise TypeError("Only length-1 arrays can be converted to Python scalars")
        else:
            return cast_type(self.compute().item())

    def __int__(self):
        return self._scalarfunc(int)

    def __float__(self):
        return self._scalarfunc(float)

    def __complex__(self):
        return self._scalarfunc(complex)

    def __index__(self):
        return self._scalarfunc(operator.index)

    def __array__(self, dtype=None, copy=None, **kwargs):
        import warnings

        if kwargs:
            warnings.warn(
                f"Extra keyword arguments {kwargs} are ignored and won't be accepted in the future",
                FutureWarning,
            )
        if copy is False:
            warnings.warn(
                "Can't acquire a memory view of a Dask array. This will raise in the future.",
                FutureWarning,
            )
        x = self.compute()
        return np.asarray(x, dtype=dtype)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        # Field access, e.g. x['a'] or x[['a', 'b']]
        if isinstance(index, str) or (isinstance(index, list) and index and all(isinstance(i, str) for i in index)):
            from dask_array._chunk import getitem

            if isinstance(index, str):
                dt = self.dtype[index]
            else:
                dt = np.dtype(
                    {
                        "names": index,
                        "formats": [self.dtype.fields[name][0] for name in index],
                        "offsets": [self.dtype.fields[name][1] for name in index],
                        "itemsize": self.dtype.itemsize,
                    }
                )

            if dt.shape:
                new_axis = list(range(self.ndim, self.ndim + len(dt.shape)))
                chunks = self.chunks + tuple((i,) for i in dt.shape)
                return self.map_blocks(getitem, index, dtype=dt.base, chunks=chunks, new_axis=new_axis)
            else:
                return self.map_blocks(getitem, index, dtype=dt)

        if not isinstance(index, tuple):
            index = (index,)

        from dask_array.slicing import (
            normalize_index,
            slice_array,
            slice_with_int_dask_array,
        )

        index2 = normalize_index(index, self.shape)

        if any(isinstance(i, Array) and i.dtype.kind in "iu" for i in index2):
            self, index2 = slice_with_int_dask_array(self, index2)
        if any(isinstance(i, Array) and i.dtype == bool for i in index2):
            from dask_array.slicing import slice_with_bool_dask_array

            self, index2 = slice_with_bool_dask_array(self, index2)

        if all(isinstance(i, slice) and i == slice(None) for i in index2):
            return self

        result = slice_array(self.expr, index2)
        return new_collection(result)

    def __setitem__(self, key, value):
        from dask_array._core_utils import unknown_chunk_message

        # Handle np.ma.masked assignment
        if value is np.ma.masked:
            value = np.ma.masked_all((), dtype=self.dtype)

        # Check for NaN/inf in integer arrays
        if not is_dask_collection(value) and self.dtype.kind in "iu":
            if np.isnan(value).any():
                raise ValueError("cannot convert float NaN to integer")
            if np.isinf(value).any():
                raise ValueError("cannot convert float infinity to integer")

        # Suppress dtype broadcasting; __setitem__ can't change the dtype.
        value = asanyarray(value, dtype=self.dtype)

        # Handle 1D integer array index case
        if isinstance(key, Array) and (
            key.dtype.kind in "iu" or (key.dtype == bool and key.ndim == 1 and self.ndim > 1)
        ):
            key = (key,)

        # Use "where" method for any dask Array key (matches legacy behavior)
        if isinstance(key, Array):
            from dask_array._broadcast import broadcast_to

            left_shape = np.array(key.shape)
            right_shape = np.array(self.shape)

            # Treat unknown shapes as matching
            match = left_shape == right_shape
            match |= np.isnan(left_shape) | np.isnan(right_shape)

            if not match.all():
                raise IndexError(f"boolean index shape {key.shape} must match indexed array's {self.shape}.")

            # If value has ndim > 0, they must be broadcastable to self.shape[idx].
            if value.ndim:
                value = broadcast_to(value, self[key].shape)

            from dask_array.routines._where import where

            y = where(key, value, self)
            self._replace_expr(y.expr)
            return

        # Check for unknown chunks
        if np.isnan(self.shape).any():
            raise ValueError(f"Arrays chunk sizes are unknown. {unknown_chunk_message}")

        # Validate indices and value shape eagerly (before creating lazy expression)
        import math

        from dask_array.slicing._utils import parse_assignment_indices

        indices, implied_shape, _, implied_shape_positions = parse_assignment_indices(key, self.shape)
        value_shape = value.shape

        # Validate value shape vs implied shape (from setitem_array validation)
        if 0 in implied_shape and value_shape and max(value_shape) > 1:
            raise ValueError(
                f"shape mismatch: value array of shape {value_shape} "
                "could not be broadcast to indexing result "
                f"of shape {tuple(implied_shape)}"
            )

        value_ndim = len(value_shape)
        offset = len(implied_shape) - value_ndim
        if offset >= 0:
            array_common_shape = implied_shape[offset:]
            value_common_shape = value_shape
            implied_positions = implied_shape_positions[offset:]
        else:
            value_offset = -offset
            array_common_shape = implied_shape
            value_common_shape = value_shape[value_offset:]
            implied_positions = implied_shape_positions
            # All extra leading dimensions must have size 1
            if value_shape[:value_offset] != (1,) * value_offset:
                raise ValueError(
                    f"could not broadcast input array from shape{value_shape} into shape {tuple(implied_shape)}"
                )

        # Check shape compatibility for each dimension
        for _, (a, b, j) in enumerate(zip(array_common_shape, value_common_shape, implied_positions)):
            index = indices[j]
            if is_dask_collection(index) and getattr(index, "dtype", None) == bool:
                # For dask boolean index, value size must not exceed index size
                if not math.isnan(b) and b > index.size:
                    raise ValueError(
                        f"shape mismatch: value array dimension size of {b} is "
                        "greater then corresponding boolean index size of "
                        f"{index.size}"
                    )
            elif b != 1 and a != b and not math.isnan(a):
                raise ValueError(
                    f"shape mismatch: value array of shape {value_shape} "
                    "could not be broadcast to indexing result of shape "
                    f"{tuple(implied_shape)}"
                )

        # Use SetItem expression for other index types
        from dask_array.slicing import SetItem

        value_expr = value.expr if isinstance(value, Array) else value
        y = new_collection(SetItem(self.expr, key, value_expr))
        self._replace_expr(y.expr)

    @check_if_handled_given_other
    def __add__(self, other):
        return elemwise(operator.add, self, other)

    @check_if_handled_given_other
    def __radd__(self, other):
        return elemwise(operator.add, other, self)

    @check_if_handled_given_other
    def __mul__(self, other):
        return elemwise(operator.mul, self, other)

    @check_if_handled_given_other
    def __rmul__(self, other):
        return elemwise(operator.mul, other, self)

    @check_if_handled_given_other
    def __sub__(self, other):
        return elemwise(operator.sub, self, other)

    @check_if_handled_given_other
    def __rsub__(self, other):
        return elemwise(operator.sub, other, self)

    @check_if_handled_given_other
    def __pow__(self, other):
        return elemwise(operator.pow, self, other)

    @check_if_handled_given_other
    def __rpow__(self, other):
        return elemwise(operator.pow, other, self)

    @check_if_handled_given_other
    def __truediv__(self, other):
        return elemwise(operator.truediv, self, other)

    @check_if_handled_given_other
    def __rtruediv__(self, other):
        return elemwise(operator.truediv, other, self)

    @check_if_handled_given_other
    def __floordiv__(self, other):
        return elemwise(operator.floordiv, self, other)

    @check_if_handled_given_other
    def __rfloordiv__(self, other):
        return elemwise(operator.floordiv, other, self)

    def __abs__(self):
        return elemwise(operator.abs, self)

    @check_if_handled_given_other
    def __and__(self, other):
        return elemwise(operator.and_, self, other)

    @check_if_handled_given_other
    def __rand__(self, other):
        return elemwise(operator.and_, other, self)

    @check_if_handled_given_other
    def __div__(self, other):
        return elemwise(operator.div, self, other)

    @check_if_handled_given_other
    def __rdiv__(self, other):
        return elemwise(operator.div, other, self)

    @check_if_handled_given_other
    def __eq__(self, other):
        return elemwise(operator.eq, self, other)

    @check_if_handled_given_other
    def __gt__(self, other):
        return elemwise(operator.gt, self, other)

    @check_if_handled_given_other
    def __ge__(self, other):
        return elemwise(operator.ge, self, other)

    def __invert__(self):
        return elemwise(operator.invert, self)

    @check_if_handled_given_other
    def __lshift__(self, other):
        return elemwise(operator.lshift, self, other)

    @check_if_handled_given_other
    def __rlshift__(self, other):
        return elemwise(operator.lshift, other, self)

    @check_if_handled_given_other
    def __lt__(self, other):
        return elemwise(operator.lt, self, other)

    @check_if_handled_given_other
    def __le__(self, other):
        return elemwise(operator.le, self, other)

    @check_if_handled_given_other
    def __mod__(self, other):
        return elemwise(operator.mod, self, other)

    @check_if_handled_given_other
    def __rmod__(self, other):
        return elemwise(operator.mod, other, self)

    @check_if_handled_given_other
    def __ne__(self, other):
        return elemwise(operator.ne, self, other)

    def __neg__(self):
        return elemwise(operator.neg, self)

    @check_if_handled_given_other
    def __or__(self, other):
        return elemwise(operator.or_, self, other)

    def __pos__(self):
        return self

    @check_if_handled_given_other
    def __ror__(self, other):
        return elemwise(operator.or_, other, self)

    @check_if_handled_given_other
    def __rshift__(self, other):
        return elemwise(operator.rshift, self, other)

    @check_if_handled_given_other
    def __rrshift__(self, other):
        return elemwise(operator.rshift, other, self)

    @check_if_handled_given_other
    def __xor__(self, other):
        return elemwise(operator.xor, self, other)

    @check_if_handled_given_other
    def __rxor__(self, other):
        return elemwise(operator.xor, other, self)

    @check_if_handled_given_other
    def __matmul__(self, other):
        from dask_array.linalg import matmul

        return matmul(self, other)

    @check_if_handled_given_other
    def __rmatmul__(self, other):
        from dask_array.linalg import matmul

        return matmul(other, self)

    @check_if_handled_given_other
    def __divmod__(self, other):
        from dask_array._ufunc import divmod

        return divmod(self, other)

    @check_if_handled_given_other
    def __rdivmod__(self, other):
        from dask_array._ufunc import divmod

        return divmod(other, self)

    def __array_function__(self, func, types, args, kwargs):
        import dask_array as module
        from dask.base import compute

        def handle_nonmatching_names(func, args, kwargs):
            if func not in _HANDLED_FUNCTIONS:
                warnings.warn(
                    f"The `{func.__module__}.{func.__name__}` function "
                    "is not implemented by Dask array. "
                    "You may want to use the da.map_blocks function "
                    "or something similar to silence this warning. "
                    "Your code may stop working in a future release.",
                    FutureWarning,
                )
                # Need to convert to array object (e.g. numpy.ndarray or
                # cupy.ndarray) as needed, so we can call the NumPy function
                # again and it gets the chance to dispatch to the right
                # implementation.
                args, kwargs = compute(args, kwargs)
                return func(*args, **kwargs)

            return _HANDLED_FUNCTIONS[func](*args, **kwargs)

        # First, verify that all types are handled by Dask. Otherwise, return NotImplemented.
        if not all(
            # Accept our own superclasses as recommended by NEP-13
            # (https://numpy.org/neps/nep-0013-ufunc-overrides.html#subclass-hierarchies)
            issubclass(type(self), type_) or is_valid_chunk_type(type_)
            for type_ in types
        ):
            return NotImplemented

        # Now try to find a matching function name.  If that doesn't work, we may
        # be dealing with an alias or a function that's simply not in the Dask API.
        # Handle aliases via the _HANDLED_FUNCTIONS dict mapping, and warn otherwise.
        for submodule in func.__module__.split(".")[1:]:
            try:
                module = getattr(module, submodule)
            except AttributeError:
                return handle_nonmatching_names(func, args, kwargs)

        if not hasattr(module, func.__name__):
            return handle_nonmatching_names(func, args, kwargs)

        da_func = getattr(module, func.__name__)
        if da_func is func:
            return handle_nonmatching_names(func, args, kwargs)

        # If ``like`` is contained in ``da_func``'s signature, add ``like=self``
        # to the kwargs dictionary.
        if has_keyword(da_func, "like"):
            kwargs["like"] = self

        return da_func(*args, **kwargs)

    def transpose(self, *axes):
        from collections.abc import Iterable

        if not axes:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], Iterable):
            axes = axes[0]

        if axes:
            if len(axes) != self.ndim:
                raise ValueError("axes don't match array")
            axes = tuple(d + self.ndim if d < 0 else d for d in axes)
        else:
            axes = tuple(range(self.ndim))[::-1]

        # Identity transpose - return self
        if axes == tuple(range(self.ndim)):
            return self

        return new_collection(Transpose(self, axes))

    @property
    def T(self):
        return self.transpose()

    @property
    def A(self):
        return self

    def swapaxes(self, axis1, axis2):
        """Interchange two axes of an array.

        Refer to :func:`dask.array.swapaxes` for full documentation.
        """
        return swapaxes(self, axis1, axis2)

    def squeeze(self, axis=None):
        """Remove axes of length one from array.

        Refer to :func:`dask.array.squeeze` for full documentation.
        """
        return squeeze(self, axis=axis)

    def reshape(self, *shape, merge_chunks=True, limit=None, order=None):
        """Reshape the array to a new shape.

        Refer to :func:`dask.array.reshape` for full documentation.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        if order not in (None, "C"):
            raise NotImplementedError("dask_array.reshape only supports C-order reshaping")
        return reshape(self, shape, merge_chunks=merge_chunks, limit=limit)

    def flatten(self):
        """Return a copy of the array collapsed into one dimension.

        Returns
        -------
        dask Array
            A 1-D array with the same data as self.
        """
        return reshape(self, (-1,))

    def ravel(self):
        """Return a flattened array.

        Returns
        -------
        dask Array
            A 1-D array with the same data as self.
        """
        return reshape(self, (-1,))

    def repeat(self, repeats, axis=None):
        """Repeat elements of an array.

        Refer to :func:`dask.array.repeat` for full documentation.
        """
        from dask_array.creation import repeat

        return repeat(self, repeats, axis)

    def choose(self, choices):
        """Use an index array to construct a new array from a set of choices.

        Refer to :func:`dask.array.choose` for full documentation.

        See Also
        --------
        dask.array.choose : equivalent function
        """
        from dask_array._routines import choose

        return choose(self, choices)

    def nonzero(self):
        """Return the indices of the elements that are non-zero.

        Refer to :func:`dask.array.nonzero` for full documentation.

        See Also
        --------
        dask.array.nonzero : equivalent function
        """
        from dask_array._routines import nonzero

        return nonzero(self)

    def round(self, decimals=0):
        """Return array with each element rounded to the given number of decimals.

        Refer to :func:`dask.array.round` for full documentation.

        See Also
        --------
        dask.array.round : equivalent function
        """
        from dask_array._routines import round

        return round(self, decimals=decimals)

    def rechunk(
        self,
        chunks="auto",
        threshold=None,
        block_size_limit=None,
        balance=False,
        method=None,
    ):
        return rechunk(self, chunks, threshold, block_size_limit, balance, method)

    def shuffle(self, indexer, axis, chunks="auto"):
        from dask_array._shuffle import shuffle

        return shuffle(self, indexer, axis, chunks)

    def _vindex(self, key):
        from dask_array.slicing import _numpy_vindex, _vindex
        from dask.base import is_dask_collection

        if not isinstance(key, tuple):
            key = (key,)
        if any(k is None for k in key):
            raise IndexError(f"vindex does not support indexing with None (np.newaxis), got {key}")
        if all(isinstance(k, slice) for k in key):
            if all(k.indices(d) == slice(0, d).indices(d) for k, d in zip(key, self.shape)):
                return self
            raise IndexError(
                "vindex requires at least one non-slice to vectorize over "
                "when the slices are not over the entire array (i.e, x[:]). "
                f"Use normal slicing instead when only using slices. Got: {key}"
            )
        elif any(is_dask_collection(k) for k in key):
            if math.prod(self.numblocks) == 1 and len(key) == 1 and self.ndim == 1:
                idxr = key[0]
                # we can broadcast in this case
                return idxr.map_blocks(_numpy_vindex, self, dtype=self.dtype, chunks=idxr.chunks)
            else:
                raise IndexError(
                    "vindex does not support indexing with dask objects. Call compute "
                    f"on the indexer first to get an evalurated array. Got: {key}"
                )
        return _vindex(self, *key)

    @property
    def vindex(self):
        """Vectorized indexing with broadcasting.

        This is equivalent to numpy's advanced indexing, using arrays that are
        broadcast against each other. This allows for pointwise indexing:

        >>> import dask_array as da
        >>> x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> x = da.from_array(x, chunks=2)
        >>> x.vindex[[0, 1, 2], [0, 1, 2]].compute()
        array([1, 5, 9])

        Mixed basic/advanced indexing with slices/arrays is also supported. The
        order of dimensions in the result follows those proposed for
        `ndarray.vindex <https://github.com/numpy/numpy/pull/6256>`_:
        the subspace spanned by arrays is followed by all slices.

        Note: ``vindex`` provides more general functionality than standard
        indexing, but it also has fewer optimizations and can be significantly
        slower.
        """
        from dask.utils import IndexCallable

        return IndexCallable(self._vindex)

    def store(self, target, **kwargs):
        """Store array in array-like object.

        Refer to :func:`dask.array.store` for full documentation.
        """
        from dask_array.io import store

        return store([self], [target], **kwargs)

    def to_zarr(self, *args, **kwargs):
        """Save array to the zarr storage format

        See https://zarr.readthedocs.io for details about the format.

        Refer to :func:`dask.array.to_zarr` for full documentation.

        See also
        --------
        dask.array.to_zarr : equivalent function
        """
        from dask_array.io import to_zarr

        return to_zarr(self, *args, **kwargs)

    def to_tiledb(self, uri, *args, **kwargs):
        """Save array to the TileDB storage manager

        See https://docs.tiledb.io for details about the format and engine.

        Refer to :func:`dask.array.to_tiledb` for full documentation.

        See also
        --------
        dask.array.to_tiledb : equivalent function
        """
        from dask_array.io._tiledb import to_tiledb

        return to_tiledb(self, uri, *args, **kwargs)

    def to_hdf5(self, filename, datapath, **kwargs):
        """Store array in HDF5 file

        >>> x.to_hdf5('myfile.hdf5', '/x')  # doctest: +SKIP

        Optionally provide arguments as though to ``h5py.File.create_dataset``

        >>> x.to_hdf5('myfile.hdf5', '/x', compression='lzf', shuffle=True)  # doctest: +SKIP

        See Also
        --------
        dask_array.to_hdf5
        h5py.File.create_dataset
        """
        from dask_array.io._store import to_hdf5

        return to_hdf5(filename, datapath, self, **kwargs)

    def to_backend(self, backend: str | None = None, **kwargs):
        """Move to a new Array backend

        Parameters
        ----------
        backend : str, Optional
            The name of the new backend to move to. The default
            is the current "array.backend" configuration.

        Returns
        -------
        Array
        """
        from dask_array.creation._utils import to_backend

        return to_backend(self, backend=backend, **kwargs)  # type: ignore[arg-type]

    def to_svg(self, size=500):
        """Convert chunks from Dask Array into an SVG Image

        Parameters
        ----------
        size : int
            Rough size of the image

        Returns
        -------
        str
            An svg string depicting the array as a grid of chunks
        """
        from dask_array._svg import svg

        return svg(self.chunks, size=size)

    def copy(self):
        """Copy array. This is a no-op for dask arrays, which are immutable."""
        return Array(self._expr)

    def __deepcopy__(self, memo):
        c = self.copy()
        memo[id(self)] = c
        return c

    def to_delayed(self, optimize_graph=True):
        """Convert into an array of :class:`dask.delayed.Delayed` objects, one per chunk.

        Parameters
        ----------
        optimize_graph : bool, optional
            If True [default], the graph is optimized before converting into
            :class:`dask.delayed.Delayed` objects.

        See Also
        --------
        dask.array.from_delayed
        """
        from dask.delayed import Delayed
        from dask.utils import ndeepmap

        keys = self.__dask_keys__()
        graph = self.__dask_graph__()
        if optimize_graph:
            graph = self.__dask_optimize__(graph, keys)
        L = ndeepmap(self.ndim, lambda k: Delayed(k, graph), keys)
        return np.array(L, dtype=object)

    def to_dask_dataframe(self, columns=None, index=None, meta=None):
        """Convert dask Array to dask Dataframe

        Parameters
        ----------
        columns: list or string
            list of column names if DataFrame, single string if Series
        index : dask.dataframe.Index, optional
            An optional *dask* Index to use for the output Series or DataFrame.

            The default output index depends on whether the array has any unknown
            chunks. If there are any unknown chunks, the output has ``None``
            for all the divisions (one per chunk). If all the chunks are known,
            a default index with known divisions is created.

            Specifying ``index`` can be useful if you're conforming a Dask Array
            to an existing dask Series or DataFrame, and you would like the
            indices to match.
        meta : object, optional
            An optional `meta` parameter can be passed for dask
            to specify the concrete dataframe type to use for partitions of
            the Dask dataframe. By default, pandas DataFrame is used.

        See Also
        --------
        dask.dataframe.from_dask_array
        """
        from dask.dataframe.dask_expr._array import from_dask_array_expr

        return from_dask_array_expr(self, columns=columns, index=index, meta=meta)

    def sum(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        """
        Return the sum of the array elements over the given axis.

        Refer to :func:`dask.array.sum` for full documentation.

        See Also
        --------
        dask.array.sum : equivalent function
        """
        from dask_array.reductions._common import sum

        return sum(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            split_every=split_every,
            out=out,
        )

    def mean(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        """Returns the average of the array elements along given axis.

        Refer to :func:`dask.array.mean` for full documentation.

        See Also
        --------
        dask.array.mean : equivalent function
        """
        from dask_array.reductions._common import mean

        return mean(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            split_every=split_every,
            out=out,
        )

    def std(self, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None):
        """Returns the standard deviation of the array elements along given axis.

        Refer to :func:`dask.array.std` for full documentation.

        See Also
        --------
        dask.array.std : equivalent function
        """
        from dask_array.reductions._common import std

        return std(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            ddof=ddof,
            split_every=split_every,
            out=out,
        )

    def var(self, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None):
        """Returns the variance of the array elements, along given axis.

        Refer to :func:`dask.array.var` for full documentation.

        See Also
        --------
        dask.array.var : equivalent function
        """
        from dask_array.reductions._common import var

        return var(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            ddof=ddof,
            split_every=split_every,
            out=out,
        )

    def moment(
        self,
        order,
        axis=None,
        dtype=None,
        keepdims=False,
        ddof=0,
        split_every=None,
        out=None,
    ):
        """Calculate the nth centralized moment.

        Refer to :func:`dask.array.moment` for the full documentation.

        See Also
        --------
        dask.array.moment : equivalent function
        """
        from dask_array.reductions._common import moment

        return moment(
            self,
            order,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            ddof=ddof,
            split_every=split_every,
            out=out,
        )

    def prod(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        """Return the product of the array elements over the given axis

        Refer to :func:`dask.array.prod` for full documentation.

        See Also
        --------
        dask.array.prod : equivalent function
        """
        from dask_array.reductions._common import prod

        return prod(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            split_every=split_every,
            out=out,
        )

    def any(self, axis=None, keepdims=False, split_every=None, out=None):
        """Returns True if any of the elements evaluate to True.

        Refer to :func:`dask.array.any` for full documentation.

        See Also
        --------
        dask.array.any : equivalent function
        """
        from dask_array.reductions._common import any

        return any(self, axis=axis, keepdims=keepdims, split_every=split_every, out=out)

    def all(self, axis=None, keepdims=False, split_every=None, out=None):
        """Returns True if all elements evaluate to True.

        Refer to :func:`dask.array.all` for full documentation.

        See Also
        --------
        dask.array.all : equivalent function
        """
        from dask_array.reductions._common import all

        return all(self, axis=axis, keepdims=keepdims, split_every=split_every, out=out)

    def min(self, axis=None, keepdims=False, split_every=None, out=None):
        """Return the minimum along a given axis.

        Refer to :func:`dask.array.min` for full documentation.

        See Also
        --------
        dask.array.min : equivalent function
        """
        from dask_array.reductions._common import min

        return min(self, axis=axis, keepdims=keepdims, split_every=split_every, out=out)

    def max(self, axis=None, keepdims=False, split_every=None, out=None):
        """Return the maximum along a given axis.

        Refer to :func:`dask.array.max` for full documentation.

        See Also
        --------
        dask.array.max : equivalent function
        """
        from dask_array.reductions._common import max

        return max(self, axis=axis, keepdims=keepdims, split_every=split_every, out=out)

    def argmin(self, axis=None, *, keepdims=False, split_every=None, out=None):
        """Return indices of the minimum values along the given axis.

        Refer to :func:`dask.array.argmin` for full documentation.

        See Also
        --------
        dask.array.argmin : equivalent function
        """
        from dask_array.reductions._common import argmin

        return argmin(self, axis=axis, keepdims=keepdims, split_every=split_every, out=out)

    def argmax(self, axis=None, *, keepdims=False, split_every=None, out=None):
        """Return indices of the maximum values along the given axis.

        Refer to :func:`dask.array.argmax` for full documentation.

        See Also
        --------
        dask.array.argmax : equivalent function
        """
        from dask_array.reductions._common import argmax

        return argmax(self, axis=axis, keepdims=keepdims, split_every=split_every, out=out)

    def topk(self, k, axis=-1, split_every=None):
        """The top k elements of an array.

        Refer to :func:`dask.array.topk` for full documentation.

        See Also
        --------
        dask.array.topk : equivalent function
        """
        from dask_array._routines import topk

        return topk(self, k, axis=axis, split_every=split_every)

    def argtopk(self, k, axis=-1, split_every=None):
        """The indices of the top k elements of an array.

        Refer to :func:`dask.array.argtopk` for full documentation.

        See Also
        --------
        dask.array.argtopk : equivalent function
        """
        from dask_array._routines import argtopk

        return argtopk(self, k, axis=axis, split_every=split_every)

    def cumsum(self, axis, dtype=None, out=None, *, method="sequential"):
        """Return the cumulative sum of the elements along the given axis.

        Refer to :func:`dask.array.cumsum` for full documentation.

        See Also
        --------
        dask.array.cumsum : equivalent function
        """
        from dask_array.reductions._cumulative import cumsum

        return cumsum(self, axis=axis, dtype=dtype, out=out, method=method)

    def cumprod(self, axis, dtype=None, out=None, *, method="sequential"):
        """Return the cumulative product of the elements along the given axis.

        Refer to :func:`dask.array.cumprod` for full documentation.

        See Also
        --------
        dask.array.cumprod : equivalent function
        """
        from dask_array.reductions._cumulative import cumprod

        return cumprod(self, axis=axis, dtype=dtype, out=out, method=method)

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None):
        """Return the sum along diagonals of the array.

        Refer to :func:`dask.array.trace` for full documentation.

        See Also
        --------
        dask.array.trace : equivalent function
        """
        from dask_array.reductions._trace import trace

        return trace(self, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)

    def dot(self, other):
        """Dot product of self and other.

        Refer to :func:`dask.array.tensordot` for full documentation.

        See Also
        --------
        dask.array.dot : equivalent function
        """
        from dask_array.linalg import tensordot

        return tensordot(self, other, axes=((self.ndim - 1,), (other.ndim - 2,)))

    def astype(self, dtype, **kwargs):
        """Copy of the array, cast to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
            Controls what kind of data casting may occur. Defaults to 'unsafe'
            for backwards compatibility.

            * 'no' means the data types should not be cast at all.
            * 'equiv' means only byte-order changes are allowed.
            * 'safe' means only casts which can preserve values are allowed.
            * 'same_kind' means only safe casts or casts within a kind,
                like float64 to float32, are allowed.
            * 'unsafe' means any data conversions may be done.
        copy : bool, optional
            By default, astype always returns a newly allocated array. If this
            is set to False and the `dtype` requirement is satisfied, the input
            array is returned instead of a copy.

            .. note::

                Dask does not respect the contiguous memory layout of the array,
                and will ignore the ``order`` keyword argument.
                The default order is 'C' contiguous.
        """
        kwargs.pop("order", None)  # `order` is not respected, so we remove this kwarg
        # Scalars don't take `casting` or `copy` kwargs - as such we only pass
        # them to `map_blocks` if specified by user (different than defaults).
        extra = set(kwargs) - {"casting", "copy"}
        if extra:
            raise TypeError(f"astype does not take the following keyword arguments: {list(extra)}")
        casting = kwargs.get("casting", "unsafe")
        dtype = np.dtype(dtype)
        if self.dtype == dtype:
            return self
        elif not np.can_cast(self.dtype, dtype, casting=casting):
            raise TypeError(f"Cannot cast array from {self.dtype!r} to {dtype!r} according to the rule {casting!r}")
        return elemwise(chunk.astype, self, dtype=dtype, astype_dtype=dtype, **kwargs)

    def map_blocks(self, func, *args, **kwargs):
        from dask_array._map_blocks import map_blocks

        return map_blocks(func, self, *args, **kwargs)

    @property
    def _elemwise(self):
        return elemwise

    @property
    def real(self):
        from dask_array._ufunc import real

        return real(self)

    @property
    def imag(self):
        from dask_array._ufunc import imag

        return imag(self)

    def conj(self):
        """Complex-conjugate all elements.

        Refer to :func:`dask.array.conj` for full documentation.

        See Also
        --------
        dask.array.conj : equivalent function
        """
        from dask_array._ufunc import conj

        return conj(self)

    def clip(self, min=None, max=None):
        """Return an array whose values are limited to ``[min, max]``.
        One of max or min must be given.

        Refer to :func:`dask.array.clip` for full documentation.

        See Also
        --------
        dask.array.clip : equivalent function
        """
        from dask_array._ufunc import clip

        return clip(self, min, max)

    def view(self, dtype=None, order="C"):
        """Get a view of the array as a new data type

        Parameters
        ----------
        dtype:
            The dtype by which to view the array.
            The default, None, results in the view having the same data-type
            as the original array.
        order: string
            'C' or 'F' (Fortran) ordering

        This reinterprets the bytes of the array under a new dtype.  If that
        dtype does not have the same size as the original array then the shape
        will change.

        Beware that both numpy and dask.array can behave oddly when taking
        shape-changing views of arrays under Fortran ordering.  Under some
        versions of NumPy this function will fail when taking shape-changing
        views of Fortran ordered arrays if the first dimension has chunks of
        size one.
        """
        if dtype is None:
            dtype = self.dtype
        else:
            dtype = np.dtype(dtype)
        mult = self.dtype.itemsize / dtype.itemsize

        def _ensure_int(f):
            i = int(f)
            if i != f:
                raise ValueError(f"Could not coerce {f:f} to integer")
            return i

        if order == "C":
            chunks = self.chunks[:-1] + (tuple(_ensure_int(c * mult) for c in self.chunks[-1]),)
        elif order == "F":
            chunks = (tuple(_ensure_int(c * mult) for c in self.chunks[0]),) + self.chunks[1:]
        else:
            raise ValueError("Order must be one of 'C' or 'F'")

        return self.map_blocks(chunk.view, dtype, order=order, dtype=dtype, chunks=chunks)

    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            if _should_delegate(self, x):
                return NotImplemented

        if method == "__call__":
            if numpy_ufunc is np.matmul:
                from dask_array.linalg import matmul

                # special case until apply_gufunc handles optional dimensions
                return matmul(*inputs, **kwargs)
            if numpy_ufunc.signature is not None:
                from dask_array._gufunc import apply_gufunc

                return apply_gufunc(numpy_ufunc, numpy_ufunc.signature, *inputs, **kwargs)
            if numpy_ufunc.nout > 1:
                from dask_array import _ufunc as ufunc

                try:
                    da_ufunc = getattr(ufunc, numpy_ufunc.__name__)
                except AttributeError:
                    return NotImplemented
                return da_ufunc(*inputs, **kwargs)
            else:
                return elemwise(numpy_ufunc, *inputs, **kwargs)
        elif method == "outer":
            from dask_array import _ufunc as ufunc

            try:
                da_ufunc = getattr(ufunc, numpy_ufunc.__name__)
            except AttributeError:
                return NotImplemented
            return da_ufunc.outer(*inputs, **kwargs)
        else:
            return NotImplemented

    def map_overlap(self, func, depth, boundary=None, trim=True, **kwargs):
        """Map a function over blocks of the array with some overlap

        Refer to :func:`dask.array.map_overlap` for full documentation.

        See Also
        --------
        dask.array.map_overlap : equivalent function
        """
        from dask_array._overlap import map_overlap

        return map_overlap(func, self, depth=depth, boundary=boundary, trim=trim, **kwargs)


# Import rechunk, reshape, ravel from their modules
from dask_array._rechunk import rechunk
from dask_array._reshape import ravel, reshape, reshape_blockwise

# Import swapaxes
from dask_array.manipulation._transpose import swapaxes

# Import squeeze from its module
from dask_array.slicing import squeeze
