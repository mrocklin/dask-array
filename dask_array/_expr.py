from __future__ import annotations

import functools
import math
import re
import warnings
from functools import cached_property, reduce
from operator import mul
from typing import NamedTuple

import numpy as np
import toolz

from dask._expr import FinalizeCompute, SingletonExpr
from dask._task_spec import List, Task, TaskRef
from dask_array._core_utils import (
    PerformanceWarning,
    T_IntOrNaN,
    common_blockdim,
    unknown_chunk_message,
)
from dask.blockwise import broadcast_dimensions
from dask.layers import ArrayBlockwiseDep
from dask import config
from dask.utils import cached_cumsum, format_bytes, funcname, parse_bytes

_OBJECT_AT_PATTERN = re.compile(r"<.+? at 0x[0-9a-fA-F]+>")


class TransferBytes(NamedTuple):
    """(min, max) bytes moved between workers; see ArrayExpr.transfer_bytes."""

    min: float
    max: float


def _collect_cached_property_names(cls):
    """Collect all cached_property names from a class and its parents."""
    names = set()
    for parent in cls.__mro__:
        for k, v in parent.__dict__.items():
            if isinstance(v, functools.cached_property):
                names.add(k)
    return frozenset(names)


def _simplify_repr(op):
    """Simplify operand representation for tree_repr display."""
    if isinstance(op, np.ndarray):
        return "<array>"
    if isinstance(op, np.dtype):
        return str(op)
    if callable(op):
        return funcname(op)
    # Simplify objects that show "object at 0x..." in repr
    r = repr(op)
    if " object at 0x" in r:
        return f"<{type(op).__name__}>"
    return op


def _clean_header(header):
    """Clean up any remaining verbose patterns in the header string."""
    # Replace "<function foo at 0x...>" or "<X object at 0x...>" with "..."
    return _OBJECT_AT_PATTERN.sub("...", header)


def _convert_dask_keys(keys):
    if isinstance(keys, list):
        return List(*(_convert_dask_keys(key) for key in keys))
    return TaskRef(keys)


class ArrayExpr(SingletonExpr):
    # Whether this expression can be fused with other blockwise operations.
    # Override to True in subclasses that support fusion (Blockwise, Random, etc.)
    _is_blockwise_fusable = False

    def _all_input_block_ids(self, block_id):
        """Return all input block_ids for dependencies.

        Returns a dict mapping dep._name to a list of block_ids.
        This handles the case where the same dependency is used multiple
        times with different index mappings (e.g., da.dot(x, x)).

        Subclasses like Blockwise override this to iterate over all args.
        """
        result = {}
        for dep in self.dependencies():
            dep_block_id = self._input_block_id(dep, block_id)
            if dep._name not in result:
                result[dep._name] = []
            result[dep._name].append(dep_block_id)
        return result

    def _input_block_id(self, dep, block_id):
        """Map output block_id to input block_id for a dependency.

        Default implementation returns the same block_id.
        Subclasses override for transformations like transpose.
        """
        return block_id

    # Pre-computed set of cached_property names for efficient serialization
    _cached_property_names: frozenset[str] = frozenset()
    _pickle_excluded_cached_properties: frozenset[str] = frozenset({"_cached_keys"})

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._cached_property_names = _collect_cached_property_names(cls)

    def __reduce__(self):
        import dask
        from dask._expr import Expr

        if dask.config.get("dask-expr-no-serialize", False):
            raise RuntimeError(f"Serializing a {type(self)} object")
        cache = {}
        if type(self)._pickle_functools_cache:
            for k in type(self)._cached_property_names:
                if k in self.__dict__ and k not in type(self)._pickle_excluded_cached_properties:
                    cache[k] = self.__dict__[k]
        return Expr._reconstruct, (
            type(self),
            *self.operands,
            self.deterministic_token,
            cache,
        )

    def __dask_graph__(self):
        from dask._expr import Expr
        from dask_array._collection import _materialize

        # Materialize (not just optimize): pinning the output keys back to
        # this node's name means any dask-side consumer that pairs this graph
        # with our (raw) keys — e.g. ``dask.optimize``'s rebuild — finds them.
        return Expr.__dask_graph__(_materialize(self))

    def _layer(self):
        from dask._expr import Expr
        from dask_array._collection import _materialize

        expr = _materialize(self)
        if expr is self:
            raise NotImplementedError(f"{type(self).__name__} must implement _layer or lower before materialization")
        return Expr.__dask_graph__(expr)

    def _operands_for_repr(self):
        return []

    def _tree_repr_lines(self, indent=0, recursive=True):
        header = funcname(type(self)) + ":"
        lines = []
        for i, op in enumerate(self.operands):
            if isinstance(op, ArrayExpr):
                if recursive:
                    lines.extend(op._tree_repr_lines(2))
            else:
                op = _simplify_repr(op)
                header = self._tree_repr_argument_construction(i, op, header)

        header = _clean_header(header)
        lines = [header] + lines
        lines = [" " * indent + line for line in lines]
        return lines

    def _table(self, color=True):
        """Display expression tree as a formatted table.

        Requires the `rich` library to be installed.
        """
        from dask_array._visualize import expr_table

        return expr_table(self, color=color)

    def _repr_html_(self):
        """Jupyter notebook display using rich table."""
        try:
            return self._table()._repr_html_()
        except (ImportError, NotImplementedError):
            return f"<pre>{chr(10).join(self._tree_repr_lines())}</pre>"

    def __repr__(self):
        """Return rich table representation if available, else simple repr."""
        try:
            return repr(self._table())
        except (ImportError, NotImplementedError):
            return "\n".join(self._tree_repr_lines())

    def pprint(self):
        """Pretty print the expression tree using rich table if available."""
        try:
            self._table().print()
        except (ImportError, NotImplementedError):
            for line in self._tree_repr_lines():
                print(line)

    @cached_property
    def shape(self) -> tuple[T_IntOrNaN, ...]:
        return tuple(cached_cumsum(c, initial_zero=True)[-1] for c in self.chunks)

    @cached_property
    def ndim(self):
        return len(self.shape)

    @cached_property
    def chunksize(self) -> tuple[T_IntOrNaN, ...]:
        return tuple(max(c) for c in self.chunks)

    @cached_property
    def dtype(self):
        if isinstance(self._meta, tuple):
            dtype = self._meta[0].dtype
        else:
            dtype = self._meta.dtype
        return dtype

    @cached_property
    def chunks(self):
        if "chunks" in self._parameters:
            return self.operand("chunks")
        raise NotImplementedError("Subclass must implement 'chunks'")

    @cached_property
    def numblocks(self):
        return tuple(map(len, self.chunks))

    @cached_property
    def size(self) -> T_IntOrNaN:
        """Number of elements in array"""
        return reduce(mul, self.shape, 1)

    @property
    def nbytes(self) -> T_IntOrNaN:
        """Number of bytes in array"""
        return self.size * self.dtype.itemsize

    @cached_property
    def transfer_bytes(self):
        """Estimated (min, max) bytes moved between workers by this node.

        Counts only this node's incoming edges: the bytes of dependency
        blocks that cross a worker boundary to compute this node's output
        blocks.  Sum over the graph for a whole-graph estimate — on the
        *optimized* expression, since un-lowered nodes (MapOverlap, Sum,
        unaligned Elemwise) don't yet contain the rechunks and partials
        their lowering introduces.

        min assumes ideal placement: corresponding blocks of inputs and
        output are co-located, each task runs where its largest input
        piece lives, and slicing happens at the source.  Aligned elemwise
        ops cost 0, overlaps cost their ghost cells, rechunks cost the
        data that changes blocks.

        max assumes no co-location: every task fetches every input block
        it touches, whole, from a remote worker.  Real schedulers land
        between the two.

        This default assumes a block-aligned mapping: 1:1 is free under
        min, a small input broadcast to f output blocks moves f-1 copies,
        and a k-into-1 gather moves the (k-1)/k that must join the
        co-located block.  Classes that mix data across blocks (Rechunk,
        Shuffle, OverlapInternal, PartialReduce, ...) override; pure alias
        nodes (Concatenate) cost (0, 0).
        """
        out_blocks = reduce(mul, self.numblocks, 1)
        lo = 0.0
        hi = 0.0
        seen = set()
        for dep in self.dependencies():
            # a duplicated operand (x + x) is one dependency in the layer
            if not isinstance(dep, ArrayExpr) or dep._name in seen:
                continue
            seen.add(dep._name)
            ratio = out_blocks / max(1, reduce(mul, dep.numblocks, 1))
            nbytes = dep.nbytes
            if ratio >= 1:
                lo += nbytes * (ratio - 1.0)
                hi += nbytes * ratio
            else:
                lo += nbytes * (1.0 - ratio)
                hi += nbytes
        return TransferBytes(lo, hi)

    @property
    def name(self):
        return self._name

    def __len__(self):
        if not self.chunks:
            raise TypeError("len() of unsized object")
        if np.isnan(self.chunks[0]).any():
            msg = f"Cannot call len() on object with unknown chunk size.{unknown_chunk_message}"
            raise ValueError(msg)
        return int(sum(self.chunks[0]))

    @functools.cached_property
    def _cached_keys(self):
        out = self.lower_completely()

        name, chunks, numblocks = out.name, out.chunks, out.numblocks

        def keys(*args):
            if not chunks:
                return List(TaskRef((name,)))
            ind = len(args)
            if ind + 1 == len(numblocks):
                result = List(*(TaskRef((name,) + args + (i,)) for i in range(numblocks[ind])))
            else:
                result = List(*(keys(*(args + (i,))) for i in range(numblocks[ind])))
            return result

        return keys()

    def __dask_keys__(self):
        key_refs = self._cached_keys

        def unwrap(task):
            if isinstance(task, List):
                return [unwrap(t) for t in task.args]
            return task.key

        return unwrap(key_refs)

    def __hash__(self):
        return hash(self._name)

    def optimize(self, fuse: bool = True):
        expr = self.simplify().lower_completely()
        if fuse:
            expr = expr.fuse()
        return expr

    def fuse(self):
        from dask_array._blockwise import optimize_blockwise_fusion_array

        return optimize_blockwise_fusion_array(self)

    def rechunk(
        self,
        chunks="auto",
        threshold=None,
        block_size_limit=None,
        balance=False,
        method=None,
    ):
        if self.ndim > 0 and all(s == 0 for s in self.shape):
            return self

        from dask_array._rechunk import Rechunk
        from dask_array._core_utils import normalize_chunks
        from dask_array._utils import validate_axis

        # Pre-resolve chunks to check for no-op and avoid singleton caching issues
        resolved_chunks = chunks
        if isinstance(chunks, dict):
            normalized_dict = {validate_axis(k, self.ndim): v for k, v in chunks.items()}
            resolved_chunks = tuple(
                (normalized_dict[i] if i in normalized_dict and normalized_dict[i] is not None else self.chunks[i])
                for i in range(self.ndim)
            )
        if isinstance(resolved_chunks, (tuple, list)):
            resolved_chunks = tuple(lc if lc is not None else rc for lc, rc in zip(resolved_chunks, self.chunks))
        resolved_chunks = normalize_chunks(
            resolved_chunks,
            self.shape,
            limit=block_size_limit,
            dtype=self.dtype,
            previous_chunks=self.chunks,
        )

        # No-op rechunk: if chunks already match, return self
        if not balance and resolved_chunks == self.chunks:
            return self

        result = Rechunk(self, resolved_chunks, threshold, block_size_limit, balance, method)
        # Ensure that chunks are compatible
        result.chunks
        return result

    def finalize_compute(self):
        return FinalizeComputeArray(self)


def coarse_blockdim(blockdims):
    """Find the coarsest block dimension from a set of block dimensions.

    Prefers the chunking with the fewest blocks, which results in larger
    chunk sizes and fewer tasks. The finer-grained inputs will be rechunked
    to match.

    Unlike common_blockdim which finds the finest common divisor, this
    function prefers larger chunks to minimize task overhead. However, if
    the chunk boundaries don't align (one chunking's boundaries aren't a
    subset of another's), falls back to common_blockdim behavior.

    Parameters
    ----------
    blockdims : set of tuples
        Set of chunk tuples for a single dimension

    Returns
    -------
    tuple
        The preferred chunk tuple (fewest blocks if alignable, otherwise
        finest common divisor)

    Examples
    --------
    >>> coarse_blockdim({(12, 12, 12, 12), (1, 1, 1, 1, 1)})  # prefer fewer chunks
    (12, 12, 12, 12)
    >>> coarse_blockdim({(10,), (5, 5)})  # single chunk preferred
    (10,)
    >>> coarse_blockdim({(4, 6), (6, 4)})  # incompatible - use common divisor
    (4, 2, 4)
    """
    if not any(blockdims):
        return ()

    # Handle unknown chunks - same logic as common_blockdim
    unknown_dims = [d for d in blockdims if np.isnan(sum(d))]
    if unknown_dims:
        all_lengths = {len(d) for d in blockdims}
        if len(all_lengths) > 1:
            raise ValueError(
                "Chunks are unknown or misaligned along dimensions with missing values.\n\n"
                "A possible solution:\n  x.compute_chunk_sizes()"
            )
        return toolz.first(unknown_dims)

    # Filter out singleton dimensions (size 1) - they don't constrain chunking
    non_trivial_dims = {d for d in blockdims if len(d) > 1}

    if len(non_trivial_dims) == 0:
        # All are singletons, pick any
        return max(blockdims, key=toolz.first)

    if len(non_trivial_dims) == 1:
        # Only one non-trivial, use it
        return toolz.first(non_trivial_dims)

    # Multiple non-trivial dimensions - verify they have the same total size
    if len(set(map(sum, non_trivial_dims))) > 1:
        raise ValueError("Chunks do not add up to same value", blockdims)

    # Find the coarsest chunking (fewest blocks)
    coarsest = min(non_trivial_dims, key=len)

    # Check if all other chunkings have boundaries that align with the coarsest
    # i.e., the coarsest boundaries are a subset of each other chunking's boundaries
    coarsest_boundaries = set(np.cumsum(coarsest[:-1]))

    for chunks in non_trivial_dims:
        if chunks == coarsest:
            continue
        other_boundaries = set(np.cumsum(chunks[:-1]))
        if not coarsest_boundaries.issubset(other_boundaries):
            # Boundaries don't align - fall back to common_blockdim
            return common_blockdim(blockdims)

    # All boundaries align with the coarsest, so use it
    return coarsest


def unify_chunks_expr(*args, warn=True):
    # TODO(expr): This should probably be a dedicated expression
    # This is the implementation that expects the inputs to be expressions, the public facing
    # variant needs to sanitize the inputs
    if not args:
        return {}, [], False
    arginds = list(toolz.partition(2, args))
    arrays, inds = zip(*arginds)
    if all(ind is None for ind in inds):
        return {}, list(arrays), False
    if all(ind == inds[0] for ind in inds) and all(a.chunks == arrays[0].chunks for a in arrays):
        return dict(zip(inds[0], arrays[0].chunks)), arrays, False

    nameinds = []
    blockdim_dict = dict()
    max_parts = 0
    for a, ind in arginds:
        # Skip scalars (empty tuple index), literals (None), and ArrayBlockwiseDep
        if ind is not None and ind != () and not isinstance(a, ArrayBlockwiseDep):
            # Use _name (cheap, unlowered) rather than name as the grouping key.
            # name on a Reduction triggers a full lower_completely() of the whole
            # operand subtree, which makes chunk unification O(tree) per layer and
            # graph construction O(tree^2) for deep stacks. _name is a unique
            # per-node identifier, which is all broadcast_dimensions needs here.
            nameinds.append((a._name, ind))
            blockdim_dict[a._name] = a.chunks
            max_parts = max(max_parts, math.prod(a.numblocks))
        else:
            nameinds.append((a, ind))

    # array.unify-chunks-policy: "coarse" (default) merges nested chunkings up to the
    # coarsest operand to cut task counts; "refine" is stock-dask behavior (finest
    # common refinement -- splits only, never merges).
    consolidate = coarse_blockdim
    if config.get("array.unify-chunks-policy", "coarse") == "refine":
        consolidate = common_blockdim
    chunkss = broadcast_dimensions(nameinds, blockdim_dict, consolidate=consolidate)

    # Size guard, the twin of the count warning below: nest-coarsening merges chunks
    # without bound, so one coarsely-chunked operand can inflate every other operand
    # (and everything downstream) into multi-GB chunks.  If the merged layout would
    # exceed array.unify-chunks-limit for any participating array, redo the coarsened
    # dims with common_blockdim instead -- refinement only splits chunks, so the
    # fallback moves no data.
    limit = config.get("array.unify-chunks-limit", None)
    if limit and consolidate is coarse_blockdim:
        limit = parse_bytes(limit) if isinstance(limit, str) else limit
        worst = 0
        for a, i in arginds:
            if i is None or i == () or isinstance(a, ArrayBlockwiseDep):
                continue
            itemsize = a.dtype.itemsize
            target = itemsize * math.prod(max(chunkss[j]) for n, j in enumerate(i) if a.shape[n] > 1)
            current = itemsize * math.prod(max(c) for n, c in enumerate(a.chunks) if a.shape[n] > 1)
            if target > current:  # only chunks the merge would manufacture count
                worst = max(worst, target)
        if worst > limit:
            fine = broadcast_dimensions(nameinds, blockdim_dict, consolidate=common_blockdim)
            coarsened = {j for j, c in chunkss.items() if len(fine[j]) > len(c)}
            if coarsened:
                if warn:
                    warnings.warn(
                        f"Chunk unification would merge chunks up to {format_bytes(worst)}, "
                        f"above the 'array.unify-chunks-limit' of {format_bytes(limit)}; "
                        f"refining to the inputs' finer chunking instead.",
                        PerformanceWarning,
                        stacklevel=3,
                    )
                chunkss = {j: fine[j] if j in coarsened else c for j, c in chunkss.items()}

    nparts = math.prod(map(len, chunkss.values())) if chunkss else 0

    if warn and nparts and nparts >= max_parts * 10:
        warnings.warn(
            f"Increasing number of chunks by factor of {int(nparts / max_parts)}",
            PerformanceWarning,
            stacklevel=3,
        )

    arrays = []
    changed = False
    for a, i in arginds:
        if i is None or i == () or isinstance(a, ArrayBlockwiseDep):
            pass  # Skip scalars, literals, ArrayBlockwiseDep
        else:
            chunks = tuple(
                (chunkss[j] if a.shape[n] > 1 else (a.shape[n],) if not np.isnan(sum(chunkss[j])) else None)
                for n, j in enumerate(i)
            )
            if chunks != a.chunks and all(a.chunks):
                # Skip rechunking known chunks to unknown - can't rechunk to nan sizes
                target_has_nan = any(c is not None and np.isnan(sum(c)) for c in chunks)
                source_is_known = not any(np.isnan(sum(c)) for c in a.chunks)
                if not (target_has_nan and source_is_known):
                    a = a.rechunk(chunks)
                    changed = True
        arrays.append(a)
    return chunkss, arrays, changed


class FinalizeComputeArray(FinalizeCompute, ArrayExpr):
    _parameters = ["arr"]

    @cached_property
    def chunks(self):
        # Each dimension has a single chunk with the full size
        return tuple((s,) for s in self.arr.shape)

    def __dask_keys__(self):
        return [self._name]

    def _layer(self):
        from dask_array._core_utils import finalize

        return {
            self._name: Task(
                self._name,
                finalize,
                _convert_dask_keys(self.arr.__dask_keys__()),
            )
        }

    def _simplify_down(self):
        return None


class ChunksOverride(ArrayExpr):
    """Override chunks metadata for an array expression.

    This creates an alias layer while preserving the underlying computation.
    Useful when the actual output chunk sizes differ from what the expression
    system infers (e.g., boolean indexing produces unknown chunk sizes).
    """

    _parameters = ["array", "_chunks"]

    @functools.cached_property
    def _name(self):
        return f"chunks-override-{self.deterministic_token}"

    @functools.cached_property
    def _meta(self):
        return self.array._meta

    @functools.cached_property
    def chunks(self):
        return self._chunks

    @functools.cached_property
    def transfer_bytes(self):
        # Pure 1:1 alias layer -- no data moves.
        return TransferBytes(0.0, 0.0)

    def _frisky_layer(self):
        from dask_array._frisky.blocks import BlocksLayer

        # Pure 1:1 alias of every block (same coord). That is BlocksLayer with an
        # identity remap per dimension — output position d maps to input block d.
        index_maps = [list(range(len(c))) for c in self._chunks]
        return BlocksLayer(self._name, self.array._name, index_maps)

    def _layer(self) -> dict:
        from itertools import product

        from dask._task_spec import Alias

        dsk = {}
        chunk_ranges = [range(len(c)) for c in self._chunks]
        for idx in product(*chunk_ranges):
            out_key = (self._name,) + idx
            in_key = (self.array._name,) + idx
            dsk[out_key] = Alias(out_key, in_key)
        return dsk


class RootAlias(ArrayExpr):
    """Pin an optimized expression's output keys to a stable public name.

    Optimization (simplify/lower/fuse) renames every node it rewrites,
    including the root — but a collection's advertised keys
    (``Array.__dask_keys__``, and thus everything the dask collection protocol
    hands back after ``compute``/``persist``) must not change between "keys
    were advertised" and "results came back".  So materialization pins the
    root: the graph ends in one alias task per output block, mapping the
    stable ``(raw root name, i, ...)`` keys to the optimized root's keys.

    Only constructed *after* optimization has finished (``_materialize`` in
    ``_collection.py``), and never fed back through simplify/lower/fuse:
    ``_name`` is pinned rather than content-derived, so the rewrite
    framework's name-based change detection cannot see through it.  For the
    same reason this class opts out of the singleton registry — two pins of
    the same name around different targets must not be conflated.
    """

    _parameters = ["array", "name"]

    def __init__(self, *args, **kwargs):
        # A non-trivial __init__ disables SingletonExpr's dedup-by-_name
        # (it only dedups when cls.__init__ is object.__init__).
        pass

    def lower_once(self, lowered):
        # Already materialized — there is nothing left to lower. Crucially,
        # this must never enter the (name-keyed) lowering cache: our name is
        # the *raw* root's name, and a later tree containing that raw subtree
        # would get this pin spliced into its middle on a cache hit.
        return self

    @functools.cached_property
    def _name(self):
        return self.operand("name")

    @functools.cached_property
    def _meta(self):
        return self.array._meta

    @functools.cached_property
    def chunks(self):
        return self.array.chunks

    @functools.cached_property
    def transfer_bytes(self):
        # Pure 1:1 alias layer -- no data moves.
        return TransferBytes(0.0, 0.0)

    def _frisky_layer(self):
        from dask_array._frisky.blocks import BlocksLayer

        # 1:1 alias of every block (same coord), like ChunksOverride.
        index_maps = [list(range(len(c))) for c in self.chunks]
        return BlocksLayer(self._name, self.array._name, index_maps)

    def _layer(self) -> dict:
        from itertools import product

        from dask._task_spec import Alias

        dsk = {}
        for idx in product(*(range(len(c)) for c in self.chunks)):
            out_key = (self._name,) + idx
            in_key = (self.array._name,) + idx
            dsk[out_key] = Alias(out_key, in_key)
        return dsk
