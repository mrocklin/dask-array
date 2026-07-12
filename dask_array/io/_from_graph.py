from __future__ import annotations

import functools
from itertools import product

from dask._task_spec import Alias
from dask._task_spec import GraphNode
from dask_array._expr import ArrayExpr
from dask_array._new_collection import new_collection


class FromGraph(ArrayExpr):
    """An opaque, already-materialized task graph wrapped as an expression.

    ``_name`` is the caller-provided ``name`` verbatim — not content-derived —
    so a persisted collection keeps its original name and keys across the
    persist round-trip. Two consequences follow:

    - This class opts out of the singleton registry (via a non-trivial
      ``__init__``): two persists of the same collection produce same-name
      instances holding *different* futures and must not be conflated.
    - Everything beneath it must already be stable: the ``layer`` is opaque
      data and ``_dependencies`` are materialized expressions, so the rewrite
      framework's name-based change detection never needs to see through the
      pinned name.
    """

    _parameters = ["layer", "_meta", "chunks", "keys", "name", "_dependencies"]
    _defaults = {"_dependencies": ()}

    def __init__(self, *args, **kwargs):
        # A non-trivial __init__ disables SingletonExpr's dedup-by-_name
        # (it only dedups when cls.__init__ is object.__init__).
        pass

    def lower_once(self, lowered):
        # An opaque graph with materialized dependencies — nothing to lower.
        # Must never enter the (name-keyed) lowering cache: a persisted
        # collection carries its original raw root name, and a later tree
        # containing that raw subtree would get this node (and its futures)
        # silently spliced in on a cache hit.
        return self

    @functools.cached_property
    def _meta(self):
        return self.operand("_meta")

    @functools.cached_property
    def chunks(self):
        return self.operand("chunks")

    @functools.cached_property
    def _name(self):
        return self.operand("name")

    def dependencies(self):
        return list(self.operand("_dependencies"))

    def _find_layer_key(self, dsk, block_id):
        """The layer's key for one of our output blocks.

        Try the expected key from ``keys`` (matches for interop layers and
        for graphs dask materialized from our expression, e.g. the
        ``dask.optimize`` rebuild), then our own key (a persist that went
        through ``Array.persist``: the pinned graph produced our keys), and
        finally locate the block by block id — a scheduler that renamed the
        outputs hands back a bare ``{(name, *block_id): value}`` layer in a
        single name (the ``dask.persist``-on-a-raw-expression rebuild).
        """
        expected = self._keys_by_block_id.get(block_id)
        if expected is not None and expected in dsk:
            return expected
        out_key = (self._name, *block_id)
        if out_key in dsk:
            return out_key
        inferred = self._inferred_layer_name
        if inferred is not None:
            return (inferred, *block_id)
        raise ValueError(
            f"from_graph cannot find output block {block_id} (expected {expected or out_key}) in the layer. "
            "This typically means the graph was optimized outside dask-array's control with a rewrite "
            "that changed the output block structure (e.g. dask.persist on a raw sliding-window "
            "reduction). Use the collection's own .persist()/.compute(), which pin the output keys."
        )

    @functools.cached_property
    def _keys_by_block_id(self):
        by_block_id = {}
        for key in self.operand("keys"):
            other = by_block_id.setdefault(key[1:], key)
            if other != key:
                raise ValueError(f"from_graph got two output keys for block {key[1:]}: {other} and {key}")
        return by_block_id

    @functools.cached_property
    def _inferred_layer_name(self):
        """The single name whose keys cover exactly our block grid, if any."""
        grid = set(product(*(range(len(c)) for c in self.chunks)))
        ndim = len(self.chunks)
        block_ids = {}
        for k in self.operand("layer"):
            if isinstance(k, tuple) and len(k) == ndim + 1 and all(isinstance(i, int) for i in k[1:]):
                block_ids.setdefault(k[0], set()).add(k[1:])
        candidates = [name for name, bids in block_ids.items() if bids == grid]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _layer(self):
        from dask import istask

        dsk = dict(self.operand("layer"))
        # Bridge each of our ``(name, *block_id)`` output keys to the layer's
        # key for that block: plain data (persisted blocks, futures) is
        # rekeyed directly — so indexing the graph by a collection key yields
        # the data itself — while tasks keep their own key and gain an alias.
        # When the layer is already keyed by our keys (persist through
        # ``Array.persist``) this is a pure passthrough.
        for block_id in product(*(range(len(c)) for c in self.chunks)):
            out_key = (self._name, *block_id)
            layer_key = self._find_layer_key(dsk, block_id)
            if out_key == layer_key:
                continue
            value = dsk[layer_key]
            if isinstance(value, GraphNode) or istask(value):
                dsk[out_key] = Alias(out_key, layer_key)
            else:
                dsk[out_key] = value
                del dsk[layer_key]
        return dsk


def from_graph(layer, _meta, chunks, keys, name, dependencies=(), rename=None):
    """Create a dask array from an existing task graph.

    This is primarily used internally for reconstructing arrays after
    persistence or when recreating arrays from lowered expressions.

    Parameters
    ----------
    layer : dict or HighLevelGraph
        The task graph layer containing the array data
    _meta : array-like
        Metadata array describing the dtype and type of chunks
    chunks : tuple of tuples
        Chunk sizes for each dimension
    keys : list
        The layer's expected output-block keys, ``(some name, *block_id)``
        tuples, one per block. If a key is absent from the layer at graph
        time (a scheduler that renamed the outputs), the block is located in
        the layer by block id instead — see ``FromGraph._layer``.
    name : str
        The collection's name. Output keys are ``(name,) + block_id``; where
        the layer's key for a block differs, the block is rekeyed (data) or
        bridged with an alias (tasks). The persist path passes the persisted
        collection's own name, so persist is name-preserving.
    dependencies : sequence, optional
        Dask-array collections or expressions that provide keys referenced by
        ``layer``.
    rename : mapping, optional
        Mapping from old layer names to new layer names, passed by Dask graph
        manipulation when cloning collections.

    Returns
    -------
    Array
        A new dask Array wrapping the provided graph
    """
    # A dependency is embedded in its materialized (optimized + key-pinned)
    # form: ``layer`` references the dependency's public keys — its raw root
    # name — and materialization is exactly the operation that guarantees a
    # graph producing those keys (see ``_materialize``).
    from dask_array._collection import _materialize

    expr_dependencies = []
    for dep in dependencies:
        expr = getattr(dep, "expr", dep)
        if hasattr(expr, "lower_completely"):
            expr = _materialize(expr)
        expr_dependencies.append(expr)

    if rename is not None:
        name = rename.get(name, name)

    return new_collection(
        FromGraph(
            layer=layer,
            _meta=_meta,
            chunks=chunks,
            keys=keys,
            name=name,
            _dependencies=tuple(expr_dependencies),
        )
    )
