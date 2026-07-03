"""Create array from existing task graph."""

from __future__ import annotations

from dask_array._new_collection import new_collection
from dask_array.io import FromGraph


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
