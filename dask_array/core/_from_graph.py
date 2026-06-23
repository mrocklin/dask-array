"""Create array from existing task graph."""

from __future__ import annotations

from dask._task_spec import Alias
from dask._task_spec import GraphNode
from dask._task_spec import TaskRef
from dask_array._new_collection import new_collection
from dask_array.io import FromGraph


def _dependency_keys_in_layer(layer, name):
    keys = set()
    for value in layer.values():
        stack = [value]
        while stack:
            value = stack.pop()
            if isinstance(value, TaskRef):
                value = value.key
            if isinstance(value, GraphNode):
                stack.extend(value.dependencies)
            elif isinstance(value, tuple):
                if value and value[0] == name:
                    keys.add(value)
                elif value and callable(value[0]):
                    stack.extend(value[1:])
            elif isinstance(value, (list, set)):
                stack.extend(value)
            elif isinstance(value, dict):
                stack.extend(value.values())
    return keys


def from_graph(layer, _meta, chunks, keys, name_prefix, dependencies=(), rename=None):
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
        Flattened list of task keys
    name_prefix : str
        Prefix for generating the array name
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
    # Lower dependencies the same way the Array collection names them
    # (``Array._name`` derives from ``_lower``). A caller builds ``layer`` with
    # the collection's keys, so a dependency we splice in must produce those
    # same keys — lowering only (without the gated simplify) would name it
    # differently and dangle the layer's references.
    from dask_array._collection import _lower

    expr_dependencies = []
    aliases = {}
    layer_dict = None
    for dep in dependencies:
        expr = getattr(dep, "expr", dep)
        lowered = _lower(expr) if hasattr(expr, "lower_completely") else expr
        expr_dependencies.append(lowered)
        if getattr(lowered, "_name", None) == getattr(expr, "_name", None):
            continue
        if layer_dict is None:
            layer_dict = dict(layer)
        for old_key in _dependency_keys_in_layer(layer_dict, expr._name):
            if len(old_key) != len(expr.numblocks) + 1:
                continue
            if not all(isinstance(i, int) for i in old_key[1:]):
                continue
            new_key = (lowered._name, *old_key[1:])
            aliases[old_key] = Alias(old_key, new_key)

    if aliases:
        layer = layer_dict
        layer.update(aliases)

    if rename is not None:
        name_prefix = rename.get(
            next(
                (key[0] for key in keys if isinstance(key, tuple) and key[0] in rename),
                name_prefix,
            ),
            rename.get(name_prefix, name_prefix),
        )
        keys = [(rename.get(key[0], key[0]), *key[1:]) if isinstance(key, tuple) else key for key in keys]

    return new_collection(
        FromGraph(
            layer=layer,
            _meta=_meta,
            chunks=chunks,
            keys=keys,
            name_prefix=name_prefix,
            _dependencies=tuple(expr_dependencies),
        )
    )
