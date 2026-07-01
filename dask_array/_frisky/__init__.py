"""Frisky graph helpers for dask-array.

The package exports native layer classes lazily so ordinary Dask graph
materialization does not import ``dask_array._rust`` just by touching the
``dask_array._frisky`` namespace.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Layer": ("dask_array._frisky.base", "Layer"),
    "ArangeLayer": ("dask_array._frisky.arange", "ArangeLayer"),
    "ArgChunkLayer": ("dask_array._frisky.arg_chunk", "ArgChunkLayer"),
    "BlocksLayer": ("dask_array._frisky.blocks", "BlocksLayer"),
    "BlockwiseLayer": ("dask_array._frisky.blockwise", "BlockwiseLayer"),
    "BroadcastLayer": ("dask_array._frisky.broadcast", "BroadcastLayer"),
    "CoarsenLayer": ("dask_array._frisky.coarsen", "CoarsenLayer"),
    "ConcatenateLayer": ("dask_array._frisky.concatenate", "ConcatenateLayer"),
    "CreationLayer": ("dask_array._frisky.creation", "CreationLayer"),
    "CumReductionLayer": ("dask_array._frisky.cumulative", "CumReductionLayer"),
    "Diag1DLayer": ("dask_array._frisky.diag", "Diag1DLayer"),
    "Diag2DSimpleLayer": ("dask_array._frisky.diag", "Diag2DSimpleLayer"),
    "ExpandDimsLayer": ("dask_array._frisky.expand_dims", "ExpandDimsLayer"),
    "EyeLayer": ("dask_array._frisky.eye", "EyeLayer"),
    "FromArrayGetterLayer": ("dask_array._frisky.from_array", "FromArrayGetterLayer"),
    "FromArrayLayer": ("dask_array._frisky.from_array", "FromArrayLayer"),
    "FromMapLayer": ("dask_array._frisky.from_map", "FromMapLayer"),
    "FusedBlockwiseLayer": ("dask_array._frisky.fused_blockwise", "FusedBlockwiseLayer"),
    "LinspaceLayer": ("dask_array._frisky.linspace", "LinspaceLayer"),
    "OverlapLayer": ("dask_array._frisky.overlap", "OverlapLayer"),
    "PartialReduceLayer": ("dask_array._frisky.reduction", "PartialReduceLayer"),
    "RandomLayer": ("dask_array._frisky.random", "RandomLayer"),
    "RechunkLayer": ("dask_array._frisky.rechunk", "RechunkLayer"),
    "ReshapeLayer": ("dask_array._frisky.reshape", "ReshapeLayer"),
    "SliceLayer": ("dask_array._frisky.slicing", "SliceLayer"),
    "ShuffleLayer": ("dask_array._frisky.shuffle", "ShuffleLayer"),
    "SqueezeLayer": ("dask_array._frisky.squeeze", "SqueezeLayer"),
    "StackLayer": ("dask_array._frisky.stack", "StackLayer"),
    "collect_task_records": ("dask_array._frisky.collect", "collect_task_records"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
