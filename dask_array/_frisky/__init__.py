"""Compact layer objects that emit a subgraph of tasks.

The graph code — how a layer expands into the individual tasks of its subgraph —
lives in Rust (``dask_array._rust``) and the per-layer modules here. The generic
translation of those neutral tasks into a concrete graph (``to_dask_graph``, the
legacy/correctness path, validated by the existing test suite) lives on the
``Layer`` base. A Frisky-tasks translator will join it later.

Each layer kind is its own module so new kinds can be added in their own files
with minimal cross-file contention — useful when several agents extend the layer
set in parallel.
"""

from dask_array._frisky.arange import ArangeLayer
from dask_array._frisky.arg_chunk import ArgChunkLayer
from dask_array._frisky.base import Layer
from dask_array._frisky.blocks import BlocksLayer
from dask_array._frisky.blockwise import BlockwiseLayer
from dask_array._frisky.broadcast import BroadcastLayer
from dask_array._frisky.coarsen import CoarsenLayer
from dask_array._frisky.collect import collect_task_records
from dask_array._frisky.concatenate import ConcatenateLayer
from dask_array._frisky.creation import CreationLayer
from dask_array._frisky.cumulative import CumReductionLayer
from dask_array._frisky.diag import Diag1DLayer, Diag2DSimpleLayer
from dask_array._frisky.expand_dims import ExpandDimsLayer
from dask_array._frisky.eye import EyeLayer
from dask_array._frisky.from_array import FromArrayLayer
from dask_array._frisky.linspace import LinspaceLayer
from dask_array._frisky.rechunk import RechunkLayer
from dask_array._frisky.reduction import PartialReduceLayer
from dask_array._frisky.reshape import ReshapeLayer
from dask_array._frisky.slicing import SliceLayer
from dask_array._frisky.squeeze import SqueezeLayer
from dask_array._frisky.stack import StackLayer

__all__ = [
    "Layer",
    "ArangeLayer",
    "ArgChunkLayer",
    "BlocksLayer",
    "BlockwiseLayer",
    "BroadcastLayer",
    "CoarsenLayer",
    "ConcatenateLayer",
    "CreationLayer",
    "CumReductionLayer",
    "Diag1DLayer",
    "Diag2DSimpleLayer",
    "ExpandDimsLayer",
    "EyeLayer",
    "FromArrayLayer",
    "LinspaceLayer",
    "PartialReduceLayer",
    "RechunkLayer",
    "ReshapeLayer",
    "SliceLayer",
    "SqueezeLayer",
    "StackLayer",
    "collect_task_records",
]
