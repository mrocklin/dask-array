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

from dask_array._frisky.base import Layer
from dask_array._frisky.blockwise import BlockwiseLayer
from dask_array._frisky.collect import collect_task_records
from dask_array._frisky.creation import CreationLayer

__all__ = ["Layer", "BlockwiseLayer", "CreationLayer", "collect_task_records"]
