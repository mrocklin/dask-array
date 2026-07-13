"""
xarray ChunkManager integration for dask-array expressions.

This module registers a ChunkManagerEntrypoint under the entry point name
"dask" — the same name used by xarray's built-in DaskManager.  We *must*
replace the built-in rather than coexist alongside it because:

1. ``dask_array.Array`` is a dask collection (implements ``__dask_graph__``)
   and a duck array, so xarray's built-in ``DaskManager.is_chunked_array``
   recognises it via ``is_duck_dask_array``.
2. If two managers both claim the same array type, xarray's
   ``get_chunked_array_type`` raises
   ``"Multiple ChunkManagers recognise type ..."``.
3. Therefore only one "dask"-flavoured manager can be active at a time.

Because both xarray and dask-array register an entry point named "dask",
the winner of ``importlib.metadata.entry_points()`` iteration is
non-deterministic (it depends on filesystem enumeration order).  To make
the result reproducible, ``_ensure_registered`` mutates the cached dict
returned by ``list_chunkmanagers()`` so that our manager is always the one
stored under the "dask" key.

This module is imported lazily -- never by a plain ``import dask_array``
(which must not pull in xarray or pandas).  It loads through three paths,
all of which run ``_ensure_registered`` via the module-level call at the
bottom:

1. Explicitly, via ``dask_array.xarray.register()``.
2. By ``import dask_array`` when xarray is *already* in ``sys.modules``
   (the package __init__ pins eagerly then -- xarray is loaded, so it
   costs nothing).
3. By xarray itself: ``list_chunkmanagers`` loads every entry point in the
   "xarray.chunkmanagers" group on first chunked-array use, and ours points
   here.  In that mid-discovery case the reentrant ``list_chunkmanagers()``
   call builds and caches the registry (reentrant ``lru_cache`` calls win
   the cache slot) and we pin our manager into it, so every lookup after
   the in-flight one sees ours regardless of enumeration order.

Known window (path 3 only): the in-flight discovery still uses its own
un-pinned dict, whose "dask" slot goes to whichever entry point enumerates
*last* -- a per-environment installation detail.  If dask_array was imported
before xarray and the built-in entry point enumerates last, the single
chunked op that triggered discovery returns a legacy dask.array-backed
result.  That object stays usable: the pinned manager also claims legacy
``dask.array`` collections (``is_chunked_array`` -- both the classic and
the query-planning flavor) and converts them at its array-accepting entry
points (``_asexpr`` -- graph-wrapping via ``from_graph``, never compute),
so everything routed through the manager succeeds: compute/load return
correct numpy values, and ``.chunk``/persist/store/manager-dispatched
computation yield dask_array-backed results with the same values, dtype,
and chunks.  Operations xarray applies to the duck array directly -- plain
arithmetic, most reductions -- keep producing legacy-backed results until
one of those converting points; correct throughout, just unoptimized.
Every later chunked op engages dask_array directly.  One limitation:
combining a legacy-backed object with an expression-backed one in a single
operation still fails -- a TypeError from the operator layer for plain
arithmetic, xarray's "Mixing chunked array types" on manager-dispatched
paths -- so ``.chunk()`` the legacy-backed object first.  Calling
``dask_array.xarray.register()`` before first use eliminates the window in
every import order.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint

if TYPE_CHECKING:
    from dask_array._collection import Array


def _is_legacy_dask_array(data: Any) -> bool:
    """Whether ``data`` is a legacy ``dask.array`` collection.

    Matched structurally -- ``type(data)`` named ``Array`` in a ``dask.array``
    module, the same shape ``from_array`` uses to reject these inputs -- so
    both the classic ``dask.array.core.Array`` and the query-planning
    ``dask.array._array_expr`` collection are recognised.  Deliberately avoids
    importing ``dask.array``: its import registers global tokenize/dispatch
    handlers as a side effect.  A legacy instance can only exist if the
    package is already in ``sys.modules``, so the passive gate is sufficient
    (and keeps the common no-legacy case a single dict lookup).
    """
    if "dask.array" not in sys.modules:
        return False
    t = type(data)
    return t.__name__ == "Array" and t.__module__.startswith("dask.array")


def _asexpr(data: Any) -> Any:
    """Convert a legacy ``dask.array`` collection to a ``dask_array.Array``.

    Graph-wrapping, never compute: the legacy array's task graph, meta,
    chunks, and output keys are wrapped via ``from_graph`` (the sanctioned
    external-graph interop), preserving values, dtype, and chunk structure
    exactly.  Anything that is not a legacy dask array passes through
    unchanged, so this can be applied blindly at every array-accepting
    manager entry point.
    """
    if not _is_legacy_dask_array(data):
        return data

    from dask_array.core import from_graph

    keys = [(data.name, *block_id) for block_id in product(*map(range, data.numblocks))]
    return from_graph(dict(data.__dask_graph__()), data._meta, data.chunks, keys, data.name)


class DaskArrayExprManager(ChunkManagerEntrypoint["Array"]):
    """
    ChunkManager for dask-array expressions.

    This integrates dask_array.Array with xarray's chunked array interface,
    enabling expression-based optimizations for xarray operations.
    """

    array_cls: type[Array]
    available: bool = True

    def __init__(self) -> None:
        from dask_array._collection import Array

        self.array_cls = Array

    def is_chunked_array(self, data: Any) -> bool:
        # Also claim legacy dask.array collections: in the adverse discovery
        # order (module docstring, "Known window") the first chunked op in a
        # process produces a legacy-backed xarray object, and once our manager
        # pins the "dask" registry slot no other manager would recognise it.
        # Claiming it here -- and converting via _asexpr at the array-accepting
        # entry points below -- keeps that object fully usable.
        return isinstance(data, self.array_cls) or _is_legacy_dask_array(data)

    def chunks(self, data: Array) -> tuple[tuple[int, ...], ...]:
        # Legacy dask arrays expose .chunks with identical semantics, and the
        # result is plain tuples -- nothing to convert.
        return data.chunks

    def rechunk(
        self,
        data: Array,
        chunks: Any,
        **kwargs: Any,
    ) -> Array:
        # The inherited implementation calls data.rechunk(...), which on a
        # legacy dask array would return another legacy array. Convert first
        # so the result is expression-backed.
        return super().rechunk(_asexpr(data), chunks, **kwargs)

    def normalize_chunks(
        self,
        chunks: Any,
        shape: tuple[int, ...] | None = None,
        limit: int | None = None,
        dtype: np.dtype[Any] | None = None,
        previous_chunks: tuple[tuple[int, ...], ...] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        from dask_array._core_utils import normalize_chunks

        return normalize_chunks(
            chunks,
            shape=shape,
            limit=limit,
            dtype=dtype,
            previous_chunks=previous_chunks,
        )

    def from_array(
        self,
        data: Any,
        chunks: Any,
        **kwargs: Any,
    ) -> Array:
        import dask_array as da
        from xarray.core.indexing import ImplicitToExplicitIndexingAdapter

        if isinstance(data, ImplicitToExplicitIndexingAdapter):
            # Lazily loaded backend arrays should use NumPy arrays for meta.
            kwargs["meta"] = np.ndarray

        return da.from_array(data, chunks, **kwargs)

    def compute(
        self,
        *data: Array | Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], ...]:
        from dask import compute

        return compute(*self._pin(data), **kwargs)

    def persist(
        self,
        *data: Array | Any,
        **kwargs: Any,
    ) -> tuple[Array | Any, ...]:
        from dask import persist

        return persist(*self._pin(data), **kwargs)

    def _pin(self, data: tuple[Array | Any, ...]) -> tuple[Array | Any, ...]:
        """Materialize each array before handing it to ``dask.base``.

        Same reason as ``Array.compute``/``Array.persist`` routing through
        ``Array._pinned``: dask's generic optimizer over the raw expression
        would run unfused and derive drifting keys; the materialized
        expression is fused and keeps the collection's names, so persist
        round-trips a Dataset's variables under their original names.

        Legacy dask arrays are converted first (``_asexpr``) so they pin and
        compute like any other expression-backed array.
        """
        return tuple(d._pinned() if isinstance(d, self.array_cls) else d for d in map(_asexpr, data))

    @property
    def array_api(self) -> Any:
        import dask_array as da

        return da

    def reduction(
        self,
        arr: Array,
        func: Callable[..., Any],
        combine_func: Callable[..., Any] | None = None,
        aggregate_func: Callable[..., Any] | None = None,
        axis: int | Sequence[int] | None = None,
        dtype: np.dtype[Any] | None = None,
        keepdims: bool = False,
    ) -> Array:
        from dask_array.reductions._reduction import reduction

        return reduction(
            _asexpr(arr),
            chunk=func,
            combine=combine_func,
            aggregate=aggregate_func,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
        )

    def scan(
        self,
        func: Callable[..., Any],
        binop: Callable[..., Any],
        ident: float,
        arr: Array,
        axis: int | None = None,
        dtype: np.dtype[Any] | None = None,
        **kwargs: Any,
    ) -> Array:
        from dask_array.reductions._cumulative import cumreduction

        return cumreduction(
            func,
            binop,
            ident,
            _asexpr(arr),
            axis=axis,
            dtype=dtype,
            **kwargs,
        )

    def apply_gufunc(
        self,
        func: Callable[..., Any],
        signature: str,
        *args: Any,
        axes: Sequence[tuple[int, ...]] | None = None,
        axis: int | None = None,
        keepdims: bool = False,
        output_dtypes: Sequence[np.dtype[Any]] | None = None,
        output_sizes: dict[str, int] | None = None,
        vectorize: bool | None = None,
        allow_rechunk: bool = False,
        meta: tuple[np.ndarray[Any, Any], ...] | None = None,
        **kwargs: Any,
    ) -> Any:
        from dask_array import apply_gufunc

        return apply_gufunc(
            func,
            signature,
            *map(_asexpr, args),
            axes=axes,
            axis=axis,
            keepdims=keepdims,
            output_dtypes=output_dtypes,
            output_sizes=output_sizes,
            vectorize=vectorize,
            allow_rechunk=allow_rechunk,
            meta=meta,
            **kwargs,
        )

    def map_blocks(
        self,
        func: Callable[..., Any],
        *args: Any,
        dtype: np.dtype[Any] | None = None,
        chunks: tuple[int, ...] | None = None,
        drop_axis: int | Sequence[int] | None = None,
        new_axis: int | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> Any:
        from dask_array import map_blocks

        return map_blocks(
            func,
            *map(_asexpr, args),
            dtype=dtype,
            chunks=chunks,
            drop_axis=drop_axis,
            new_axis=new_axis,
            **kwargs,
        )

    def map_blocks_multi_output(
        self,
        func: Callable[..., Any],
        input_exprs: Sequence[Any],
        input_indices: Sequence[Iterable[Any]],
        shared_indices: Iterable[Any],
        block_specs: Mapping[tuple[int, ...], Any],
        outputs: Sequence[Mapping[str, Any]],
        *,
        token: str,
    ) -> list[Array]:
        from dask_array._map_blocks import map_blocks_multi_output

        # Inputs are expressions; convert any legacy dask array like the
        # sibling methods do and unwrap the resulting collection to its expr.
        converted = []
        for x in input_exprs:
            x = _asexpr(x)
            converted.append(x.expr if isinstance(x, self.array_cls) else x)

        return map_blocks_multi_output(
            func,
            converted,
            input_indices,
            shared_indices,
            block_specs,
            outputs,
            token=token,
        )

    def blockwise(
        self,
        func: Callable[..., Any],
        out_ind: Iterable[Any],
        *args: Any,
        name: str | None = None,
        token: Any | None = None,
        dtype: np.dtype[Any] | None = None,
        adjust_chunks: dict[Any, Callable[..., Any]] | None = None,
        new_axes: dict[Any, int] | None = None,
        align_arrays: bool = True,
        concatenate: bool | None = None,
        meta: tuple[np.ndarray[Any, Any], ...] | None = None,
        **kwargs: Any,
    ) -> Array:
        from dask_array import blockwise

        # args alternate (array, index) pairs; _asexpr passes indices through.
        return blockwise(
            func,
            out_ind,
            *map(_asexpr, args),
            name=name,
            token=token,
            dtype=dtype,
            adjust_chunks=adjust_chunks,
            new_axes=new_axes,
            align_arrays=align_arrays,
            concatenate=concatenate,
            meta=meta,
            **kwargs,
        )

    def unify_chunks(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[dict[str, tuple[tuple[int, ...], ...]], list[Array]]:
        from dask_array import unify_chunks

        # args alternate (array, index) pairs; _asexpr passes indices through.
        return unify_chunks(*map(_asexpr, args), **kwargs)

    def store(
        self,
        sources: Array | Sequence[Array],
        targets: Any,
        **kwargs: Any,
    ) -> Any:
        from dask_array import store

        if isinstance(sources, (list, tuple)):
            sources = type(sources)(map(_asexpr, sources))
        else:
            sources = _asexpr(sources)
        return store(
            sources=sources,
            targets=targets,
            **kwargs,
        )

    def shuffle(
        self,
        x: Array,
        indexer: list[list[int]],
        axis: int,
        chunks: Any,
    ) -> Array:
        from dask_array import shuffle

        if chunks is None:
            chunks = "auto"
        if chunks != "auto":
            raise NotImplementedError("Only chunks='auto' is supported at present.")
        return shuffle(_asexpr(x), indexer, axis, chunks="auto")

    def get_auto_chunk_size(self) -> int:
        from dask import config as dask_config
        from dask.utils import parse_bytes

        return parse_bytes(dask_config.get("array.chunk-size"))


def _ensure_registered() -> None:
    """Ensure DaskArrayExprManager is the "dask" chunk manager in xarray.

    Both xarray and this package register an entry point named "dask" under
    the ``xarray.chunkmanagers`` group.  ``list_chunkmanagers`` builds a dict
    from those entry points, so the *last* one enumerated wins.  Because
    ``importlib.metadata.entry_points`` iteration order is non-deterministic,
    we fix the race here by replacing the cached value after the fact.
    """
    try:
        from dask_array._backends import register_collection_types
        from xarray.namedarray.parallelcompat import list_chunkmanagers
    except ImportError:
        return

    register_collection_types()
    managers = list_chunkmanagers()
    if not isinstance(managers.get("dask"), DaskArrayExprManager):
        managers["dask"] = DaskArrayExprManager()


# Any import of this module (explicit register() or xarray's entry-point
# discovery) pins our manager; see the module docstring.
_ensure_registered()
