"""
xarray ChunkManager integration for dask-array expressions.

``_ensure_registered`` installs ``DaskArrayExprManager`` under the name
"dask" -- the same name as xarray's built-in DaskManager, which it
displaces.  We *must* replace the built-in rather than coexist alongside
it because:

1. ``dask_array.Array`` is a dask collection (implements ``__dask_graph__``)
   and a duck array, so xarray's built-in ``DaskManager.is_chunked_array``
   recognises it via ``is_duck_dask_array``.
2. If two managers both claim the same array type, xarray's
   ``get_chunked_array_type`` raises
   ``"Multiple ChunkManagers recognise type ..."``.
3. Therefore only one "dask"-flavoured manager can be active at a time.

Registration is *opt-in*: it happens only when the user calls
``dask_array.xarray.register()``, and nothing about installing or importing
this package triggers it.  We deliberately ship no "xarray.chunkmanagers"
entry point.  An entry point activates on install, which means adding
dask-array anywhere in an environment -- even as a transitive dependency
nobody imports -- would change what ``Dataset.chunk()`` returns for every
other library in that environment.  Downstream code that reaches past the
chunk manager for ``dask.array`` internals (icechunk's
``dask.array.reduction`` over ``store(return_stored=True)`` blocks, for
one) then breaks in ways whose cause is nowhere near the symptom.  For the
same reason ``import dask_array`` does not register either: xarray imports
us behind the user's back -- ``Dataset.__dask_exprs__`` calls
``import_module("dask_array")`` whenever we are installed -- so pinning on
import would be an install-time takeover wearing a different hat.

Registration works by mutating the dict cached by xarray's
``list_chunkmanagers()`` (an ``lru_cache``), replacing whatever sits in the
"dask" slot.  That takes effect in any import order and needs no entry
point of our own.

``register()`` before the first chunked operation gives a process where
every xarray array is expression-backed.  Registering *after* xarray has
already produced legacy ``dask.array`` objects is still supported: the
manager claims legacy collections too (``is_chunked_array`` -- both the
classic and the query-planning flavor) and converts them at its
array-accepting entry points (``_asexpr`` -- graph-wrapping via
``from_graph``, never compute).  So compute/load return correct numpy
values, and ``.chunk``/persist/store/manager-dispatched computation yield
dask_array-backed results with the same values, dtype, and chunks.
Operations xarray applies to the duck array directly -- plain arithmetic,
most reductions -- keep producing legacy-backed results until one of those
converting points; correct throughout, just unoptimized.  One limitation:
combining a legacy-backed object with an expression-backed one in a single
operation fails -- a TypeError from the operator layer for plain
arithmetic, xarray's "Mixing chunked array types" on manager-dispatched
paths -- so ``.chunk()`` the legacy-backed object first.
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
        # Also claim legacy dask.array collections: register() may be called
        # after xarray has already produced legacy-backed objects, and once our
        # manager holds the "dask" registry slot no other manager would
        # recognise them. Claiming them here -- and converting via _asexpr at
        # the array-accepting entry points below -- keeps them fully usable.
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
    """Make DaskArrayExprManager the "dask" chunk manager in xarray.

    ``list_chunkmanagers`` builds its dict from "xarray.chunkmanagers" entry
    points and caches it (``lru_cache``); we ship no entry point, so we take
    the "dask" slot by replacing the cached value.  Reached only through
    ``dask_array.xarray.register()`` -- importing this module registers
    nothing, see the module docstring.
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
