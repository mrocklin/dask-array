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
returned by ``list_chunkmanagers()`` at import time so that our manager
is always the one stored under the "dask" key.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint

if TYPE_CHECKING:
    from dask_array._collection import Array


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
        # Recognize dask_array.Array
        if isinstance(data, self.array_cls):
            return True
        # Also recognize legacy dask.array.Array
        try:
            import dask.array
            return isinstance(data, dask.array.Array)
        except ImportError:
            return False

    def chunks(self, data: Array) -> tuple[tuple[int, ...], ...]:
        return data.chunks

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

        return da.from_array(data, chunks, **kwargs)

    def rechunk(
        self,
        data: Array,
        chunks: Any,
        **kwargs: Any,
    ) -> Array:
        return data.rechunk(chunks, **kwargs)

    def compute(
        self,
        *data: Array | Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], ...]:
        from dask import compute

        return compute(*data, **kwargs)

    def persist(
        self,
        *data: Array | Any,
        **kwargs: Any,
    ) -> tuple[Array | Any, ...]:
        from dask import persist

        return persist(*data, **kwargs)

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
        from dask_array import reduction

        return reduction(
            arr,
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
        from dask_array import cumreduction

        return cumreduction(
            func,
            binop,
            ident,
            arr,
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
            *args,
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
            *args,
            dtype=dtype,
            chunks=chunks,
            drop_axis=drop_axis,
            new_axis=new_axis,
            **kwargs,
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

        return blockwise(
            func,
            out_ind,
            *args,
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

        return unify_chunks(*args, **kwargs)

    def store(
        self,
        sources: Array | Sequence[Array],
        targets: Any,
        **kwargs: Any,
    ) -> Any:
        from dask_array import store

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
        return shuffle(x, indexer, axis, chunks="auto")

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
        from xarray.namedarray.parallelcompat import list_chunkmanagers
    except ImportError:
        return

    managers = list_chunkmanagers()
    if not isinstance(managers.get("dask"), DaskArrayExprManager):
        managers["dask"] = DaskArrayExprManager()
