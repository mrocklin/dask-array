"""
xarray ChunkManager integration for dask-array expressions.

Design Choice: Replacing the Standard DaskManager
-------------------------------------------------
This module registers under the entry point name "dask", which REPLACES
xarray's built-in DaskManager. This is intentional for the following reasons:

1. Both dask_array.Array and dask.array.Array are valid dask collections
   (they inherit from dask.base.DaskMethodsMixin and implement __dask_graph__).

2. xarray's built-in DaskManager uses duck-typing (is_duck_dask_array) which
   recognizes ANY dask collection with array properties, including our
   dask_array.Array.

3. If we registered under a different name (e.g., "dask_array_expr"), xarray
   would see two ChunkManagers claiming the same array type and raise:
   "Multiple ChunkManagers recognise type <class 'dask_array._collection.Array'>"

4. By replacing the standard manager, we handle BOTH array types:
   - dask_array.Array: uses optimized expression-based operations
   - dask.array.Array: delegates to standard dask.array functions

This means when this package is installed, it becomes the default chunk manager
for all dask-based chunked arrays in xarray.

Usage:
    import xarray as xr

    # Works automatically - no need to specify chunk manager
    ds = xr.open_dataset("file.nc", chunks={"x": 100})

    # Can also be explicit
    ds = xr.open_dataset("file.nc", chunks={"x": 100}, chunked_array_type="dask")
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
