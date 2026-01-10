from __future__ import annotations

import os
import warnings

import numpy as np

from dask.base import tokenize

from dask_array._core_utils import normalize_chunks, unknown_chunk_message


class PerformanceWarning(Warning):
    """A warning given when bad chunking may cause poor performance."""


def _zarr_v3() -> bool:
    """Check if zarr version is 3.x or higher."""
    try:
        import zarr
        from packaging.version import Version
    except ImportError:
        return False
    else:
        return Version(zarr.__version__).major >= 3


def _setup_zarr_store(
    url: str, storage_options: dict[str, object] | None = None, **kwargs: object
):
    """
    Set up a Zarr store for reading or writing, handling both Zarr v2 and v3.

    This function prepares a Zarr-compatible storage object (`store`) from a URL or existing
    store. It supports optional storage options for fsspec-based stores and automatically
    selects the appropriate store type depending on the Zarr version.

    Parameters
    ----------
    url: Zarr Array or str or MutableMapping
        Location of the data. A URL can include a protocol specifier like s3://
        for remote data. Can also be any MutableMapping instance, which should
        be serializable if used in multiple processes.
    storage_options: dict | None, default = None
        Any additional parameters for the storage backend (ignored for local
        paths)
    **kwargs:
        Passed to determine whether the store should be readonly by evaluating the following:
        'kwargs.pop("read_only", kwargs.pop("mode", "a") == "r")'

    Returns
    -------
    store : zarr.store.Store or original url
        A Zarr-compatible store object. Can be:
        - `zarr.storage.FsspecStore` for Zarr v3 with storage options
        - `zarr.storage.FSStore` for Zarr v2 with storage options
        - The original URL/path if no storage options are provided
    """
    # Cannot directly import FSStore from storage.
    from zarr import storage

    if storage_options is not None:
        if _zarr_v3():
            read_only = kwargs.pop("read_only", kwargs.pop("mode", "a") == "r")
            store = storage.FsspecStore.from_url(
                url, read_only=read_only, storage_options=storage_options
            )
        else:
            store = storage.FSStore(url, **storage_options)
    else:
        store = url
    return store


def from_zarr(
    url,
    component=None,
    storage_options=None,
    chunks=None,
    name=None,
    inline_array=False,
    **kwargs,
):
    """Load array from the zarr storage format

    See https://zarr.readthedocs.io for details about the format.

    Parameters
    ----------
    url: Zarr Array or str or MutableMapping
        Location of the data. A URL can include a protocol specifier like s3://
        for remote data. Can also be any MutableMapping instance, which should
        be serializable if used in multiple processes.
    component: str or None
        If the location is a zarr group rather than an array, this is the
        subcomponent that should be loaded, something like ``'foo/bar'``.
    storage_options: dict
        Any additional parameters for the storage backend (ignored for local
        paths)
    chunks: tuple of ints or tuples of ints
        Passed to :func:`dask_array.from_array`, allows setting the chunks on
        initialisation, if the chunking scheme in the on-disc dataset is not
        optimal for the calculations to follow.
    name : str, optional
         An optional keyname for the array.  Defaults to hashing the input
    kwargs:
        Passed to :class:`zarr.core.Array`.
    inline_array : bool, default False
        Whether to inline the zarr Array in the values of the task graph.
        See :meth:`dask_array.from_array` for an explanation.

    See Also
    --------
    from_array
    """
    import zarr

    from dask_array.core import from_array

    storage_options = storage_options or {}
    if isinstance(url, zarr.Array):
        z = url
    elif isinstance(url, (str, os.PathLike)):
        if isinstance(url, os.PathLike):
            url = os.fspath(url)

        zarr_store = _setup_zarr_store(url, storage_options, **kwargs)
        z = zarr.open_array(store=zarr_store, path=component, **kwargs)
    else:
        z = zarr.open_array(store=url, path=component, **kwargs)
    chunks = chunks if chunks is not None else z.chunks
    if name is None:
        name = "from-zarr-" + tokenize(z, component, storage_options, chunks, **kwargs)
    return from_array(z, chunks, name=name, inline_array=inline_array)


def _get_zarr_write_chunks(zarr_array) -> tuple[int, ...]:
    """Get the appropriate chunk shape for writing to a Zarr array.

    For Zarr v3 arrays with sharding, returns the shard shape.
    For arrays without sharding, returns the chunk shape.
    For Zarr v2 arrays, returns the chunk shape.

    Parameters
    ----------
    zarr_array : zarr.Array
        The target zarr array

    Returns
    -------
    tuple
        The chunk shape to use for rechunking the dask array
    """
    # Zarr V3 array with shards
    if hasattr(zarr_array, "shards") and zarr_array.shards is not None:
        return zarr_array.shards
    # Zarr V3 array without shards, or Zarr V2 array
    return zarr_array.chunks


def _check_regular_chunks(chunkset):
    """Check if the chunks are regular

    "Regular" in this context means that along every axis, the chunks all
    have the same size, except the last one, which may be smaller

    Parameters
    ----------
    chunkset: tuple of tuples of ints
        From the ``.chunks`` attribute of an ``Array``

    Returns
    -------
    True if chunkset passes, else False

    Examples
    --------
    >>> _check_regular_chunks(((5, 5),))
    True

    >>> _check_regular_chunks(((3, 3, 3, 1),))
    True

    >>> _check_regular_chunks(((3, 1, 3, 3),))
    False
    """
    for chunks in chunkset:
        if len(chunks) == 1:
            continue
        if len(set(chunks[:-1])) > 1:
            return False
        if chunks[-1] > chunks[0]:
            return False
    return True


def _write_dask_to_existing_zarr(
    url, arr, region, zarr_mem_store_types, compute, return_stored
):
    """Write dask array to existing zarr store.

    Parameters
    ----------
    url: zarr.Array
        The zarr array.
    arr:
        The dask array to be stored
    region: tuple of slices or None
        The region of data that should be written if ``url`` is a zarr.Array.
        Not to be used with other types of ``url``.
    zarr_mem_store_types: tuple[Type[dict] | Type[zarr.storage.MemoryStore] | Type[zarr.storage.KVStore], ...]
        The type of zarr memory store that is allowed.
    compute: bool
        See :func:`~dask_array.store` for more details.
    return_stored: bool
        See :func:`~dask_array.store` for more details.

    Returns
    -------
    If return_stored=True
        tuple of Arrays
    If return_stored=False and compute=True
        None
    If return_stored=False and compute=False
        Delayed
    """
    from dask_array.slicing._utils import new_blockdim, normalize_index

    z = url
    if isinstance(z.store, zarr_mem_store_types):
        try:
            from distributed import default_client

            default_client()
        except (ImportError, ValueError):
            pass
        else:
            raise RuntimeError(
                "Cannot store into in memory Zarr Array using "
                "the distributed scheduler."
            )
    zarr_write_chunks = _get_zarr_write_chunks(z)
    dask_write_chunks = normalize_chunks(
        chunks="auto",
        shape=z.shape,
        dtype=z.dtype,
        previous_chunks=zarr_write_chunks,
    )

    if region is not None:
        index = normalize_index(region, z.shape)
        dask_write_chunks = tuple(
            tuple(new_blockdim(s, c, r))
            for s, c, r in zip(z.shape, dask_write_chunks, index)
        )

    for ax, (dw, zw) in enumerate(
        zip(dask_write_chunks, zarr_write_chunks, strict=True)
    ):
        if len(dw) >= 1:
            nominal_dask_chunk_size = dw[0]
            if not nominal_dask_chunk_size % zw == 0:
                safe_chunk_size = np.prod(zarr_write_chunks) * max(1, z.dtype.itemsize)
                msg = (
                    f"The input Dask array will be rechunked along axis {ax} with chunk size "
                    f"{nominal_dask_chunk_size}, but a chunk size divisible by {zw} is "
                    f"required for Dask to write safely to the Zarr array {z}. "
                    "To avoid risk of data loss when writing to this Zarr array, set the "
                    '"array.chunk-size" configuration parameter to at least the size in'
                    " bytes of a single on-disk "
                    f"chunk (or shard) of the Zarr array, which in this case is "
                    f"{safe_chunk_size} bytes. "
                    f'E.g., dask.config.set({{"array.chunk-size": {safe_chunk_size}}})'
                )

                warnings.warn(
                    msg,
                    PerformanceWarning,
                    stacklevel=3,
                )
                break

    arr = arr.rechunk(dask_write_chunks)

    if region is not None:
        regions = [region]
    else:
        regions = None

    return arr.store(
        z, lock=False, regions=regions, compute=compute, return_stored=return_stored
    )


def to_zarr(
    arr,
    url,
    component=None,
    storage_options=None,
    region=None,
    compute=True,
    return_stored=False,
    zarr_array_kwargs=None,
    zarr_read_kwargs=None,
    **kwargs,
):
    """Save array to the zarr storage format

    See https://zarr.readthedocs.io for details about the format.

    Parameters
    ----------
    arr: dask.array
        Data to store
    url: Zarr Array or str or MutableMapping
        Location of the data. A URL can include a protocol specifier like s3://
        for remote data. Can also be any MutableMapping instance, which should
        be serializable if used in multiple processes.
    component: str or None
        If the location is a zarr group rather than an array, this is the
        subcomponent that should be created/over-written. If both `component`
        and 'name' in `zarr_array_kwargs` are specified, `component` takes
        precedence. This will change in a future version.
    storage_options: dict
        Any additional parameters for the storage backend (ignored for local
        paths)
    overwrite: bool
        If given array already exists, overwrite=False will cause an error,
        where overwrite=True will replace the existing data. Deprecated, please
        add to zarr_kwargs
    region: tuple of slices or None
        The region of data that should be written if ``url`` is a zarr.Array.
        Not to be used with other types of ``url``.
    compute: bool
        See :func:`~dask_array.store` for more details.
    return_stored: bool
        See :func:`~dask_array.store` for more details.
    zarr_array_kwargs: dict or None
        Keyword arguments passed to :func:`zarr.create_array` (for zarr v3) or
        :func:`zarr.create` (for zarr v2). This function automatically sets
        ``shape``, ``chunks``, and ``dtype`` based on the dask array, but these
        can be overridden.

        Common options include:

        - ``compressor``: Compression algorithm (e.g., ``zarr.Blosc()``)
        - ``filters``: List of filters to apply
        - ``fill_value``: Value to use for uninitialized portions
        - ``order``: Memory layout ('C' or 'F')
        - ``dimension_separator``: Separator for chunk keys ('/' or '.')

        For the complete list of available arguments, see the zarr documentation:

        - zarr v3: https://zarr.readthedocs.io/en/stable/api/zarr/index.html#zarr.create_array
        - zarr v2: https://zarr.readthedocs.io/en/stable/api/core.html#zarr.create
    zarr_read_kwargs: dict or None
        Keyword arguments passed to the storage backend when creating a zarr
        store from a URL string. Only used when ``url`` is a string (not when
        ``url`` is already a zarr.Array or MutableMapping instance).

        Common options include:

        - ``mode``: File access mode. Options include:
            - ``'r'``: Read-only, must exist
            - ``'r+'``: Read/write, must exist
            - ``'a'``: Read/write, create if doesn't exist (default)
            - ``'w'``: Create, remove existing data if present
            - ``'w-'``: Create, fail if exists
        - ``read_only``: If True, open the store in read-only mode (alternative to ``mode='r'``)

        Additional backend-specific options may be available depending on the
        storage system (e.g., fsspec parameters for cloud storage).
    **kwargs:
        .. deprecated:: 2025.12.0
            Passing storage-related arguments via **kwargs is deprecated.
            Please use the ``zarr_read_kwargs`` parameter instead.

    Raises
    ------
    ValueError
        If ``arr`` has unknown chunk sizes, which is not supported by Zarr.
        If ``region`` is specified and ``url`` is not a zarr.Array

    See Also
    --------
    dask_array.store
    dask_array.Array.compute_chunk_sizes

    """
    import zarr

    if np.isnan(arr.shape).any():
        raise ValueError(
            "Saving a dask array with unknown chunk sizes is not "
            f"currently supported by Zarr.{unknown_chunk_message}"
        )

    zarr_array_kwargs = {} if zarr_array_kwargs is None else dict(zarr_array_kwargs)
    if component is not None and "name" in zarr_array_kwargs:
        raise ValueError(
            "Cannot specify both 'component' and 'name' in zarr_array_kwargs. Please use 'name' in "
            "zarr_array_kwargs"
        )

    if kwargs:
        warnings.warn(
            "Passing storage-related arguments via **kwargs is deprecated. "
            "Please use the 'zarr_store_kwargs' parameter instead. **kwargs will be "
            "removed in a future version.",
            FutureWarning,
            stacklevel=2,
        )
        if zarr_read_kwargs is None:
            zarr_read_kwargs = kwargs
        else:
            zarr_read_kwargs = {**kwargs, **zarr_read_kwargs}

    if _zarr_v3():
        zarr_mem_store_types = (zarr.storage.MemoryStore,)
    else:
        zarr_mem_store_types = (dict, zarr.storage.MemoryStore, zarr.storage.KVStore)

    if isinstance(url, zarr.Array):
        return _write_dask_to_existing_zarr(
            url, arr, region, zarr_mem_store_types, compute, return_stored
        )

    if not _check_regular_chunks(arr.chunks):
        warnings.warn(
            "The array uses irregular chunk sizes. Rechunking to regular (uniform) chunks "
            "to ensure the data can be written safely. If you want to avoid this automatic "
            "rechunking, manually rechunk the array so that all chunks, except possibly the "
            "final chunk, in each dimension—have the same size (e.g., arr = arr.rechunk(...)).",
            UserWarning,
            stacklevel=2,
        )
        # We almost certainly get here because auto chunking has been used
        # on irregular chunks. The max will then be smaller than auto, so using
        # max is a safe choice
        arr = arr.rechunk(tuple(map(max, arr.chunks)))

    if region is not None:
        raise ValueError("Cannot use `region` keyword when url is not a `zarr.Array`.")

    zarr_read_kwargs = {} if zarr_read_kwargs is None else dict(zarr_read_kwargs)
    zarr_store = _setup_zarr_store(url, storage_options, **zarr_read_kwargs)

    zarr_array_kwargs.setdefault("shape", arr.shape)
    zarr_array_kwargs.setdefault("chunks", tuple(c[0] for c in arr.chunks))
    zarr_array_kwargs.setdefault("dtype", arr.dtype)

    array_name = component or zarr_array_kwargs.pop("name", None)
    if _zarr_v3():
        root = zarr.open_group(store=zarr_store, mode="a") if array_name else None
        if array_name:
            z = root.create_array(name=array_name, **zarr_array_kwargs)
        else:
            zarr_array_kwargs["store"] = zarr_store
            z = zarr.create_array(**zarr_array_kwargs)
    else:
        # TODO: drop this as soon as zarr v2 gets dropped.
        # https://github.com/dask/dask/issues/12188
        z = zarr.create(
            store=zarr_store,
            path=array_name,
            **zarr_array_kwargs,
        )

    return arr.store(z, lock=False, compute=compute, return_stored=return_stored)
