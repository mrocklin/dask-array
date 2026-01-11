from __future__ import annotations

from collections.abc import Collection
from threading import Lock
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from dask.delayed import Delayed

from dask.base import named_schedulers
from dask.utils import SerializableLock

from dask_array._utils import is_arraylike
from dask_array.slicing._utils import fuse_slice


def get_scheduler_lock(collection, scheduler):
    """Get an appropriate lock for the given collection and scheduler."""
    if scheduler is None:
        scheduler = collection.__dask_scheduler__
    actual_get = named_schedulers.get(scheduler, scheduler)
    # Only use locks for non-distributed schedulers
    if actual_get is named_schedulers.get("synchronous", None):
        return False
    return SerializableLock()


def load_store_chunk(
    x: Any,
    out: Any,
    index: slice | None,
    region: slice | None,
    lock: Any,
    return_stored: bool,
    load_stored: bool,
) -> Any:
    """
    A function inserted in a Dask graph for storing a chunk.

    Parameters
    ----------
    x: array-like
        An array (potentially a NumPy one)
    out: array-like
        Where to store results.
    index: slice-like
        Where to store result from ``x`` in ``out``.
    lock: Lock-like or False
        Lock to use before writing to ``out``.
    return_stored: bool
        Whether to return ``out``.
    load_stored: bool
        Whether to return the array stored in ``out``.
        Ignored if ``return_stored`` is not ``True``.

    Returns
    -------

    If return_stored=True and load_stored=False
        out
    If return_stored=True and load_stored=True
        out[index]
    If return_stored=False and compute=False
        None

    Examples
    --------

    >>> a = np.ones((5, 6))
    >>> b = np.empty(a.shape)
    >>> load_store_chunk(a, b, (slice(None), slice(None)), None, False, False, False)
    """
    if region:
        # Equivalent to `out[region][index]`
        if index:
            index = fuse_slice(region, index)
        else:
            index = region
    if lock:
        lock.acquire()
    try:
        if x is not None and x.size != 0:
            if is_arraylike(x):
                out[index] = x
            else:
                out[index] = np.asanyarray(x)

        if return_stored and load_stored:
            return out[index]
        elif return_stored and not load_stored:
            return out
        else:
            return None
    finally:
        if lock:
            lock.release()


A = TypeVar("A", bound="ArrayLike")


def load_chunk(out: A, index: slice, lock: Any, region: slice | None) -> A:
    """Load a chunk from an array-like object.

    This is used for loading stored chunks back into dask arrays.
    """
    return load_store_chunk(
        None,
        out=out,
        region=region,
        index=index,
        lock=lock,
        return_stored=True,
        load_stored=True,
    )


def store(
    sources,
    targets,
    lock: bool | Lock = True,
    regions: tuple[slice, ...] | Collection[tuple[slice, ...]] | None = None,
    compute: bool = True,
    return_stored: bool = False,
    load_stored: bool | None = None,
    **kwargs,
):
    """Store dask arrays in array-like objects, overwrite data in target

    This stores dask arrays into object that supports numpy-style setitem
    indexing.  It stores values chunk by chunk so that it does not have to
    fill up memory.  For best performance you can align the block size of
    the storage target with the block size of your array.

    If your data fits in memory then you may prefer calling
    ``np.array(myarray)`` instead.

    Parameters
    ----------

    sources: Array or collection of Arrays
    targets: array-like or Delayed or collection of array-likes and/or Delayeds
        These should support setitem syntax ``target[10:20] = ...``.
        If sources is a single item, targets must be a single item; if sources is a
        collection of arrays, targets must be a matching collection.
    lock: boolean or threading.Lock, optional
        Whether or not to lock the data stores while storing.
        Pass True (lock each file individually), False (don't lock) or a
        particular :class:`threading.Lock` object to be shared among all writes.
    regions: tuple of slices or collection of tuples of slices, optional
        Each ``region`` tuple in ``regions`` should be such that
        ``target[region].shape = source.shape``
        for the corresponding source and target in sources and targets,
        respectively. If this is a tuple, the contents will be assumed to be
        slices, so do not provide a tuple of tuples.
    compute: boolean, optional
        If true compute immediately; return :class:`dask.delayed.Delayed` otherwise.
    return_stored: boolean, optional
        Optionally return the stored result (default False).
    load_stored: boolean, optional
        Optionally return the stored result, loaded in to memory (default None).
        If None, ``load_stored`` is True if ``return_stored`` is True and
        ``compute`` is False. *This is an advanced option.*
        When False, store will return the appropriate ``target`` for each chunk that is stored.
        Directly computing this result is not what you want.
        Instead, you can use the returned ``target`` to execute followup operations to the store.
    kwargs:
        Parameters passed to compute/persist (only used if compute=True)

    Returns
    -------

    If return_stored=True
        tuple of Arrays
    If return_stored=False and compute=True
        None
    If return_stored=False and compute=False
        Delayed

    Examples
    --------

    >>> import h5py  # doctest: +SKIP
    >>> f = h5py.File('myfile.hdf5', mode='a')  # doctest: +SKIP
    >>> dset = f.create_dataset('/data', shape=x.shape,
    ...                                  chunks=x.chunks,
    ...                                  dtype='f8')  # doctest: +SKIP

    >>> store(x, dset)  # doctest: +SKIP

    Alternatively store many arrays at the same time

    >>> store([x, y, z], [dset1, dset2, dset3])  # doctest: +SKIP
    """
    from dask.base import persist
    from dask.layers import ArraySliceDep

    from dask_array._collection import Array
    from dask_array._map_blocks import map_blocks

    if isinstance(sources, Array):
        sources = [sources]
        targets = [targets]
    targets = cast("Collection[ArrayLike | Delayed]", targets)

    if any(not isinstance(s, Array) for s in sources):
        raise ValueError("All sources must be dask array objects")

    if len(sources) != len(targets):
        raise ValueError(f"Different number of sources [{len(sources)}] and targets [{len(targets)}]")

    if isinstance(regions, tuple) or regions is None:
        regions_list = [regions] * len(sources)
    else:
        regions_list = list(regions)
        if len(sources) != len(regions_list):
            raise ValueError(
                f"Different number of sources [{len(sources)}] and "
                f"targets [{len(targets)}] than regions [{len(regions_list)}]"
            )
    del regions

    if load_stored is None:
        load_stored = return_stored and not compute

    if lock is True:
        lock = get_scheduler_lock(Array, kwargs.get("scheduler"))

    arrays = []
    for s, t, r in zip(sources, targets, regions_list):
        slices = ArraySliceDep(s.chunks)
        arrays.append(
            map_blocks(
                load_store_chunk,
                s,
                t,
                slices,
                region=r,
                lock=lock,
                return_stored=return_stored,
                load_stored=load_stored,
                name="store-map",
                meta=s._meta,
            )
        )

    if compute:
        if not return_stored:
            import dask

            dask.compute(arrays, **kwargs)
            return None
        else:
            stored_persisted = persist(*arrays, **kwargs)
            arrays = []
            for s, r in zip(stored_persisted, regions_list):
                slices = ArraySliceDep(s.chunks)
                arrays.append(
                    map_blocks(
                        load_chunk,
                        s,
                        slices,
                        lock=lock,
                        region=r,
                        name="load-stored",
                        meta=s._meta,
                    )
                )
    if len(arrays) == 1:
        return arrays[0]
    return tuple(arrays)


def to_hdf5(filename, *args, chunks=True, **kwargs):
    """Store arrays in HDF5 file

    This saves several dask arrays into several datapaths in an HDF5 file.
    It creates the necessary datasets and handles clean file opening/closing.

    Parameters
    ----------
    chunks: tuple or ``True``
        Chunk shape, or ``True`` to pass the chunks from the dask array.
        Defaults to ``True``.

    Examples
    --------

    >>> da.to_hdf5('myfile.hdf5', '/x', x)  # doctest: +SKIP

    or

    >>> da.to_hdf5('myfile.hdf5', {'/x': x, '/y': y})  # doctest: +SKIP

    Optionally provide arguments as though to ``h5py.File.create_dataset``

    >>> da.to_hdf5('myfile.hdf5', '/x', x, compression='lzf', shuffle=True)  # doctest: +SKIP

    >>> da.to_hdf5('myfile.hdf5', '/x', x, chunks=(10,20,30))  # doctest: +SKIP

    This can also be used as a method on a single Array

    >>> x.to_hdf5('myfile.hdf5', '/x')  # doctest: +SKIP

    See Also
    --------
    da.store
    h5py.File.create_dataset
    """
    from dask_array._collection import Array

    if len(args) == 1 and isinstance(args[0], dict):
        data = args[0]
    elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], Array):
        data = {args[0]: args[1]}
    else:
        raise ValueError("Please provide {'/data/path': array} dictionary")

    import h5py

    with h5py.File(filename, mode="a") as f:
        dsets = [
            f.require_dataset(
                dp,
                shape=x.shape,
                dtype=x.dtype,
                chunks=tuple(c[0] for c in x.chunks) if chunks is True else chunks,
                **kwargs,
            )
            for dp, x in data.items()
        ]
        store(list(data.values()), dsets)
