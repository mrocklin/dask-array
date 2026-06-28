from __future__ import annotations

import functools
from itertools import product

import numpy as np

from dask.base import tokenize
from dask.utils import SerializableLock

from dask_array.io._base import IO
from dask_array._core_utils import (
    getter,
    getter_nofancy,
    graph_from_arraylike,
    normalize_chunks,
    slices_from_chunks,
)
from dask_array._utils import meta_from_array


_NUMPY_SLICE_PUSHDOWN_NBYTES_LIMIT = 64 * 1024 * 1024


def _source_storage_chunks(array):
    """Native storage chunk grid backing ``array`` (the unit it reads at).

    Returns the source's ``.shards`` or ``.chunks`` (zarr/h5py read one whole
    chunk at a time), or ``None`` for unchunked sources such as plain ndarrays.

    xarray opens backend arrays behind lazy-indexing adapters
    (``ImplicitToExplicitIndexingAdapter`` -> ... -> ``ZarrArrayWrapper``) that
    wrap the store but do not re-expose its ``.chunks``/``.shards``.  Walk the
    wrapper chain (linked by ``.array``, innermost store held in ``._array``) so
    the storage grid is still found and a sub-native rechunk is not fused below
    it.

    Returns chunk *sizes*, used as a read-tiling unit.  If a wrapper hides a
    deferred slice (e.g. xarray's ``LazilyIndexedArray``) the FromArray's shape
    is in that sliced frame, so reads align to multiples of the chunk size
    rather than to absolute native boundaries -- still one coarse read per
    target block (never worse than fusing below native), just not perfectly
    storage-aligned.  Values are unaffected: the adapter's ``__getitem__``
    applies the hidden transform.
    """
    for _ in range(16):  # bounded; xarray nests only a handful of adapters
        raw = getattr(array, "shards", None) or getattr(array, "chunks", None)
        if raw is not None:
            return raw
        nxt = getattr(array, "array", None)
        if nxt is None:
            nxt = getattr(array, "_array", None)
        if nxt is None or nxt is array:
            return None
        array = nxt
    return None


class FromArray(IO):
    _parameters = [
        "array",
        "_chunks",
        "lock",
        "getitem",
        "inline_array",
        "meta",
        "asarray",
        "fancy",
        "_name_override",
        "_name_is_exact",
        "_region",  # Slice region for pushdown (tuple of slices or None)
    ]
    _defaults = {
        "_chunks": "auto",
        "getitem": None,
        "inline_array": False,
        "meta": None,
        "asarray": None,
        "fancy": True,
        "lock": False,
        "_name_override": None,
        "_name_is_exact": False,
        "_region": None,
    }
    # FromArray reads static data, so rechunk can be pushed in safely
    _can_rechunk_pushdown = True
    # Slicing can be pushed into FromArray by slicing the source array
    _slice_pushdown = True

    @functools.cached_property
    def _name(self):
        prefix = self.operand("_name_override") or "fromarray"
        if self.operand("_name_is_exact"):
            return prefix
        return f"{prefix}-{self.deterministic_token}"

    @functools.cached_property
    def _effective_shape(self):
        """Shape after applying region slice."""
        region = self.operand("_region")
        if region is None:
            return self.array.shape
        # Compute shape from region slices
        return tuple(len(range(*slc.indices(dim_size))) for slc, dim_size in zip(region, self.array.shape))

    @functools.cached_property
    def chunks(self):
        # Normalize chunks lazily - keeps repr compact with user-provided chunks
        # Pass previous_chunks from underlying array (h5py, zarr) for alignment
        previous_chunks = getattr(self.array, "chunks", None)
        # Handle zarr 3.x shards attribute for write alignment
        if hasattr(self.array, "shards") and self.array.shards is not None and self.operand("_chunks") == "auto":
            previous_chunks = self.array.shards
        return normalize_chunks(
            self.operand("_chunks"),
            self._effective_shape,
            dtype=self.array.dtype,
            previous_chunks=previous_chunks,
        )

    @functools.cached_property
    def _meta(self):
        if self.operand("meta") is not None:
            return meta_from_array(self.operand("meta"), ndim=len(self._effective_shape), dtype=self.array.dtype)
        return meta_from_array(self.array, dtype=getattr(self.array, "dtype", None))

    @functools.cached_property
    def asarray_arg(self):
        if self.operand("asarray") is None:
            return not hasattr(self.array, "__array_function__")
        else:
            return self.operand("asarray")

    def _layer(self):
        lock = self.operand("lock")
        region = self.operand("_region")
        # Note: lock=True is already normalized to SerializableLock() in from_array()

        is_ndarray = type(self.array) in (np.ndarray, np.ma.core.MaskedArray)
        is_single_block = all(len(c) == 1 for c in self.chunks)

        # Get slices for chunks (based on effective shape after region)
        slices = slices_from_chunks(self.chunks)

        # If region is set, offset all slices by the region start
        if region is not None:
            region_starts = tuple(slc.indices(dim_size)[0] for slc, dim_size in zip(region, self.array.shape))
            slices = [
                tuple(slice(s.start + offset, s.stop + offset, s.step) for s, offset in zip(slc, region_starts))
                for slc in slices
            ]

        # Always use the getter for h5py etc. Not using isinstance(x, np.ndarray)
        # because np.matrix is a subclass of np.ndarray.
        if is_ndarray and not is_single_block and not lock:
            # eagerly slice numpy arrays to prevent memory blowup
            # GH5367, GH5601
            keys = product([self._name], *(range(len(bds)) for bds in self.chunks))
            values = [self.array[slc] for slc in slices]
            dsk = dict(zip(keys, values))
        elif is_ndarray and is_single_block and not lock:
            # Single block - slice with region (or full array) and copy
            if region is not None:
                dsk = {(self._name,) + (0,) * self.array.ndim: self.array[region].copy()}
            else:
                dsk = {(self._name,) + (0,) * self.array.ndim: self.array.copy()}
        else:
            getitem = self.operand("getitem")
            if getitem is None:
                if self.operand("fancy"):
                    getitem = getter
                else:
                    getitem = getter_nofancy

            # For non-numpy arrays with region, we need custom graph generation
            # to apply the offset slices
            if region is not None:
                keys = list(product([self._name], *(range(len(bds)) for bds in self.chunks)))
                if self.inline_array:
                    dsk = {k: (getitem, self.array, slc, self.asarray_arg, lock) for k, slc in zip(keys, slices)}
                else:
                    # Put array in graph once, reference by key
                    arr_key = ("array-" + self._name,)
                    dsk = {arr_key: self.array}
                    dsk.update({k: (getitem, arr_key, slc, self.asarray_arg, lock) for k, slc in zip(keys, slices)})
            else:
                dsk = graph_from_arraylike(
                    self.array,
                    chunks=self.chunks,
                    shape=self.array.shape,
                    name=self._name,
                    lock=lock,
                    getitem=getitem,
                    asarray=self.asarray_arg,
                    inline_array=self.inline_array,
                    dtype=self.array.dtype,
                )
        return dict(dsk)  # this comes as a legacy HLG for now

    def _frisky_layer(self):
        """from_array as a records-path data source.

        Two native cases, both honoring a pushed-in `_region` (a deferred slice,
        `from_array(arr)[a:b]`):

        - Plain ndarray (no lock): the eager-slice `FromArrayLayer`. dask's own
          `_layer` always takes the eager-slice branch here and ignores
          getitem/inline_array/asarray, so it matches regardless of those operands
          (this unblocks the small inline constant arrays in pad/triu/tril/isin/…).
        - Any other array-like slicing target (zarr/h5py/icechunk/…): the native
          `FromArrayGetterLayer`, mirroring `graph_from_arraylike` — array placed
          once, each block `getter(ref, slice)`.

        A lock, or a user-supplied custom `getitem`, falls back to the dask path
        in `_layer` (the getter layer is faithful only for the default
        getter/getter_nofancy)."""
        from dask_array._frisky import FromArrayGetterLayer, FromArrayLayer

        if self.operand("lock"):
            raise NotImplementedError("from_array: lock is not supported on the records path")

        is_ndarray = type(self.array) in (np.ndarray, np.ma.core.MaskedArray)
        if is_ndarray:
            return FromArrayLayer(self._name, self.array, self.chunks, self.operand("_region"))

        # Array-like getter path. Only the default getter/getter_nofancy are
        # faithfully represented — a custom getitem falls back. fancy selects
        # between them exactly as `_layer` does.
        getitem = self.operand("getitem")
        if getitem is None:
            getitem = getter if self.operand("fancy") else getter_nofancy
        elif getitem not in (getter, getter_nofancy):
            raise NotImplementedError("from_array: custom getitem is not supported on the records path")

        return FromArrayGetterLayer(
            self._name,
            self.array,
            self.chunks,
            getitem,
            self.asarray_arg,
            self.inline_array,
            self.operand("_region"),
        )

    def __str__(self):
        return "FromArray(...)"

    def __dask_tokenize__(self):
        # Cache the token in `_determ_token` (and return it on subsequent
        # calls) exactly like the base `Expr.__dask_tokenize__`. This matters
        # because `_determ_token` is what `__reduce__` pickles and restores: a
        # source array whose own tokenization is not stable across a pickle
        # round-trip (e.g. an xarray/icechunk lazy-indexing adapter, which
        # tokenizes differently in a fresh process) would otherwise produce a
        # *different* token every time a parent expr re-tokenizes this node.
        # That broke expression submission: the client derives the output key
        # from the live array while the scheduler re-derives it from the
        # unpickled array, and the two disagreed, so the gather hung. Honoring
        # the cached/pickled token keeps a parent's view of this node identical
        # to the value baked into our own `_name`.
        if not self._determ_token:
            from dask.tokenize import _tokenize_deterministic

            # Handle non-serializable locks by using their id()
            # Locks are identity-based objects, so using id() is semantically correct
            lock = self.operand("lock")
            if lock and not isinstance(lock, (bool, SerializableLock)):
                lock_token = ("lock-id", id(lock))
            else:
                lock_token = lock

            if self.operand("_name_is_exact"):
                self._determ_token = (type(self), self.operand("_name_override"))
            else:
                operands = [lock_token if p == "lock" else self.operand(p) for p in self._parameters]
                self._determ_token = _tokenize_deterministic(type(self), *operands)
        return self._determ_token

    def _simplify_up(self, parent, dependents):
        """Allow slice operations to push into FromArray."""
        from dask_array.slicing import SliceSlicesIntegers

        if isinstance(parent, SliceSlicesIntegers):
            if not parent.allow_getitem_optimization:
                return None
            return self._accept_slice(parent)
        return None

    def _accept_rechunk(self, chunks, threshold=None, block_size_limit=None, method=None):
        # For chunked stores, only push reads to chunk grids that do not split
        # native storage chunks. Smaller output chunks stay as a Rechunk above
        # the storage-aligned FromArray.
        raw_storage_chunks = _source_storage_chunks(self.array)
        storage_chunks = None
        if raw_storage_chunks is not None:
            try:
                storage_chunks = tuple(int(c) for c in raw_storage_chunks)
            except (TypeError, ValueError):
                storage_chunks = None
            else:
                if len(storage_chunks) != len(self._effective_shape) or any(c <= 0 for c in storage_chunks):
                    storage_chunks = None

        if storage_chunks is not None:
            if any(np.isnan(c) for dim in chunks for c in dim):
                return None

            region = self.operand("_region")
            if region is not None:
                read_chunks = []
                for dim_chunks, storage, slc, dim_size in zip(chunks, storage_chunks, region, self.array.shape):
                    start, stop, step = slc.indices(dim_size)

                    target_matches_storage = start % storage == 0
                    for chunk in dim_chunks[:-1]:
                        if chunk % storage:
                            target_matches_storage = False
                            break
                    if target_matches_storage:
                        last_chunk = dim_chunks[-1]
                        target_matches_storage = last_chunk <= storage or last_chunk % storage == 0

                    if target_matches_storage:
                        read_chunks.append(dim_chunks)
                    else:
                        if step != 1:
                            return None

                        boundaries = [0]
                        first_storage_boundary = ((start + storage - 1) // storage) * storage
                        for boundary in range(first_storage_boundary, stop, storage):
                            if boundary > start:
                                boundaries.append(boundary - start)
                        boundaries.append(stop - start)
                        read_chunks.append(tuple(right - left for left, right in zip(boundaries, boundaries[1:])))

                read_chunks = tuple(read_chunks)
                if read_chunks == chunks:
                    return self._with_chunks(chunks)
                if read_chunks == self.chunks:
                    return None

                from dask_array._rechunk import Rechunk

                return Rechunk(
                    self._with_chunks(read_chunks),
                    chunks,
                    threshold,
                    block_size_limit,
                    False,
                    method,
                )

            respects_storage = True
            for dim_chunks, storage in zip(chunks, storage_chunks):
                boundary = 0
                for chunk in dim_chunks[:-1]:
                    boundary += chunk
                    if boundary % storage:
                        respects_storage = False
                        break
                if not respects_storage:
                    break

            if not respects_storage:
                read_chunk_sizes = []
                for dim_chunks, storage in zip(chunks, storage_chunks):
                    target = max(dim_chunks)
                    chunk_size = max(target, storage)
                    read_chunk_sizes.append(((chunk_size + storage - 1) // storage) * storage)

                read_chunks = normalize_chunks(tuple(read_chunk_sizes), self._effective_shape, dtype=self.array.dtype)
                if read_chunks == self.chunks:
                    return None

                from dask_array._rechunk import Rechunk

                return Rechunk(
                    self._with_chunks(read_chunks),
                    chunks,
                    threshold,
                    block_size_limit,
                    False,
                    method,
                )

        return self._with_chunks(chunks)

    def _with_chunks(self, chunks):
        name = f"{self._name}-rechunk-{tokenize(self.chunks, chunks)}"
        return FromArray(
            self.array,
            chunks,
            lock=self.operand("lock"),
            getitem=self.operand("getitem"),
            inline_array=self.inline_array,
            meta=self.operand("meta"),
            asarray=self.operand("asarray"),
            fancy=self.operand("fancy"),
            _name_override=name,
            _name_is_exact=True,
            _region=self.operand("_region"),
        )

    def _accept_slice(self, slice_expr):
        """Accept a slice by setting region (deferred slice).

        Pushes the slice into the FromArray expression by recording it as a region,
        which is then applied during layer generation.
        """
        from numbers import Integral

        from dask_array.slicing._basic import (
            SliceSlicesIntegers,
            _compose_slices,
            _compute_sliced_chunks,
        )

        index = slice_expr.index

        # Only handle slices and integers (no None/newaxis, no fancy indexing)
        if any(idx is None for idx in index):
            return None
        if any(not isinstance(idx, (slice, Integral)) for idx in index):
            return None
        # Don't push non-unit step slices - _layer doesn't handle them correctly
        if any(isinstance(idx, slice) and idx.step is not None and idx.step != 1 for idx in index):
            return None

        source = self.array
        old_chunks = self.chunks  # Use normalized chunks property
        old_region = self.operand("_region")

        # Pad index to full dimensions
        full_index = index + (slice(None),) * (source.ndim - len(index))

        # Check if any integers are present - they need special handling
        has_integers = any(isinstance(idx, Integral) for idx in full_index)
        region_index = tuple(slice(idx, idx + 1) if isinstance(idx, Integral) else idx for idx in full_index)

        # Compute new region by combining with existing region
        if old_region is not None:
            # Compose slices: new slice is relative to old region
            new_region = tuple(
                _compose_slices(old_slc, new_slc, dim_size)
                for old_slc, new_slc, dim_size in zip(old_region, region_index, source.shape)
            )
        else:
            new_region = region_index

        # Compute new chunks - use same chunk sizes but clipped to new shape
        new_chunks = tuple(
            _compute_sliced_chunks(dim_chunks, slc, dim_size)
            for dim_chunks, slc, dim_size in zip(old_chunks, region_index, self._effective_shape)
        )

        # Name the sliced node from the logical slice (before any NumPy
        # source/region shrinking below) so it stays deterministic.
        name = f"{self._name}-getitem-{tokenize(old_region, region_index, new_region)}"

        is_ndarray = type(source) in (np.ndarray, np.ma.core.MaskedArray)
        region_shape = tuple(len(range(*slc.indices(dim_size))) for slc, dim_size in zip(new_region, source.shape))
        region_nbytes = int(np.prod(region_shape, dtype=object)) * source.dtype.itemsize
        if is_ndarray:
            if region_nbytes == source.nbytes:
                new_region = None
            elif region_nbytes <= _NUMPY_SLICE_PUSHDOWN_NBYTES_LIMIT:
                source = source[new_region].copy()
                new_region = None

        # Create new FromArray with region (deferred slice), except for NumPy
        # arrays where a small slice avoids carrying the full source through
        # later expression submission.
        new_io = FromArray(
            source,
            new_chunks,
            lock=self.operand("lock"),
            getitem=self.operand("getitem"),
            inline_array=self.inline_array,
            meta=self.operand("meta"),
            asarray=self.operand("asarray"),
            fancy=self.operand("fancy"),
            _name_override=name,
            _name_is_exact=True,
            _region=new_region,
        )

        if has_integers:
            extract_index = tuple(0 if isinstance(idx, Integral) else slice(None) for idx in full_index)
            extract_token = "-".join(f"i{idx}" if isinstance(idx, Integral) else "s" for idx in extract_index)
            return SliceSlicesIntegers(
                new_io,
                extract_index,
                False,
                _determ_token=f"{new_io._name}-extract-{extract_token}",
            )

        return new_io
