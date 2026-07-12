"""from_array data-source layers.

Unlike the computed layers (blockwise, reduction, rechunk, ...), from_array is a
data *source*: each output block is a slice of the backing array, so the
per-task work is Python slicing either way. ``FromArrayGetterLayer`` (array-like
sources) is Rust-backed only for the O(n_tasks) *expansion*;
``FromArrayLayer`` (plain ndarrays, eager slices) is pure Python. (I/O / source
layers are the seam where records originate in Python.)

``FromArrayLayer`` mirrors dask's plain-ndarray path: eager per-block slices as data nodes
(`{(name, *idx): array[slc]}`; a single block is copied). The dask path emits the
bare values; the records path wraps each in a `toolz.identity` task, since Frisky
submits tasks, not bare data nodes. Any plain ndarray without a lock is handled
here (dask itself eager-slices that case and ignores getitem/inline_array/
asarray), including a pushed-in `_region` (a deferred slice) — each block's slice
is offset by the region start, exactly as the legacy `_layer`. non-ndarray / lock
fall back to dask — see `FromArray._frisky_layer`.
"""

from __future__ import annotations

from itertools import product

import toolz

from dask_array._core_utils import slices_from_chunks
from dask_array._frisky.base import Layer, _rust


class FromArrayGetterLayer(Layer):
    """Array-like (zarr / h5py / icechunk / any slicing target) from_array.

    The native counterpart to dask's ``graph_from_arraylike(inline_array=...)``:
    the source array is placed once as a data node (``original-<name>``) and each
    block is ``getter(array_ref, block_slice)``. The Rust layer does the
    O(n_tasks) cartesian-product expansion (key strings + per-block slices)
    without re-lowering a legacy dict — the ~8x the generic ``GraphRecordsLayer``
    pays. ``FromArray._frisky_layer`` builds this for the array-like case; the
    plain-ndarray case stays on the eager-slice ``FromArrayLayer`` below.

    Faithful only for ``getitem`` = ``getter``/``getter_nofancy`` and no lock;
    a user-supplied custom getter or a lock falls back upstream. The binary
    ``to_records_chunk`` declines (``NotImplementedError``, so the walk uses
    ``to_task_records``) for the inline-array and 5-arg-getter cases, which a
    shared chunk can't express.
    """

    def __init__(self, name, array, chunks, getitem, asarray, inline_array, region):
        # Per-dim (chunk_sizes, region_offset). graph_from_arraylike offsets each
        # block slice by the region start; offset is 0 when there is no region.
        if region is not None:
            offsets = tuple(slc.indices(dim)[0] for slc, dim in zip(region, array.shape))
        else:
            offsets = (0,) * len(chunks)
        dims = [(list(sizes), int(off)) for sizes, off in zip(chunks, offsets)]
        # Mirror graph_from_arraylike's kwargs branch: it only passes asarray/lock
        # (the 5-arg getter call) when getitem takes both keywords and the common
        # case doesn't hold. lock is always False here (lock falls back upstream),
        # so the extra args are needed only when asarray is False.
        extra_args = (bool(asarray), False) if not asarray else None
        self._name = name
        self._array = array
        self._rust = _rust.FromArrayGetterLayer(name, array, getitem, dims, bool(inline_array), extra_args)

    def chunk_side_records(self):
        """The source array as a single plain "holder" record, emitted alongside
        the binary getter chunk. The chunk's ``Dep`` slots reference it by the key
        ``('original-<name>',)`` (a Dep with empty coord), so the array is shipped
        once, not embedded per block."""
        holder = f"original-{self._name}"
        return [(str((holder,)), toolz.identity, (self._array,), {}, [])]


class FromArrayLayer:
    def __init__(self, name, array, chunks, region=None):
        self._name = name
        self.array = array
        self.chunks = chunks
        self.region = region

    def _blocks(self):
        """Yield `(block_index, value)` per output block, matching dask's
        eager-slice (numpy) path: a single block is copied, others are views.

        When a `_region` (deferred slice) is present, the per-block slices are
        offset by the region start, mirroring `FromArray._layer` exactly."""
        region = self.region
        slices = slices_from_chunks(self.chunks)
        if region is not None:
            region_starts = tuple(slc.indices(dim_size)[0] for slc, dim_size in zip(region, self.array.shape))
            slices = [
                tuple(slice(s.start + offset, s.stop + offset, s.step) for s, offset in zip(slc, region_starts))
                for slc in slices
            ]
        indices = product(*(range(len(bds)) for bds in self.chunks))
        single = all(len(c) == 1 for c in self.chunks)
        for idx, slc in zip(indices, slices):
            if single:
                # Single block: copy the (region-)sliced source, matching legacy.
                val = self.array[region].copy() if region is not None else self.array.copy()
            else:
                val = self.array[slc]
            yield idx, val

    def to_dask_graph(self):
        return {(self._name, *idx): val for idx, val in self._blocks()}

    def to_task_records(self):
        return [(str((self._name, *idx)), toolz.identity, (val,), {}, []) for idx, val in self._blocks()]
