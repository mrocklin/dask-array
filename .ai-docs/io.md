# IO and FromArray

The IO system handles reading data into dask arrays. `FromArray` is the core pattern that other IO sources build on.

## FromArray (`io/_from_array.py`)

The primary way to create dask arrays from existing array-like objects.

```python
import dask_array as da
import numpy as np

x = np.random.random((1000, 1000))
d = da.from_array(x, chunks=(100, 100))
```

### Parameters

```python
class FromArray(IO):
    _parameters = [
        "array",          # Source array (numpy, h5py, zarr, etc.)
        "_chunks",        # Chunk specification
        "lock",           # Thread synchronization
        "getitem",        # Custom indexing function
        "inline_array",   # Embed array in graph vs reference
        "meta",           # Array metadata
        "asarray",        # Force np.asarray on chunks
        "fancy",          # Source supports fancy indexing
        "_name_override", # Custom name prefix
        "_region",        # Deferred slice region (internal)
    ]
```

### Key Flags

```python
_can_rechunk_pushdown = True  # Rechunking can be pushed into FromArray
_slice_pushdown = True        # Slicing can be deferred via _region
```

## Slice Fusion into FromArray

This is a critical optimization. Slices don't execute immediately—they're recorded and applied at read time:

```python
# User code
x = da.from_array(big_array, chunks=(100, 100))
y = x[:500, :500]

# After optimization: FromArray with _region
# Only reads the needed portion, not the full array
```

### How It Works

`FromArray._accept_slice()` records slices in `_region`:

```python
def _accept_slice(self, slice_expr):
    # Validate: only basic slices + integers
    # Compose new region with existing region
    # Compute new chunks for sliced region
    # Return new FromArray with _region set
```

When `_layer()` generates tasks, it applies `_region` offsets:

```python
if region is not None:
    region_starts = tuple(slc.indices(dim)[0] for slc, dim in zip(region, shape))
    slices = [
        tuple(slice(s.start + offset, s.stop + offset, s.step) for s, offset in zip(slc, region_starts))
        for slc in slices
    ]
```

## The `inline_array` Parameter

Controls how the source array is stored in the task graph.

**`inline_array=False` (default):**
```python
# Array stored once, referenced by key
{
    ("array-abc123",): big_array,
    ("fromarray-abc123", 0, 0): (getter, TaskRef(("array-abc123",)), (slice(0,100), slice(0,100))),
    ("fromarray-abc123", 0, 1): (getter, TaskRef(("array-abc123",)), (slice(0,100), slice(100,200))),
    ...
}
```
- Single serialization of array
- Good for: large arrays, h5py, zarr, expensive-to-serialize objects

**`inline_array=True`:**
```python
# Array embedded in each task
{
    ("fromarray-abc123", 0, 0): (getter, big_array, (slice(0,100), slice(0,100))),
    ("fromarray-abc123", 0, 1): (getter, big_array, (slice(0,100), slice(100,200))),
    ...
}
```
- Array serialized per task
- Good for: small arrays, when scheduler needs to move inputs

## Task Generation Paths

`FromArray._layer()` has different paths based on array type:

### Path A: NumPy, Multi-block, No Lock
Eagerly slices at graph construction to prevent memory issues:

```python
values = [self.array[slc] for slc in slices]
dsk = dict(zip(keys, values))
```

### Path B: NumPy, Single Block, No Lock
Copies the array (or region) at graph time:

```python
dsk = {(self._name, 0, 0): self.array[region].copy()}
```

### Path C: Non-NumPy (h5py, zarr, etc.)
Uses getter functions with deferred access:

```python
dsk = {
    k: (getter, TaskRef(arr_key), slc, asarray, lock)
    for k, slc in zip(keys, slices)
}
```

## Chunk Specification

`normalize_chunks()` handles various formats:

```python
# Integer: uniform chunks
da.from_array(x, chunks=100)  # → ((100, 100, ...), (100, 100, ...))

# Tuple: per-dimension
da.from_array(x, chunks=(100, 200))

# Tuple of tuples: explicit
da.from_array(x, chunks=((100, 100, 50), (200, 200)))

# Dict: named dimensions
da.from_array(x, chunks={0: 100, 1: 200})

# "auto": use config + dtype to determine
da.from_array(x, chunks="auto")

# -1 or None: full dimension
da.from_array(x, chunks=(100, -1))  # Full columns
```

### Zarr Alignment

For zarr arrays, respects existing chunk boundaries:

```python
# Uses zarr's chunks as previous_chunks for alignment
previous_chunks = getattr(self.array, "chunks", None)
if hasattr(self.array, "shards") and self.array.shards is not None:
    if self.operand("_chunks") == "auto":
        previous_chunks = self.array.shards  # Zarr 3.x shards
```

## Other IO Sources

| Source | File | Notes |
|--------|------|-------|
| Zarr | `io/_zarr.py` | Handles v2/v3, shards, storage options |
| TileDB | `io/_tiledb.py` | Simple wrapper around `from_array` |
| NPY Stack | `io/_from_npy_stack.py` | Loads from directory of `.npy` files |
| FromDelayed | `io/_from_delayed.py` | Single-chunk array from `dask.delayed` |
| FromGraph | `io/_from_graph.py` | Wraps raw task dict/layer |

## Adding a New IO Source

### Template

```python
from dask_array.io._base import IO
from dask_array._core_utils import normalize_chunks, slices_from_chunks
from dask._collections import new_collection

class FromMySource(IO):
    _parameters = ["source", "_chunks", "other_params"]
    _defaults = {"_chunks": "auto"}

    # Enable optimizations
    _can_rechunk_pushdown = True
    _slice_pushdown = True

    @cached_property
    def _meta(self):
        return np.empty((0,) * self.source.ndim, dtype=self.source.dtype)

    @cached_property
    def chunks(self):
        return normalize_chunks(
            self.operand("_chunks"),
            self.source.shape,
            dtype=self.source.dtype,
            previous_chunks=getattr(self.source, "chunks", None),
        )

    def _layer(self):
        keys = product([self._name], *[range(len(c)) for c in self.chunks])
        slices = slices_from_chunks(self.chunks)
        return {
            k: (my_getter, self.source, slc)
            for k, slc in zip(keys, slices)
        }

# Public entry point
def from_mysource(path, chunks=None, **kwargs):
    source = open_mysource(path)
    return new_collection(FromMySource(source, chunks or "auto", **kwargs))
```

### Key Decisions

1. **Inherit from `IO`** (not `ArrayExpr`) for IO-specific behavior
2. **Set `_can_rechunk_pushdown = True`** if chunks don't affect values
3. **Implement `_accept_slice()`** for slice fusion (or set `_slice_pushdown = True` to inherit)
4. **Use `_name_override`** for deterministic naming
5. **Handle locks** if source isn't thread-safe

## Getter Functions (`_core_utils.py`)

Three variants for different use cases:

```python
# Full-featured: handles None, fancy indexing, locks
def getter(arr, index, asarray=True, lock=None):
    ...

# Signals no fancy indexing support
def getter_nofancy(arr, index, asarray=True, lock=None):
    ...

# Optimization-friendly (safe to inline)
def getter_inline(arr, index, asarray=True, lock=None):
    ...
```

## Locking for Thread Safety

For non-thread-safe sources (h5py files, etc.):

```python
# Use lock=True for automatic SerializableLock
d = da.from_array(h5py_dataset, chunks=(100, 100), lock=True)

# Or provide custom lock
import threading
lock = threading.Lock()
d = da.from_array(h5py_dataset, chunks=(100, 100), lock=lock)
```

## Key Files

| File | Purpose |
|------|---------|
| `io/_from_array.py` | `FromArray` class |
| `io/_base.py` | `IO` base class |
| `io/_zarr.py` | Zarr integration |
| `io/_from_npy_stack.py` | NPY stack loading |
| `io/_from_delayed.py` | From `dask.delayed` |
| `io/_from_graph.py` | From raw task graph |
| `core/_conversion.py` | `from_array()` entry point |
| `_core_utils.py` | `getter`, `normalize_chunks`, `slices_from_chunks` |
