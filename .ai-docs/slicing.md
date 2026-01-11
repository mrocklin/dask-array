# Slicing

The slicing system handles array indexing with multiple specialized expression types for different access patterns.

## Slice Expression Types

| Type | Example | Output Chunks | Handles Unknown Chunks |
|------|---------|---------------|------------------------|
| `SliceSlicesIntegers` | `x[5:10, :, 3]` | Computed | Only identity slices |
| `VIndexArray` | `x.vindex[[1,3], [2,4]]` | Computed | Yes (binary search) |
| `BooleanIndexFlattened` | `x[bool_mask]` | All `np.nan` | Creates unknown |
| `TakeUnknownOneChunk` | `x[idx]` (1 unknown chunk) | Unknown | Special case |
| `Blocks` | `x.blocks[0, 1]` | Selected | N/A |

## Entry Point: `__getitem__`

```python
def __getitem__(self, index):
    index2 = normalize_index(index, self.shape)

    # Route based on index type
    if any(isinstance(i, Array) and i.dtype.kind in "iu" for i in index2):
        self, index2 = slice_with_int_dask_array(self, index2)

    if any(isinstance(i, Array) and i.dtype == bool for i in index2):
        self, index2 = slice_with_bool_dask_array(self, index2)

    # Skip if identity
    if all(isinstance(i, slice) and i == slice(None) for i in index2):
        return self

    result = slice_array(self.expr, index2)
    return new_collection(result)
```

## SliceSlicesIntegers (`slicing/_basic.py`)

The most common slice type. Handles basic NumPy-style slicing.

**Parameters:** `["array", "index", "allow_getitem_optimization"]`

### Slice Composition

Consecutive slices fuse into one:

```python
def _simplify_down(self):
    # x[a][b] → x[fuse_slice(a, b)]
    if isinstance(self.array, SliceSlicesIntegers):
        try:
            fused = fuse_slice(self.array.index, self.index)
            return SliceSlicesIntegers(self.array.array, fused, ...)
        except NotImplementedError:
            pass  # Can't fuse, keep separate

    # Delegate to child's _accept_slice if available
    if hasattr(self.array, "_accept_slice"):
        result = self.array._accept_slice(self)
        if result is not None:
            return result
```

### Task Generation

```python
def _layer(self):
    # Compute which blocks each slice affects
    block_slices = [_slice_1d(shape, chunks, idx) for ...]

    for out_name, in_name, slices in zip(...):
        if all(sl == slice(None) for sl in slices):
            dsk[out_name] = Alias(out_name, in_name)  # No-op optimization
        else:
            dsk[out_name] = Task(out_name, getitem, TaskRef(in_name), slices)
```

### Unknown Chunks Constraint

```python
for dim, ind in zip(shape, index):
    if np.isnan(dim) and ind != slice(None):
        raise ValueError(f"Arrays chunk sizes are unknown: {shape}")
```

Only identity slices (`:`) allowed on dimensions with unknown chunks.

## VIndexArray (`slicing/_vindex.py`)

Point-wise vectorized indexing across multiple axes.

**Parameters:** `["array", "dict_indexes", "broadcast_shape", "npoints"]`

```python
# Point-wise indexing: selects (1,2), (3,4) not rectangular region
x.vindex[[1, 3], [2, 4]]
```

### How It Works

1. Index arrays are broadcast together
2. Each output point maps to one input point
3. Binary search finds which input chunks contain each point
4. Two-phase task generation:
   - **Slice phase**: Extract needed portions from input chunks
   - **Merge phase**: Combine into output chunks

### Chunk Strategy

Creates new first dimension with points distributed:

```python
@cached_property
def chunks(self):
    # Non-indexed dimensions keep their chunks
    chunks = [c for i, c in enumerate(self.array.chunks) if i not in axes]
    # New point dimension added at front
    chunks.insert(0, point_chunks)
    return tuple(chunks)
```

## BooleanIndexFlattened (`slicing/_bool_index.py`)

Boolean mask indexing with unknown output size.

**Parameters:** `["array"]`

```python
x[x > 5]  # Output size depends on data values
```

### Creates Unknown Chunks

```python
@cached_property
def chunks(self):
    nblocks = reduce(mul, self.array.numblocks, 1)
    return ((np.nan,) * nblocks,)  # All chunks unknown
```

Output is always 1D with `np.nan` chunk sizes because the number of `True` values isn't known until compute time.

### ChunksOverride

Boolean indexing wraps results with `ChunksOverride` to mark dimensions as unknown:

```python
new_chunks = tuple(
    tuple(np.nan for _ in range(len(c))) if dim in bool_dims else c
    for dim, c in enumerate(out.chunks)
)
result = ChunksOverride(out.expr, new_chunks)
```

## Blocks (`slicing/_blocks.py`)

Block-coordinate indexing via `.blocks` accessor:

```python
x.blocks[0, 1]    # Get block at position (0, 1)
x.blocks[:2, :]   # Get first two block-rows
```

**Implementation:** Creates `Alias` tasks pointing to selected input blocks.

## Utility Functions

### `_slice_1d(dim_size, chunks, slice_obj)`

Maps a 1D slice to affected blocks:

```python
_slice_1d(100, [20, 20, 20, 20, 20], slice(24, 50))
# → {1: slice(4, 20), 2: slice(0, 10)}
# Block 1: elements 4-20, Block 2: elements 0-10
```

### `new_blockdim(dim_size, chunks, slice_obj)`

Computes new chunk sizes after slicing:

```python
new_blockdim(100, [20, 20, 20, 20, 20], slice(24, 50))
# → [16, 10]  # New chunk sizes
```

### `fuse_slice(a, b)`

Composes two consecutive slices:

```python
fuse_slice(slice(1000, 2000), slice(10, 15))
# → slice(1010, 1015)

fuse_slice(slice(0, 100, 2), slice(10, 20))
# → slice(20, 40, 2)
```

### `normalize_slice(slc, dim_size)`

Converts to canonical form:

```python
normalize_slice(slice(0, 10, 1), 10)  # → slice(None)  (identity)
normalize_slice(slice(-3, None), 10)  # → slice(7, None)
```

### `normalize_index(index, shape)`

Preprocessing pipeline:
1. Replace `...` with full slices
2. Add missing trailing colons
3. Convert numpy arrays to lists
4. Handle negative indices
5. Validate bounds

## Slice Pushdown

Slices optimize by pushing through other operations. See [optimizations.md](./optimizations.md).

**Operations supporting `_accept_slice()`:**
- `Blockwise` / `Elemwise`
- `Transpose`
- `Concatenate` / `Stack`
- `BroadcastTo`
- `FromArray` (IO)
- `PartialReduce`
- `Reshape`
- `ExpandDims`

Example transformation:
```python
# Before
(x + y)[:5]

# After pushdown
x[:5] + y[:5]
```

### Slice Fusion into FromArray

The most important pushdown. Slices fuse directly into `FromArray`, which serves as a stand-in for most IO operations:

```python
# Before optimization
x = da.from_array(big_array, chunks=(100, 100))
y = x[:500, :500]

# After optimization: FromArray with _region=(slice(0,500), slice(0,500))
# Only reads the needed portion at compute time
```

This is critical because it means slicing happens at read time, not after loading entire arrays. See [io.md](./io.md) for details on the `_region` mechanism.

## Unknown Chunk Handling Summary

| Operation | With Unknown Chunks |
|-----------|---------------------|
| Basic slice `x[5:10]` | Error (unless identity) |
| Integer index `x[5]` | Error |
| Boolean mask `x[mask]` | Creates unknown output |
| Fancy index `x[[1,2,3]]` | Error (unless single chunk) |
| `.blocks[...]` | Works (block coordinates) |

## Key Files

| File | Purpose |
|------|---------|
| `slicing/_basic.py` | `SliceSlicesIntegers`, `TakeUnknownOneChunk`, utilities |
| `slicing/_vindex.py` | `VIndexArray` for vectorized indexing |
| `slicing/_bool_index.py` | `BooleanIndexFlattened` |
| `slicing/_blocks.py` | `Blocks` for `.blocks` accessor |
| `slicing/_setitem.py` | `SetItem` for `x[idx] = val` |
| `slicing/_squeeze.py` | `Squeeze` for dimension removal |
| `slicing/_utils.py` | `_slice_1d`, `fuse_slice`, `normalize_slice`, etc. |
