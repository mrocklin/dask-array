# Blockwise Operations

Blockwise operations apply functions to aligned blocks across multiple arrays. This is the foundation for element-wise operations, broadcasting, and many transformations.

## Core Concept

A blockwise operation maps output blocks to input blocks using symbolic indices:

```
Output indices: (i, j)
Input A indices: (i, j)  # Same block positions
Input B indices: (j,)    # Broadcasts along i
```

For each output block `(i, j)`, the operation:
1. Maps indices to input block coordinates
2. Calls the function on those blocks
3. Produces the output block

## Blockwise Class (`_blockwise.py`)

```python
class Blockwise(ArrayExpr):
    _parameters = [
        "func",           # Function to apply to blocks
        "out_ind",        # Output index pattern (e.g., (0, 1))
        "name",           # Optional name prefix
        "token",          # Optional token for naming
        "dtype",          # Output data type
        "adjust_chunks",  # Dict: index -> chunk size transformation
        "new_axes",       # Dict: new index -> size
        "align_arrays",   # Whether to unify chunks across inputs
        "concatenate",    # Whether to concatenate along contracted dims
        "_meta_provided", # User-provided metadata
        "kwargs",         # Extra kwargs for func
    ]
```

### Key Attributes

**`args`** - Operands as (array, indices) pairs:
```python
@cached_property
def args(self):
    return self.operands[len(self._parameters):]
# Format: [arr1, (0, 1), arr2, (1,), literal, None, ...]
```

**`out_ind`** - Output indices, typically `(0, 1, 2, ...)` for each dimension.

**`adjust_chunks`** - Transform output chunk sizes:
```python
# Make all chunks size 1 on axis 0
adjust_chunks = {0: lambda c: 1}
# Fixed size chunks
adjust_chunks = {0: 10}
```

**`new_axes`** - Add dimensions to output:
```python
new_axes = {2: 100}  # New axis at index 2 with size 100
```

## Index Mapping

### `_idx_to_block(block_id)`

Maps output block coordinates to symbolic index values:

```python
def _idx_to_block(self, block_id: tuple[int, ...]) -> dict:
    idx_to_block = {idx: block_id[dim] for dim, idx in enumerate(self.out_ind)}
    for idx in self.new_axes:
        idx_to_block[idx] = 0  # New axes always use block 0
    return idx_to_block
```

### `_compute_block_id(ind, idx_to_block, numblocks)`

Maps symbolic indices to input block coordinates:

```python
def _compute_block_id(ind, idx_to_block, numblocks):
    result = []
    for dim, i in enumerate(ind):
        if i in idx_to_block:
            result.append(idx_to_block[i] % numblocks[dim])  # Modulo for broadcasting
        elif numblocks[dim] == 1:
            result.append(0)  # Contracted dimension
        else:
            raise ValueError(...)
    return tuple(result)
```

### Example

```python
# Output: (i, j) with shape (10, 20), chunks ((5,5), (10,10))
# Input A: (i, j) with same shape
# Input B: (j,) with shape (20,), chunks ((10,10))

# For output block (1, 0):
idx_to_block = {i: 1, j: 0}

# Input A block: (1, 0)
# Input B block: (0,)  # Only j dimension
```

## Task Generation

### `_task(key, block_id)`

Generates a single task for one output block:

```python
def _task(self, key, block_id: tuple[int, ...]):
    idx_to_block = self._idx_to_block(block_id)

    args = []
    for arr, ind in toolz.partition(2, self.args):
        if ind is None:
            args.append(arr)  # Literal value
        else:
            input_block_id = self._dep_block_id(arr, ind, idx_to_block)
            args.append(TaskRef((arr._name, *input_block_id)))

    return Task(key, self.func, *args, **self.kwargs)
```

## Elemwise Class (`_blockwise.py`)

Simplified Blockwise for element-wise operations with broadcasting:

```python
class Elemwise(Blockwise):
    _parameters = ["op", "dtype", "name", "where", "out", "_user_kwargs"]

    align_arrays = True
    new_axes = {}
    adjust_chunks = None
    concatenate = None
```

### Key Differences from Blockwise

1. **Simplified parameters**: Just `op` instead of full blockwise spec
2. **Auto-computed `out_ind`**: Derived from broadcast shapes
3. **Broadcasting semantics**: Uses `_broadcast_block_id` for dimension alignment

### `out_ind` Property

```python
@property
def out_ind(self):
    shapes = [getattr(arg, "shape", ()) for arg in self.elemwise_args]
    out_ndim = len(broadcast_shapes(*shapes))
    return tuple(range(out_ndim)[::-1]  # Reversed for conventional indexing
```

### Broadcasting Block Mapping

```python
def _broadcast_block_id(numblocks, block_id):
    """Adjust block_id for broadcasting."""
    out_ndim = len(block_id)
    arr_ndim = len(numblocks)
    offset = out_ndim - arr_ndim

    result = []
    for i, nb in enumerate(numblocks):
        out_idx = offset + i
        if nb == 1:
            result.append(0)  # Single block - broadcast
        else:
            result.append(block_id[out_idx])
    return tuple(result)
```

## Fusability

### `_is_blockwise_fusable` Property

Determines if an expression can be fused with others:

```python
@property
def _is_blockwise_fusable(self):
    # Cannot fuse if concatenate is enabled
    if self.concatenate:
        return False

    # Cannot fuse with Delayed operands
    if any(isinstance(op, Delayed) for op in self.operands):
        return False

    # Cannot fuse if contracted dims have multiple blocks
    out_idx_set = set(self.out_ind)
    for arr, ind in toolz.partition(2, self.args):
        if ind is not None and hasattr(arr, "numblocks"):
            for dim, i in enumerate(ind):
                if i not in out_idx_set and arr.numblocks[dim] > 1:
                    return False
    return True
```

## FusedBlockwise (`_blockwise.py`)

Represents multiple fused blockwise operations:

```python
class FusedBlockwise(ArrayExpr):
    _parameters = ["exprs"]  # Tuple of expressions, first is root

    def _task(self, key, block_id):
        expr_block_ids = self._compute_block_ids(block_id)

        # Generate tasks in dependency order (leaves first)
        internal_tasks = []
        for expr in reversed(self.exprs):
            expr_block_id = expr_block_ids[expr._name]
            subname = (expr._name, *expr_block_id)
            t = expr._task(subname, expr_block_id)
            internal_tasks.append(t)

        return Task.fuse(*internal_tasks, key=key)
```

### Block ID Tracing

```python
def _compute_block_ids(self, output_block_id):
    """Trace block coordinates through fused expressions."""
    expr_names = {e._name for e in self.exprs}
    expr_block_ids = {self.exprs[0]._name: output_block_id}

    for expr in self.exprs:
        my_block_id = expr_block_ids[expr._name]
        for dep in expr.dependencies():
            if dep._name in expr_names and dep._name not in expr_block_ids:
                dep_block_id = expr._input_block_id(dep, my_block_id)
                expr_block_ids[dep._name] = dep_block_id

    return expr_block_ids
```

## Common Patterns

### Element-wise Operations

```python
# From elemwise() function
Elemwise(np.add, dtype, name, where=True, out=None, **kwargs, *arrays)
```

### Tensor Operations

```python
# From tensordot
blockwise(
    _tensordot,
    out_index,          # e.g., (0, 1, 2, 3)
    lhs, left_index,    # e.g., (0, 1, 4)
    rhs, right_index,   # e.g., (4, 2, 3)
    dtype=dt,
    concatenate=True,   # Sum over contracted dimension
    adjust_chunks={4: lambda c: 1},
)
```

### Transpose as Blockwise Subclass

```python
class Transpose(Blockwise):
    func = staticmethod(np.transpose)

    @property
    def out_ind(self):
        return self.axes  # Permuted indices

    def _input_block_id(self, dep, block_id):
        # Permute block coordinates
        return tuple(block_id[self._inverse_axes[d]] for d in range(len(block_id)))
```

## Optimization Methods

See [optimizations.md](./optimizations.md) for details.

### `_accept_slice(slice_expr)`

Allows slices to push through blockwise:

```python
def _accept_slice(self, slice_expr):
    # Fine-grained: push slice directly to inputs
    # Coarse-grained: when adjust_chunks is used
    ...
```

### `_accept_shuffle(shuffle_expr)`

Allows shuffles to push through:

```python
def _accept_shuffle(self, shuffle_expr):
    # Map shuffle axis through blockwise indices
    ...
```

## Fusion Algorithm (`_blockwise.py`)

1. **Build dependency graph**: Find all fusable operations
2. **Find roots**: Operations with no fusable dependents
3. **Group connected components**: Starting from roots, collect fusable dependencies
4. **Detect conflicts**: Use symbolic mapping to find conflicting access patterns (e.g., `a + a.T`)
5. **Create FusedBlockwise**: Wrap each group

### Conflict Detection

```python
def _remove_conflicting_exprs(group):
    """Remove expressions accessed with conflicting block patterns.

    Example conflict: a + a.T
    - Direct: a[i, j]
    - Transposed: a[j, i]
    """
    # Uses symbolic dimension tracing, not sampling
    ...
```

## Key Files

| File | Purpose |
|------|---------|
| `_blockwise.py` | Blockwise, Elemwise, FusedBlockwise, fusion |
| `core/_blockwise_funcs.py` | Public `blockwise()`, `elemwise()` functions |
