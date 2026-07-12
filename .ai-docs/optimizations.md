# Optimizations

The optimization system transforms expression trees to reduce computation before generating task graphs.

## Pipeline Overview

```python
def optimize(self, fuse=True):
    expr = self.simplify().lower_completely()
    if fuse:
        expr = expr.fuse()
    return expr
```

**Three phases:**
1. **Simplify**: Apply algebraic transformations recursively
2. **Lower**: Convert abstract expressions to concrete form
3. **Fuse**: Combine adjacent blockwise operations

## Simplification (`_simplify_down`)

Each expression implements `_simplify_down()` to rewrite itself based on its children:

```python
def _simplify_down(self):
    """Return a better expression or None to keep self."""
    # Check for optimization opportunities
    if self.is_identity:
        return self.array
    return None
```

The framework calls this recursively until no more changes occur.

### Pattern 1: Identity Removal

Remove no-op operations:

```python
# Rechunk._simplify_down() in _rechunk.py
def _simplify_down(self):
    if not self.balance and self.chunks == self.array.chunks:
        return self.array  # No-op rechunk
```

**Examples:**
- `Rechunk(x, chunks=x.chunks)` → `x`
- `Transpose(x, axes=(0,1,2))` → `x` (identity permutation)
- `x[:, :]` → `x` (full slice)

### Pattern 2: Nested Fusion

Combine consecutive operations of the same type:

```python
# Rechunk._pushdown() in _rechunk.py, reached via the Pattern-3 gate
if type(self.array) is Rechunk and self.array.method != "p2p":
    return Rechunk(
        self.array.array,  # Skip middle
        self._chunks,      # Use final chunks
        ...
    )
```

**Examples:**
- `Rechunk(Rechunk(x, c1), c2)` → `Rechunk(x, c2)`
- `Slice(Slice(x, i1), i2)` → `Slice(x, fused_index)`
- `Transpose(Transpose(x, a1), a2)` → `Transpose(x, composed)`

Rechunk-Rechunk fusion runs through the sharing gate (Pattern 3), not
`_simplify_down`: bypassing a shared inner rechunk reaches through it to
its input, sidestepping the per-pass rewrite cache that keeps siblings
converging onto one replacement node (e.g. an inner rechunk being absorbed
into a shared read would get duplicated).

### Pattern 3: Pushdown via `_simplify_up`

Children rewrite a slice/rechunk/shuffle parent through their
`_simplify_up` hook, routed through a sharing-aware gate:

```python
# Blockwise._simplify_up() in _blockwise.py
def _simplify_up(self, parent, dependents):
    if isinstance(parent, SliceSlicesIntegers):
        return self._slice_pushdown(parent, dependents)
    if isinstance(parent, Shuffle):
        return self._shuffle_pushdown(parent, dependents)
```

The gates live on `ArrayExpr` (in `_expr.py`) and decline when another
dependent of the child needs it in full — a full consumer materializes the
child anyway, so pushing would duplicate its work:

- `_slice_pushdown` declines when any *other* dependent is not a slice.
  When every dependent is a slice (multi-window selection), all of them
  push and the child is never computed in full anywhere.
- `_rechunk_pushdown` / `_shuffle_pushdown` decline when there is *any*
  other dependent. A pushed rechunk/shuffle re-derives the child in full
  (same elements, different layout), so there is no multi-window sharing
  among pushed copies — even two rechunks pushing would each duplicate the
  whole chain (and, at an IO leaf, the read).

The child implements the actual rewrite in `_accept_slice` /
`_accept_shuffle`; for rechunks the gate calls back into
`Rechunk._pushdown`, which dispatches on the child type
(Transpose/Elemwise/Concatenate/IO).

Pushdown deliberately does NOT dispatch from the pushed node's own
`_simplify_down` — that side cannot see `dependents`, so it would be
sharing-blind (and would defeat the other guards indirectly: a rechunk
splitting a shared node empties the sibling set a slice's gate checks).

## Slice Pushdown

The most important optimization. Reduces data processed by pushing slices toward leaves.

### Into FromArray (IO)

Slices fuse directly into `FromArray`, which serves as a stand-in for most IO operations:

```python
# da.from_array(x, chunks=(10,10))[:5] → FromArray with sliced source
```

This is critical because it means slicing happens at read time, not after loading entire arrays.

### Through Elemwise

```python
# (x + y)[:5] → x[:5] + y[:5]
```

Maps output slice to each input dimension.

### Through Blockwise

**Fine-grained** (no adjust_chunks):
1. Convert integer indices to size-1 slices
2. Map output indices to input dimensions
3. Apply slices to all inputs
4. If integers were used, wrap with extraction

**Coarse-grained** (with adjust_chunks):
1. Find which OUTPUT blocks the slice needs
2. Map to corresponding INPUT blocks
3. Create block-aligned input slices
4. Wrap output with adjusted slice if needed

```python
def _accept_slice(self, slice_expr):
    if needs_coarse:
        return self._accept_slice_coarse(slice_expr, full_index, adjust_chunks)
    # Fine-grained pushdown...
```

### Through Concatenate

```python
# concat([a, b, c])[0:5] → a[0:5]  (if slice fits in first)
# concat([a, b, c])[0:15] → concat([a, b[:5]])  (spans multiple)
```

### Through Transpose

```python
# x.T[:5, :10] → x[:10, :5].T
```

Reorders slice indices through the permutation.

## Rechunk Pushdown (`_rechunk.py`)

Push rechunk toward leaves to apply at lowest level. Dispatched from the
child's `_simplify_up` through the `_rechunk_pushdown` gate (Pattern 3),
which then calls:

```python
def _pushdown(self):  # on Rechunk; sharing already checked by the gate
    # Rechunk(Rechunk(x)) → single Rechunk to final chunks

    # Rechunk(Transpose) → Transpose(rechunked input)
    if isinstance(self.array, Transpose):
        return self._pushdown_through_transpose()

    # Rechunk(Elemwise) → Elemwise(rechunked inputs)
    if isinstance(self.array, Elemwise):
        return self._pushdown_through_elemwise()

    # Rechunk(Concatenate) → Concatenate(rechunked inputs)
    if isinstance(self.array, Concatenate):
        return self._pushdown_through_concatenate()

    # Rechunk(IO) → IO with new chunks (via _accept_rechunk)
    ...
```

Only no-op removal stays in `Rechunk._simplify_down`: it substitutes the
existing child node, so sharing can only increase.

The IO pushdown (`Rechunk(FromArray)` → read at the target chunks, via
`_pushdown_into_io`) also runs from `Rechunk._lower()`: rechunks inserted
*during* lowering — chunk unification, reshape, overlap — arrive after
simplify has finished, and would otherwise lower to a `TasksRechunk` that
moves every block when reading at the target layout is free.  (The lowering
path has no `dependents` context, so it cannot consult the sharing gate —
lower-born rechunks have a single consumer by construction, but their IO
child could in principle be shared, in which case the push re-reads it.)

## Rechunk Insertion (chunk unification, `_expr.py`)

`unify_chunks_expr` is where the optimizer *inserts* rechunks nobody asked
for, and the `auto` policy decides their direction per dimension by cost
(`moved_fraction`, the min-model of `Rechunk.transfer_bytes`):

- **nested layouts** merge up to the coarsest operand, unless the moved
  bytes exceed `_MERGE_COST_RATIO` × the bytes already at that layout
  (then refine — splits only);
- **interleaved layouts** (boundaries that don't nest — the `x + roll(x, s)`
  pattern) realign the misaligned operands to an operand's existing grid
  (fewest blocks first) under the same ratio, instead of manufacturing the
  finest common refinement, which would double the block count with sliver
  chunks that every downstream op inherits.

`bench/bench_rechunk_insertion.py` measures these choices against
hand-inserted alternatives; `bench/bench_unify_policy.py` covers the
merge-vs-refine direction.

## Blockwise Fusion (`_blockwise.py`)

Combines adjacent blockwise operations into single fused tasks.

### Algorithm

1. **Build dependency graph** of fusable operations
2. **Find roots** (operations with no fusable dependents)
3. **Group connected components** working down from roots
4. **Detect conflicts** via symbolic mapping
5. **Create FusedBlockwise** for each group

### Fusability Requirements

```python
@property
def _is_blockwise_fusable(self):
    if self.concatenate:
        return False  # Can't fuse with concatenate
    if any(isinstance(op, Delayed) for op in self.operands):
        return False  # Can't fuse with Delayed
    # Check contracted dimensions have single blocks
    ...
```

### Conflict Detection

Detects when same expression is accessed via incompatible paths:

```python
# a + a.T - conflict!
# Direct: a[i, j]
# Transposed: a[j, i]
```

Uses symbolic dimension tracing (not sampling) to detect conflicts reliably.

## Testing Optimizations

**Always use structural equality for optimization tests.** See [testing.md](./testing.md) for details.

### Core Test Pattern

```python
def test_slice_through_elemwise():
    """(x + y)[:5] should optimize to x[:5] + y[:5]"""
    x = da.ones((100, 100), chunks=(10, 10))
    y = da.ones((100, 100), chunks=(10, 10))

    result = (x + y)[:5]
    expected = x[:5] + y[:5]

    # 1. Structural equality - verifies optimization happened
    assert result.expr.simplify()._name == expected.expr.simplify()._name

    # 2. Value correctness - ensures no breakage
    assert_eq(result, expected)
```

### Why `.simplify()` Both Sides

The expected expression often contains optimizable patterns too:

```python
# x[:5] + y[:5] might simplify further (e.g., slices into from_array)
assert result.expr.simplify()._name == expected.expr.simplify()._name
```

### Task Count Verification

```python
def test_optimization_reduces_tasks():
    x = da.from_array(np.ones((100, 100)), chunks=(10, 10))

    full_tasks = len(x.sum(axis=0).optimize().__dask_graph__())
    sliced_tasks = len(x.sum(axis=0)[:5].optimize().__dask_graph__())

    assert sliced_tasks < full_tasks
```

## Adding New Optimizations

### 1. Identify the Pattern

```python
# What should optimize?
Slice(Transpose(x)) → Transpose(Slice(x))
```

### 2. Implement in `_simplify_down`

Add the optimization logic directly in the expression's `_simplify_down` method:

```python
def _simplify_down(self):
    # Check if parent is a slice we can push through
    # (The framework provides access to children via self.array, etc.)

    if isinstance(self.array, SomeType):
        # Transform and return new expression
        return transformed_expr
    return None
```

Slice, rechunk, and shuffle pushdown specifically must go through
`_simplify_up` + their `ArrayExpr` gates (see Pattern 3) so the sharing
guards apply; other rewrites can live wherever clearest.

### 3. Write Tests (TDD)

```python
def test_slice_through_transpose():
    x = da.ones((10, 20), chunks=(5, 10))

    result = x.T[:5]
    expected = x[:, :5].T

    assert result.expr.simplify()._name == expected.expr.simplify()._name
    assert_eq(result, expected)
```

### 4. Use `substitute_parameters` When Appropriate

When a superclass has many attributes but subclasses may not define the same ones, use `substitute_parameters` to preserve the subclass type:

```python
# Problem: Blockwise has 11+ parameters, subclasses like Transpose have fewer
# Manually constructing would require knowing all parent parameters

# Solution: substitute_parameters preserves the type and all other attributes
sliced_input = new_collection(self.array)[slices]
result = self.substitute_parameters({"array": sliced_input.expr})
```

This is especially useful when pushing operations through expression hierarchies.

## Current Pushdown Status

### Slice Through...

| Target | Status | Location |
|--------|--------|----------|
| Slice | Done | Fuses nested slices |
| Elemwise | Done | `_blockwise.py` |
| Blockwise | Done | `_blockwise.py` |
| Transpose | Done | Maps indices |
| Concatenate | Done | `slicing/_basic.py` |
| BroadcastTo | Done | |
| FromArray (IO) | Done | |

### Rechunk Through...

| Target | Status | Location |
|--------|--------|----------|
| Rechunk | Done | `_rechunk.py` |
| Transpose | Done | `_rechunk.py` |
| Elemwise | Done | `_rechunk.py` |
| Concatenate | Done | Concat axis redistributes to (target ∩ part), residual seam merge |

## Key Files

| File | Purpose |
|------|---------|
| `_expr.py` | `optimize()` entry point |
| `_blockwise.py` | Slice through Blockwise, fusion algorithm |
| `_rechunk.py` | Rechunk simplification |
| `slicing/_basic.py` | Slice expression + pushdown |
