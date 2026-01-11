# Expression System

The expression system separates **intent** from **execution**, enabling query optimization before generating task graphs.

## Architecture Overview

```
User code: (x + y).sum(axis=0)
                ↓
Expression tree: Sum(Elemwise(+, x.expr, y.expr), axis=0)
                ↓
Optimization: simplify() → lower_completely() → fuse()
                ↓
Task graph: generated only at compute time
```

### Two-Layer Design

| Layer | Class | Purpose |
|-------|-------|---------|
| Expression | `ArrayExpr` | Represents computation intent, enables optimization |
| Collection | `Array` | User-facing wrapper, provides API methods |

**Key insight**: The `Array` class is a thin wrapper around `ArrayExpr`. Most operations create new expression nodes rather than executing immediately.

## Core Classes

### ArrayExpr (`_expr.py`)

Base class for all array expressions. Inherits from `SingletonExpr` (from `dask._expr`).

```python
class ArrayExpr(SingletonExpr):
    _is_blockwise_fusable = False  # Override in fusable subclasses

    # Key protocol methods
    def optimize(self, fuse=True):
        expr = self.simplify().lower_completely()
        if fuse:
            expr = expr.fuse()
        return expr
```

### Array (`_collection.py`)

User-facing wrapper that delegates to the underlying expression:

```python
class Array(DaskMethodsMixin):
    def __init__(self, expr):
        self._expr = expr

    @property
    def expr(self) -> ArrayExpr:
        return self._expr

    def __dask_graph__(self):
        return self.expr.lower_completely().__dask_graph__()

    def compute(self, **kwargs):
        return DaskMethodsMixin.compute(self.optimize(), **kwargs)

    def optimize(self):
        return new_collection(self.expr.optimize())
```

## Expression Protocol

Every expression class must define:

### Required Attributes

```python
class MyExpr(ArrayExpr):
    _parameters = ["array", "option"]  # Named operands
    _defaults = {"option": None}       # Default values
```

### Required Properties

```python
@cached_property
def chunks(self):
    """Output chunk structure: tuple of tuples."""
    return self.array.chunks  # or compute new chunks

@cached_property
def _meta(self):
    """Metadata (dtype, type info). Used for type inference."""
    return self.array._meta

@cached_property
def _name(self):
    """Unique identifier for this expression node."""
    return f"myexpr-{self.deterministic_token}"
```

### Derived Properties (inherited)

These are computed from the required properties:

- `dtype` - from `_meta.dtype`
- `shape` - from `chunks`
- `ndim` - from `len(shape)`
- `numblocks` - from `tuple(map(len, chunks))`

### Task Generation

```python
def _layer(self) -> dict:
    """Generate task dictionary for all output blocks."""
    dsk = {}
    for block_id in product(*[range(len(c)) for c in self.chunks]):
        key = (self._name,) + block_id
        input_key = (self.array._name,) + block_id
        dsk[key] = Task(key, my_func, TaskRef(input_key))
    return dsk
```

## Expression Composition

Expressions compose through their `operands`. Each expression stores references to its inputs:

```python
# User code
result = (x + y)[:5]

# Creates expression tree:
# SliceSlicesIntegers(
#     Elemwise(op=add, x.expr, y.expr),
#     index=(slice(None, 5),)
# )
```

Access operands by name or index:
```python
expr.array           # Named access via _parameters
expr.operand("array") # Explicit named access
expr.operands[0]      # Positional access
```

## Optimization Methods

Expressions optimize via three methods (see [optimizations.md](./optimizations.md)):

```python
def _simplify_down(self):
    """Rewrite self based on children. Return new expr or None."""
    # Example: identity simplification
    if self.is_identity:
        return self.array
    return None

def _simplify_up(self, parent, dependents):
    """Rewrite self based on parent context."""
    # Less common - used for push-through patterns
    return None

def _lower(self):
    """Convert to concrete form before graph generation."""
    # Example: rechunk inputs to align chunks
    return None
```

## Creating New Expression Types

### Minimal Template

```python
from functools import cached_property
from dask_array._expr import ArrayExpr
from dask._task_spec import Task, TaskRef

class MyOp(ArrayExpr):
    _parameters = ["array", "param"]
    _defaults = {"param": 1}

    @cached_property
    def _name(self):
        return f"myop-{self.deterministic_token}"

    @cached_property
    def _meta(self):
        # Compute or delegate
        return self.array._meta

    @cached_property
    def chunks(self):
        # Same chunks as input, or compute new ones
        return self.array.chunks

    def _layer(self) -> dict:
        dsk = {}
        for block_id in product(*[range(len(c)) for c in self.chunks]):
            key = (self._name,) + block_id
            input_key = (self.array._name,) + block_id
            dsk[key] = Task(key, my_chunk_func, TaskRef(input_key), self.param)
        return dsk
```

### With Optimization

```python
def _simplify_down(self):
    # Identity: MyOp(x, param=default) -> x
    if self.param == self._defaults["param"]:
        return self.array

    # Fusion: MyOp(MyOp(x, p1), p2) -> MyOp(x, combined)
    if isinstance(self.array, MyOp):
        return MyOp(self.array.array, self.param + self.array.param)

    return None
```

### Subclassing Blockwise

For element-aligned operations, extend `Blockwise` or `Elemwise`:

```python
from dask_array._blockwise import Elemwise

class MyElemwise(Elemwise):
    _parameters = ["x", "y"]
    op = staticmethod(np.add)  # Chunk-level function
```

See [blockwise.md](./blockwise.md) for details.

## Key Utilities

### new_collection

Wrap an expression in an Array:
```python
from dask._collections import new_collection
result = new_collection(new_expr)
```

### substitute_parameters

Modify expression while preserving subclass type:
```python
# Instead of manually constructing:
new_expr = expr.substitute_parameters({"array": new_input})
```

## Debugging

### View Expression Tree

```python
arr.pprint()                    # Pretty-print tree
arr.expr.simplify().pprint()    # After optimization
```

### View Task Graph

```python
arr.__dask_graph__()            # Task dictionary
len(arr.optimize().__dask_graph__())  # Task count
```

### Inspect Expression

```python
arr.expr._name                  # Unique name
arr.expr.operands               # All inputs
arr.expr.dependencies()         # Expression dependencies
type(arr.expr)                  # Expression class
```

## Key Files

| File | Purpose |
|------|---------|
| `_expr.py` | Base `ArrayExpr` class, utilities |
| `_collection.py` | `Array` wrapper, method definitions |
| `_blockwise.py` | `Blockwise`, `Elemwise` base classes |

## Anti-patterns

- **Don't** use `from_graph` except for external graph interop
- **Don't** create legacy Array objects and convert them
- **Don't** put function implementations in `_collection.py`
- **Don't** import expression modules at top-level (circular imports)
