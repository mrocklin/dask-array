# dask-array

Dask array with query optimization

## Motivation

Dask Array is powerful, but requires expertise to drive effectively.

The `dask-array` project reimplements `dask.array`, but represents your
calculation at a level where it can be optimized intelligently before executed.
This allows the project to reorder and replace calculations to provide the same
result but with a more efficient path.

## Installation

```bash
pip install dask-array
```

## Usage

This project looks and feels like Dask Array

```python
import dask_array as da

x = da.ones((1000, 1000), chunks=(100, 100))

y = x + x.T
result = y[:100, :100]
```

But when you go to compute, your calculation gets rewritten to be more
efficient.  This is apparent if you look at the query structure of the
underlying array.

```python
>>> result.pprint()
Operation             Shape    Bytes   Chunks
  Getitem          (100, 100)   78 kiB  100×100
  └ Add          (1000, 1000)  7.6 MiB  100×100
    ├ Ones       (1000, 1000)  7.6 MiB  100×100
    └ Transpose  (1000, 1000)  7.6 MiB  100×100
      └ Ones     (1000, 1000)  7.6 MiB  100×100
```

This calculation starts from a large array expression, then takes a small
slice. It is more efficient to push that slice into the expression before
building the task graph.

The `optimize` function rewrites things automatically.

```python
>>> result.optimize().pprint()
Operation            Shape   Bytes   Chunks
  FusedBlockwise  (100, 100)  78 kiB  100×100
  └ Ones          (100, 100)  78 kiB  100×100
```

You don't need to call `optimize` though.  Dask `compute`/`persist` machinery
will do this for you.  You just need to change your import.

```python
# import dask.array as da
import dask_array as da
```

## Native Frisky acceleration

The normal Dask scheduler path is pure Python and works without a Rust
toolchain. On platforms with a native wheel, `dask-array` also includes a Rust
accelerator that Frisky can use to submit compact task records instead of
materializing large Python task graphs. This is automatic when computing with a
Frisky scheduler; otherwise computations stay on the standard Dask path.

## Xarray

Dask-array can replace the default dask.array by registering itself as an
Xarray chunk manager.

```python
from dask_array.xarray import register
register()
```

After this all xarray calculations will benefit from query optimization.

Call `register()` before you create any chunked arrays. It is the only thing
that turns this on: installing or importing dask-array leaves xarray alone, so
that adding this package to an environment cannot change how other libraries
that use xarray and dask behave.
