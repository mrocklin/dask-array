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
z = y.rechunk((1000, 10))
result = z[:100, :100]
```

But when you go to compute, your calculation gets rewritten to be more
efficient.  This is apparent if you look at the query structure of the
underlying array.

```python
>>> result.pprint()
Operation               Shape    Bytes   Chunks
  Getitem            (100, 100)   78 kiB   100×10
  └ Rechunk        (1000, 1000)  7.6 MiB  1000×10
    └ Add          (1000, 1000)  7.6 MiB  100×100
      ├ Ones       (1000, 1000)  7.6 MiB  100×100
      └ Transpose  (1000, 1000)  7.6 MiB  100×100
        └ Ones     (1000, 1000)  7.6 MiB  100×100
```

This calculation created 7 MiB of data, then did work, then rechunked, then
sliced.  It would have been more efficient to create smaller arrays, and chunk
them immediately in the desired form.

The `optimize` function rewrites things automatically.

```python
>>> result.optimize().pprint()
Operation         Shape   Bytes  Chunks
Add          (100, 100)  78 kiB  100×10
├ Ones       (100, 100)  78 kiB  100×10
└ Transpose  (100, 100)  78 kiB  100×10
  └ Ones     (100, 100)  78 kiB  10×100
```

You don't need to call `optimize` though.  Dask `compute`/`persist` machinery
will do this for you.  You just need to change your import.

```python
# import dask.array as da
import dask_array as da
```

## Xarray

Dask-array can replace the default dask.array by registering itself as an
Xarray chunk manager.

```python
from dask_array.xarray import regstier
register()
```

After this all xarray calculations will benefit from query optimization.
