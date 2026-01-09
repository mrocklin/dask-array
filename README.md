# dask-array

Expression-based dask array implementation.

## Installation

```bash
pip install dask-array
```

## Usage

```python
import dask_array as da

x = da.ones((1000, 1000), chunks=(100, 100))
result = x.sum().compute()
```
