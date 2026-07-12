"""Array manipulation functions: flip, transpose, reshape, expand_dims, etc."""

# Import from module files
from dask_array.manipulation._expand import (
    atleast_1d,
    atleast_2d,
    atleast_3d,
    expand_dims,
)
from dask_array.manipulation._flip import flip, fliplr, flipud, rot90
from dask_array.manipulation._reshape import ravel, reshape, reshape_blockwise
from dask_array.manipulation._roll import roll
from dask_array.manipulation._transpose import (
    moveaxis,
    rollaxis,
    swapaxes,
    transpose,
)

__all__ = [
    "flip",
    "flipud",
    "fliplr",
    "rot90",
    "swapaxes",
    "moveaxis",
    "rollaxis",
    "transpose",
    "expand_dims",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "roll",
    "reshape",
    "reshape_blockwise",
    "ravel",
]
