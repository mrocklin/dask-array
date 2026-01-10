"""NumPy compatibility utilities.

Since dask_array targets numpy >= 2.0, this module provides simplified shims.
"""

from __future__ import annotations

import numpy as np

# Parse numpy version to tuple for comparisons
_np_version = tuple(int(x) for x in np.__version__.split(".")[:2])

# Version flags - NUMPY_GE_200 is always True since we require numpy >= 2.0
NUMPY_GE_200 = True
NUMPY_GE_210 = _np_version >= (2, 1)
NUMPY_GE_220 = _np_version >= (2, 2)
NUMPY_GE_240 = _np_version >= (2, 4)

# These are available directly in numpy >= 2.0
from numpy.exceptions import AxisError, ComplexWarning
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple


class _Recurser:
    """
    Utility class for recursing over nested iterables.

    This was copied almost verbatim from numpy.core.shape_base._Recurser.
    See numpy license at https://github.com/numpy/numpy/blob/master/LICENSE.txt
    """

    def __init__(self, recurse_if):
        self.recurse_if = recurse_if

    def map_reduce(
        self,
        x,
        f_map=lambda x, **kwargs: x,
        f_reduce=lambda x, **kwargs: x,
        f_kwargs=lambda **kwargs: kwargs,
        **kwargs,
    ):
        """
        Iterate over the nested list, applying:
        * ``f_map`` (T -> U) to items
        * ``f_reduce`` (Iterable[U] -> U) to mapped items

        For instance, ``map_reduce([[1, 2], 3, 4])`` is::

            f_reduce([
              f_reduce([
                f_map(1),
                f_map(2)
              ]),
              f_map(3),
              f_map(4)
            ]])


        State can be passed down through the calls with `f_kwargs`,
        to iterables of mapped items. When kwargs are passed, as in
        ``map_reduce([[1, 2], 3, 4], **kw)``, this becomes::

            kw1 = f_kwargs(**kw)
            kw2 = f_kwargs(**kw1)
            f_reduce([
              f_reduce([
                f_map(1), **kw2)
                f_map(2,  **kw2)
              ],      **kw1),
              f_map(3, **kw1),
              f_map(4, **kw1)
            ]],     **kw)
        """

        def f(x, **kwargs):
            if not self.recurse_if(x):
                return f_map(x, **kwargs)
            else:
                next_kwargs = f_kwargs(**kwargs)
                return f_reduce((f(xi, **next_kwargs) for xi in x), **kwargs)

        return f(x, **kwargs)

    def walk(self, x, index=()):
        """
        Iterate over x, yielding (index, value, entering), where

        * ``index``: a tuple of indices up to this point
        * ``value``: equal to ``x[index[0]][...][index[-1]]``. On the first iteration, is
                     ``x`` itself
        * ``entering``: bool. The result of ``recurse_if(value)``
        """
        do_recurse = self.recurse_if(x)
        yield index, x, do_recurse

        if not do_recurse:
            return
        for i, xi in enumerate(x):
            yield from self.walk(xi, index + (i,))


def moveaxis(a, source, destination):
    """Move axes of an array to new positions."""
    source = normalize_axis_tuple(source, a.ndim, "source")
    destination = normalize_axis_tuple(destination, a.ndim, "destination")
    if len(source) != len(destination):
        raise ValueError(
            "`source` and `destination` arguments must have "
            "the same number of elements"
        )

    order = [n for n in range(a.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    result = a.transpose(order)
    return result


def rollaxis(a, axis, start=0):
    """Roll the specified axis backwards, until it lies in a given position."""
    n = a.ndim
    axis = normalize_axis_index(axis, n)
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not (0 <= start < n + 1):
        raise ValueError(msg % ("start", -n, "start", n + 1, start))
    if axis < start:
        start -= 1
    if axis == start:
        return a[...]
    axes = list(range(0, n))
    axes.remove(axis)
    axes.insert(start, axis)
    return a.transpose(axes)
