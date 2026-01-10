"""Array creation dispatch for backend-specific array creation.

This module provides `array_creation_dispatch` which is used by random modules
and creation functions to dispatch to backend-specific implementations.
"""

from __future__ import annotations

import numpy as np

from dask.backends import CreationDispatch, DaskBackendEntrypoint


class ArrayBackendEntrypoint(DaskBackendEntrypoint):
    """Dask-Array version of ``DaskBackendEntrypoint``

    See Also
    --------
    NumpyBackendEntrypoint
    """

    @property
    def RandomState(self):
        """Return the backend-specific RandomState class

        For example, the 'numpy' backend simply returns
        ``numpy.random.RandomState``.
        """
        raise NotImplementedError

    @property
    def default_bit_generator(self):
        """Return the default BitGenerator type"""
        raise NotImplementedError

    @staticmethod
    def ones(shape, *, dtype=None, meta=None, **kwargs):
        """Create an array of ones

        Returns a new array having a specified shape and filled
        with ones.
        """
        raise NotImplementedError

    @staticmethod
    def zeros(shape, *, dtype=None, meta=None, **kwargs):
        """Create an array of zeros

        Returns a new array having a specified shape and filled
        with zeros.
        """
        raise NotImplementedError

    @staticmethod
    def empty(shape, *, dtype=None, meta=None, **kwargs):
        """Create an empty array

        Returns an uninitialized array having a specified shape.
        """
        raise NotImplementedError

    @staticmethod
    def full(shape, fill_value, *, dtype=None, meta=None, **kwargs):
        """Create a uniformly filled array

        Returns a new array having a specified shape and filled
        with fill_value.
        """
        raise NotImplementedError

    @staticmethod
    def arange(start, /, stop=None, step=1, *, dtype=None, meta=None, **kwargs):
        """Create an ascending or descending array

        Returns evenly spaced values within the half-open interval
        ``[start, stop)`` as a one-dimensional array.
        """
        raise NotImplementedError


class NumpyBackendEntrypoint(ArrayBackendEntrypoint):
    @property
    def RandomState(self):
        return np.random.RandomState

    @property
    def default_bit_generator(self):
        return np.random.PCG64


array_creation_dispatch = CreationDispatch(
    module_name="array",
    default="numpy",
    entrypoint_class=ArrayBackendEntrypoint,
    name="array_creation_dispatch",
)


array_creation_dispatch.register_backend("numpy", NumpyBackendEntrypoint())
