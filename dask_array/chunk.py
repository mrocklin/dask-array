"""Public alias for the per-chunk NumPy functions in ``dask_array._chunk``.

Upstream code imports ``dask.array.chunk`` as a module; this keeps
``import dask_array.chunk`` working. The implementation stays in ``_chunk``:
internal modules import it by that name, and keeping it there leaves the
functions' ``__module__`` (and hence task tokens) unchanged.
"""

from dask_array._chunk import *  # noqa: F403
