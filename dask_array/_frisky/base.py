"""Base ``Layer``: a thin wrapper over the Rust layer.

The expansion and both converters now live in Rust (``dask_array._rust``): each
layer expands (pure Rust) into a neutral form held as a Rust ``Vec``, and the
generic Rust converters turn that into either a dask task graph
(``to_dask_graph`` — the correctness/legacy path that ``Expr._layer`` routes
through, validated by the test suite) or the compact form the Frisky client
serializes (``to_frisky_tasks``). Subclasses just build ``self._rust``.
"""

from __future__ import annotations

from typing import Any

from dask_array import _rust

# Fail loudly if a stale native extension is imported (source changed but the
# .so wasn't rebuilt) rather than silently producing wrong tasks. Bump this and
# PROTOCOL_REVISION in crates/dask-array-python/src/lib.rs together.
_PROTOCOL_REVISION = 21
if _rust.protocol_revision() != _PROTOCOL_REVISION:
    raise ImportError(
        f"dask_array._rust is at protocol revision {_rust.protocol_revision()}, "
        f"expected {_PROTOCOL_REVISION}; rebuild the native extension with "
        f"`maturin develop`."
    )


class Layer:
    _rust: Any  # the Rust layer; each subclass builds it in __init__

    def to_dask_graph(self) -> dict:
        return self._rust.to_dask_graph()

    def to_task_records(self):
        """Plain ``(key, func, args, kwargs, deps)`` records, one per task — the
        boring mirror the Frisky client serializes (see ``collect_task_records``)."""
        return self._rust.to_task_records()
