"""Base ``Layer``: a thin wrapper over the Rust layer.

The expansion and the converters live in Rust (``dask_array._rust``): each layer
expands (pure Rust) into a neutral form held as a Rust ``Vec``, and the generic
Rust converters turn that into a dask task graph (useful for focused parity
checks), the plain record tuples the Frisky client serializes
(``to_task_records``), or one binary records LAYER chunk
(``to_records_chunk``). Ordinary ``Expr._layer`` implementations stay on the
Python Dask path; these layers are only used by the Frisky graph protocol.

Every module that touches ``dask_array._rust`` must import it from here
(``from dask_array._frisky.base import _rust``) so the build-freshness check
below is guaranteed to have run first — enforced by a source-scan test in
``test_frisky_protocol.py``.
"""

from __future__ import annotations

from typing import Any

from dask_array import _rust

# Fail loudly if a stale native extension is imported (source changed but the
# .so wasn't rebuilt) rather than silently mishandling a changed call. Bump this
# and NATIVE_BUILD_GENERATION in crates/dask-array-python/src/lib.rs together on
# any Rust change. This is a LOCAL build-freshness check, not a wire protocol;
# the Frisky-coordinated version is the records grammar (common::RECORDS_PROTOCOL
# _VERSION), which a plain layer addition does not touch.
_NATIVE_BUILD_GENERATION = 44
if _rust.native_build_generation() != _NATIVE_BUILD_GENERATION:
    raise ImportError(
        f"dask_array._rust is at native build generation {_rust.native_build_generation()}, "
        f"expected {_NATIVE_BUILD_GENERATION}; rebuild the native extension with "
        f"`maturin develop`."
    )


class Layer:
    _rust: Any  # the Rust layer; each subclass builds it in __init__

    def to_dask_graph(self) -> dict:
        return self._rust.to_dask_graph()

    def to_task_records(self):
        """Plain ``(key, func, args, kwargs, deps)`` records, one per task."""
        return self._rust.to_task_records()

    def to_records_chunk(self):
        """One binary records LAYER chunk (the protocol shared with Frisky's
        ``records_proto``), letting Frisky build task specs without materializing
        Python record tuples. Raises ``NotImplementedError`` when the layer can't
        ride the binary grammar — a by-design decline, not a porting gap:
        arbitrary Python literals (e.g. ``FromMapLayer``'s per-block values,
        whose Rust backend has no ``to_records_chunk`` at all), per-task kwargs,
        or an encoding capacity limit (>255-element coords/slots). The walk then
        falls back to ``to_task_records`` for just this layer."""
        fn = getattr(self._rust, "to_records_chunk", None)
        if fn is None:
            raise NotImplementedError
        return fn()
