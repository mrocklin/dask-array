"""Engagement spy shared by the live-cluster parity harnesses.

Frisky's patched ``dask.compute`` / ``dask.persist`` try two Frisky paths in
order before falling back to stock dask:

  1. ``expression`` — ``_frisky_compute_via_expression`` /
     ``_frisky_persist_via_expression``: cloudpickle the leaf expressions and
     expand them scheduler-side. The preferred path; it handles the vast
     majority of computes on the current stack.
  2. ``records``    — ``_frisky_compute_collections`` /
     ``_frisky_persist_collections``: build flat task records on the client
     and submit those.

A harness that watches only the records helper reads ~zero engagement even
when Frisky handled every compute, so this spy wraps all four helpers. A
helper "handled" a submission exactly when it returns non-None (returning
None is its fall-through signal).

The wrappers take ``*args, **kwargs`` because frisky threads extra positional
context through these helpers (``fallback_reasons`` today); the old
fixed-arity spies raised TypeError from inside the patched compute when the
signatures grew.

Usage::

    from _spy import frisky_spy

    with frisky_spy() as spy:
        with LocalCluster(n_workers=2) as cluster, Client(cluster.scheduler):
            before = spy.snapshot()
            dask.compute(x)
            spy.path_since(before)  # "expression" | "records" | "none"
"""

import contextlib

import frisky.dask as fdask

_HELPERS = {
    "expression": (
        "_frisky_compute_via_expression",
        "_frisky_persist_via_expression",
    ),
    "records": (
        "_frisky_compute_collections",
        "_frisky_persist_collections",
    ),
}

#: Fixed-width tags for per-case output lines, keyed by ``path_since`` result.
TAGS = {"expression": "expr", "records": "rec ", "none": "dask"}


class _Spy:
    def __init__(self):
        self.counts = {"expression": 0, "records": 0}

    def snapshot(self):
        return dict(self.counts)

    def path_since(self, snapshot):
        """Which Frisky path handled a submission since ``snapshot``:
        ``"expression"``, ``"records"``, or ``"none"`` (fell back to dask)."""
        for path in ("expression", "records"):
            if self.counts[path] > snapshot[path]:
                return path
        return "none"

    def engaged_since(self, snapshot):
        return self.path_since(snapshot) != "none"


@contextlib.contextmanager
def frisky_spy():
    """Patch all four Frisky compute/persist helpers, yield a counter, restore."""
    spy = _Spy()
    originals = {}

    def wrap(path, orig):
        def wrapper(*args, **kwargs):
            out = orig(*args, **kwargs)
            spy.counts[path] += out is not None
            return out

        return wrapper

    for path, names in _HELPERS.items():
        for name in names:
            originals[name] = getattr(fdask, name)
            setattr(fdask, name, wrap(path, originals[name]))
    try:
        yield spy
    finally:
        for name, orig in originals.items():
            setattr(fdask, name, orig)
