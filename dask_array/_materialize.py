"""Materialization: turn an expression into a task graph.

``_materialize`` is the single choke point where an ``ArrayExpr`` becomes a
task graph — behind ``ArrayExpr.__dask_graph__``, ``Array._lowered_expr``,
and the Frisky records walks. It optimizes fully (simplify → lower → fuse)
and pins the graph's output keys back to the raw root name (``RootAlias``),
so results always come back under the keys a collection advertised.
"""

from __future__ import annotations

import math
import weakref

from dask import config
from dask_array._expr import ArrayExpr, RootAlias, _chunks_match

# Process-wide, ``_name``-keyed cache of one-pass lowering results, shared across
# every ``_lower`` call.  Lowering is a deterministic, context-free function of a
# node's structure (captured by its ``_name``), so memoizing by ``_name`` is safe.
# This relies on every ``_lower`` override depending only on ``self`` — not on
# config, parent context, or randomness (unlike ``_simplify_up``, or the ``_layer``
# methods that legitimately read config at graph-build time).  A ``_lower`` that
# read config would make this cache serve stale results across config changes.
# The payoff is cross-collection sharing: the 666 quantities of one model Dataset
# share a deep ancestry, and without this each collection re-lowers (and so
# re-tokenizes) that shared subtree from scratch — O(N^2) over the Dataset.  With
# the shared cache a shared subtree lowers once.  Weak values bound the cache to
# lowered expressions that are still live (each is reachable from some collection's
# ``_lowered_expr``), so it self-evicts as collections are dropped.
_LOWER_CACHE: weakref.WeakValueDictionary[str, ArrayExpr] = weakref.WeakValueDictionary()


def _lower(expr, optimize_graph):
    """Simplify (when optimizing) and lower ``expr`` to a concrete form.

    Equivalent to ``expr.simplify().lower_completely()`` but lowering goes
    through the shared ``_LOWER_CACHE`` (see above), so subtrees shared by
    many collections lower once.
    """
    if optimize_graph:
        expr = expr.simplify()
    while True:
        new = expr.lower_once(_LOWER_CACHE)
        if new._name == expr._name:
            return expr
        expr = new


def _materialize(expr, optimize_graph=None):
    """Optimize an expression fully (simplify → lower → fuse) and pin its
    output keys back to the raw root name.

    This is the single place where an expression becomes a task graph — used
    by ``__dask_graph__``, ``__dask_keys__``'s contract, and the Frisky
    records walk. Optimization renames every node it rewrites, but a
    collection's advertised keys are its *raw* root expression's name
    (``Array._name``), assigned at construction and stable forever. So when
    optimization renamed the root, the result is wrapped in a ``RootAlias``
    whose layer aliases ``(raw name, i, ...)`` to the optimized root's keys.
    Results therefore always come back under the keys that were advertised —
    ``persist`` round-trips without any name reconciliation.

    When ``array.optimize-graph`` is True (the default) we ``simplify()`` and
    ``fuse()``; set it False to observe the raw, un-simplified graph (e.g.
    structural tests asserting on task counts). Lowering always runs (a graph
    cannot be built from un-lowered nodes) and goes through the shared
    ``_LOWER_CACHE`` (see above) so a subtree shared by many collections is
    lowered once rather than once per collection.
    """
    if optimize_graph is None:
        optimize_graph = config.get("array.optimize-graph", True)
    if isinstance(expr, RootAlias):
        return expr  # only ever built here, over an already-materialized tree
    name = expr._name
    chunks = expr.chunks

    expr = _lower(expr, optimize_graph)
    if optimize_graph:
        expr = expr.fuse()
    if expr._name != name:
        if not _chunks_match(expr.chunks, chunks):
            # A rewrite chose a different output chunking (e.g. the
            # sliding-window reductions avoid a padding rechunk by emitting
            # coarser blocks). The collection's advertised chunks are a
            # promise, so bridge back to them. Cheap when the rewrite's grid
            # is coarser (pure splits); the real fix is for such rewrites to
            # advertise their chunking at construction time.
            if any(math.isnan(s) for dim in chunks for s in dim):
                raise RuntimeError(
                    f"optimization changed the block structure of {name} "
                    f"({chunks} -> {expr.chunks}) and the advertised chunks "
                    "are unknown, so they cannot be restored"
                )
            expr = _lower(expr.rechunk(chunks), optimize_graph=False)
        if any(node._name == name for node in expr.walk()):
            # The pin's alias keys would collide with that node's own layer.
            # No rewrite embeds its input root today; fail loudly if one ever
            # does rather than silently merging two layers under one name.
            raise RuntimeError(
                f"optimization embedded the original root {name} inside its rewrite; cannot pin output keys"
            )
        expr = RootAlias(expr, name)
    return expr
