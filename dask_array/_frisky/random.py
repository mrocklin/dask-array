"""random data-source layer (``da.random.random``, ``.normal``, ``.poisson``, …).

Like ``from_array``, ``Random`` is a *source*: each output block draws from a
per-block RNG seed. There's no cross-block graph structure to expand in Rust —
the per-block seed (a compact ~97-byte integer for the RandomState path, or a
``SeedSequence`` for the Generator path; see ``Random._info``) is the work — so
this is a plain Python layer.

``to_dask_graph`` reuses the expr's own ``_task`` (which handles every distribution
subclass and the ``extra_chunks`` book-keeping). ``to_task_records`` is the hot
path for big random graphs, so it builds the flat record directly — mirroring the
no-dependency branch of ``_task`` — instead of constructing a throwaway dask
``Task`` per block and reading it back. ``Random._frisky_layer`` only routes here
when no distribution parameter is an array (no dependencies), so every per-block
task is the same positional call with block-specific bit-generator state and size.
"""

from __future__ import annotations

from itertools import product


class RandomLayer:
    def __init__(self, expr):
        self.expr = expr

    def _block_ids(self):
        return product(*(range(len(bd)) for bd in self.expr.chunks))

    def to_dask_graph(self):
        e = self.expr
        return {(e._name, *bid): e._task((e._name, *bid), bid) for bid in self._block_ids()}

    def to_task_records(self):
        e = self.expr
        # Hoist the per-layer state out of the per-block loop (these are otherwise
        # re-fetched through dask's _expr __getattr__ on every block).
        bitgens, name, sizes, gen, func_applier = e._info
        distribution = e.distribution
        args = tuple(e.args)  # no array deps (guaranteed by _frisky_layer)
        kwargs = e.kwargs or {}
        records = []
        if not e.extra_chunks:
            # chunks == base_chunks, so block ids enumerate in flat C order — the
            # same index into bitgens/sizes — and the stride math is unnecessary.
            for flat_idx, bid in enumerate(self._block_ids()):
                rec_args = (gen, distribution, bitgens[flat_idx], sizes[flat_idx], args, kwargs)
                records.append((str((name, *bid)), func_applier, rec_args, {}, []))
        else:
            for bid in self._block_ids():
                flat_idx = e._block_id_to_flat_index(bid)
                rec_args = (gen, distribution, bitgens[flat_idx], sizes[flat_idx], args, kwargs)
                records.append((str((name, *bid)), func_applier, rec_args, {}, []))
        return records
