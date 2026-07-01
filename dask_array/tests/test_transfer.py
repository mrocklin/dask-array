import math
import operator

import numpy as np

import dask_array as da
from dask_array._expr import ArrayExpr, TransferBytes
from dask_array._overlap import OverlapInternal
from dask_array._rechunk import Rechunk, _rechunk_stage_transfer
from dask_array.reductions._cumulative import CumReduction


def _walk(expr):
    seen = set()
    stack = [expr]
    while stack:
        e = stack.pop()
        if e._name in seen:
            continue
        seen.add(e._name)
        yield e
        stack.extend(d for d in e.dependencies() if hasattr(d, "chunks"))


def _find(expr, cls_name):
    return [e for e in _walk(expr) if type(e).__name__ == cls_name]


def test_elemwise_aligned_is_free():
    x = da.ones((10, 10), chunks=(5, 5))
    y = da.zeros((10, 10), chunks=(5, 5))
    lo, hi = (x + y).transfer_bytes
    assert lo == 0
    assert hi == x.nbytes + y.nbytes


def test_elemwise_scalar_operand():
    x = da.ones((10, 10), chunks=(5, 5))
    lo, hi = (x + 1).transfer_bytes
    assert lo == 0
    assert hi == x.nbytes


def test_elemwise_broadcast_fanout():
    x = da.ones((10, 10), chunks=(5, 5))
    y = da.ones((10, 1), chunks=(5, 1))
    lo, hi = (x + y).transfer_bytes
    # each y block feeds 2 output blocks along the broadcast axis; one is
    # co-located under min, both fetch under max
    assert lo == y.nbytes
    assert hi == x.nbytes + 2 * y.nbytes


def test_rechunk_aligned_merge():
    x = da.ones((10,), chunks=(5,))
    r = x.rechunk((10,)).expr
    assert isinstance(r, Rechunk)
    lo, hi = r.transfer_bytes
    # merging two blocks: half the data joins the other half
    assert lo == x.nbytes / 2
    assert hi == x.nbytes


def test_rechunk_jittered():
    lo, hi = _rechunk_stage_transfer(((4, 6),), ((5, 5),), 8)
    # new[0:5) takes 4 from old0 + 1 from old1 -> 1 element moves;
    # new[5:10) sits inside old1 -> free under min
    assert lo == 8
    # old1 is cut and fetched whole by both split tasks (2*6*8); the
    # two-source merge new[0:5) fetches its 5 elements; new[5:10) is a
    # single-source alias
    assert hi == 2 * 6 * 8 + 5 * 8


def test_rechunk_identity_stage():
    # every new block aliases an uncut old block: nothing moves
    assert _rechunk_stage_transfer(((5, 5),), ((5, 5),), 8) == (0, 0)


def test_rechunk_pure_split():
    lo, hi = _rechunk_stage_transfer(((10,),), ((5, 5),), 8)
    # splitting is free under min (both halves cut at the source)
    assert lo == 0
    # the old block is fetched whole by each of the two split tasks; the
    # single-source outputs alias the splits
    assert hi == 2 * 10 * 8


def test_slice_within_blocks():
    x = da.ones((10,), chunks=(5,))
    s = x[2:].expr
    lo, hi = s.transfer_bytes
    # each output block is cut from one input block: nothing must move
    assert lo == 0
    # both input blocks fetched whole, the second is a full-block alias
    assert hi == 5 * 8


def test_slice_full_is_free_via_alias():
    x = da.ones((10,), chunks=(5,))
    from dask_array.slicing._basic import SliceSlicesIntegers

    s = SliceSlicesIntegers(x.expr, (slice(None),), True)
    assert s.transfer_bytes == TransferBytes(0.0, 0.0)


def test_overlap_ghost_cells():
    x = da.ones((10, 4), chunks=(5, 4))
    o = OverlapInternal(x.expr, {0: (1, 1)})
    lo, hi = o.transfer_bytes
    # one internal boundary, (1+1) hyperplanes of 4 float64s each
    assert lo == 2 * 4 * 8
    # max: both center blocks fetched whole (320), each of the two ghost
    # fragment tasks fetches its whole neighbor block (320), plus the
    # fragments themselves (64)
    assert hi == x.nbytes + 2 * (5 * 4 * 8) + lo


def test_partial_reduce():
    x = da.ones((10,), chunks=(5,))
    opt = x.sum().expr.optimize(fuse=False)
    (pr,) = _find(opt, "PartialReduce")
    lo, hi = pr.transfer_bytes
    # two (1,)-sized partials combine; one hosts the combine under min
    assert lo == 8
    assert hi == 16


def test_cumsum_carries():
    x = da.ones((10,), chunks=(5,))
    c = CumReduction(x.expr, np.cumsum, operator.add, 0, 0, None)
    lo, hi = c.transfer_bytes
    # one boundary carry of one float64 hyperplane
    assert lo == 8
    # per-block scans fetch input blocks, the extra/result tasks each
    # re-fetch a per-block block whole (k=2), and the carry is fetched
    # twice (by the next extra and by its result)
    assert hi == x.nbytes * 2 + 2 * 8


def test_concatenate_is_alias_routing():
    x = da.ones((10,), chunks=(5,))
    y = da.ones((10,), chunks=(5,))
    c = da.concatenate([x, y]).expr
    assert type(c).__name__ == "Concatenate"
    assert c.transfer_bytes == TransferBytes(0.0, 0.0)


def test_leaves_are_free():
    x = da.ones((10,), chunks=(5,))
    assert x.transfer_bytes == TransferBytes(0.0, 0.0)


def test_graph_wide_sanity():
    # a small workload with the shapes we care about: overlap, rechunk,
    # slicing, reductions, broadcasting
    x = da.ones((100, 8), chunks=(10, 4))
    y = da.map_overlap(lambda b: b, x, depth=(2, 0), boundary=np.nan)
    z = (x * y) - y.mean(axis=1, keepdims=True)
    r = z.rechunk((50, 4))[3:].sum()
    for expr in _walk(r.expr.optimize(fuse=False)):
        lo, hi = expr.transfer_bytes
        assert not math.isnan(lo) and not math.isnan(hi), type(expr).__name__
        assert 0 <= lo <= hi, (type(expr).__name__, lo, hi)


def test_graph_wide_sanity_fused():
    x = da.ones((100, 8), chunks=(10, 4))
    z = da.log(x + 1) * x
    for expr in _walk(z.expr.optimize()):
        lo, hi = expr.transfer_bytes
        assert 0 <= lo <= hi, (type(expr).__name__, lo, hi)


def test_stack_partitions_output():
    xs = [da.ones((10,), chunks=(5,)) for _ in range(4)]
    s = da.stack(xs).expr
    assert type(s).__name__ == "Stack"
    lo, hi = s.transfer_bytes
    # inputs partition the output along the new axis: no broadcast fanout
    assert lo == 0
    assert hi == sum(x.nbytes for x in xs)


def test_shuffle_block_reversal_is_free():
    from dask_array._shuffle import Shuffle

    x = da.ones((10,), chunks=(5,))
    s = Shuffle(x.expr, [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]], 0, "shuffle")
    lo, hi = s.transfer_bytes
    # each output chunk is a whole source block: free under min
    assert lo == 0
    # each split fetches its source block whole; single-source merges alias
    assert hi == x.nbytes


def test_shuffle_mixing_charges_min():
    from dask_array._shuffle import Shuffle

    x = da.ones((10,), chunks=(5,))
    s = Shuffle(x.expr, [[0, 1, 2, 3, 9], [4, 5, 6, 7, 8]], 0, "shuffle")
    lo, hi = s.transfer_bytes
    # each output chunk takes 4 from one block + 1 from the other
    assert lo == 2 * 8
    # both source blocks fetched whole per output chunk (4 splits of 5) plus
    # two multi-source merges of 5 elements each
    assert hi == (4 * 5 + 2 * 5) * 8


class _Gather(ArrayExpr):
    """Minimal fan-in node exercising the ArrayExpr.transfer_bytes default."""

    _parameters = ["array"]

    @property
    def _meta(self):
        return self.array._meta

    @property
    def chunks(self):
        return ((10,),)


def test_default_charges_fan_in():
    x = da.ones((10,), chunks=(5,))
    lo, hi = _Gather(x.expr).transfer_bytes
    # a 2-into-1 gather must move the half that joins the co-located block
    assert lo == x.nbytes / 2
    assert hi == x.nbytes


def test_blockwise_contraction_gather():
    x = da.ones((4, 6), chunks=(2, 2))
    z = da.blockwise(lambda a: a.sum(axis=1), "i", x, "ij", dtype="f8", concatenate=True)
    lo, hi = z.expr.transfer_bytes
    # each output block gathers 3 x-blocks along the contracted axis, one of
    # which hosts the combine
    assert lo == x.nbytes * (1 - 1 / 3)
    assert hi == x.nbytes


def test_duplicate_operand_counts_once():
    x = da.ones((10,), chunks=(5,))
    assert (x + x).transfer_bytes == (x + 1).transfer_bytes
