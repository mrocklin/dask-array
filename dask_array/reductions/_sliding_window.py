from __future__ import annotations

import math
from bisect import bisect_right
from functools import partial
from itertools import product

import numpy as np

from dask.utils import cached_property

from dask_array._expr import ArrayExpr

# Per-reducer plumbing for the native-chunk sliding-window path: the
# accumulating ufunc, the value substituted for NaNs (NaN-skipping reducers),
# and whether a per-window valid count is carried (nanmean).
NATIVE_SLIDING_REDUCERS = {
    "sum": (np.add, None, False),
    "prod": (np.multiply, None, False),
    "min": (np.minimum, None, False),
    "max": (np.maximum, None, False),
    "any": (np.logical_or, None, False),
    "all": (np.logical_and, None, False),
    "mean": (np.add, None, False),
    "nansum": (np.add, 0, False),
    "nanprod": (np.multiply, 1, False),
    "nanmin": (np.fmin, None, False),
    "nanmax": (np.fmax, None, False),
    "nanmean": (np.add, 0, True),
}

# bottleneck moving-window functions (xarray's dask rolling path) and the
# NaN-skipping reducer with the same per-window semantics; windows with fewer
# than ``min_count`` valid values become NaN on top.
MOVING_WINDOW_REDUCERS = {
    "move_sum": "nansum",
    "move_mean": "nanmean",
    "move_min": "nanmin",
    "move_max": "nanmax",
}


def supports_native_sliding_window(chunks, window):
    """Whether ``SlidingWindowReduction`` can replace the overlap-based plan.

    True when the overlap path would coarsen chunks (some chunk smaller than
    the window's depth, or the trailing chunk no larger than it) and the
    banded decomposition is valid: chunk sizes are known and positive, the
    array is at least one window long, and every output-emitting block is no
    larger than ``window - 1`` so each window's right edge lands past the
    block's own end.
    """
    depth = window - 1
    if depth <= 0:
        return False
    if any(math.isnan(c) for c in chunks) or min(chunks) <= 0:
        return False
    if sum(chunks) < window:
        return False
    if min(chunks) >= depth and chunks[-1] > depth:
        return False  # the overlap path already keeps these chunks
    out_len = sum(chunks) - depth
    start = 0
    for c in chunks:
        if start >= out_len:
            break
        if c > depth:
            return False
        start += c
    return True


def _prepared_values(part, nan_identity, with_count, out_dtype):
    values = np.asarray(part)
    count = None
    if nan_identity is not None or with_count:
        valid = ~np.isnan(values)
        if with_count:
            count = valid.astype(np.int64)
        if nan_identity is not None:
            values = np.where(valid, values, nan_identity)
    return values.astype(out_dtype, copy=False), count


def _sliding_window_block_total(block, *, reducer, sliding_axis, out_dtype, with_count=None):
    ufunc, nan_identity, table_count = NATIVE_SLIDING_REDUCERS[reducer]
    if with_count is None:
        with_count = table_count
    values, count = _prepared_values(block, nan_identity, with_count, out_dtype)
    total = ufunc.reduce(values, axis=sliding_axis, keepdims=True)
    if with_count:
        return {"total": total, "count": np.add.reduce(count, axis=sliding_axis, keepdims=True)}
    return total


def _sliding_window_banded_reduce(
    block,
    totals,
    right_parts,
    out_len,
    band_offset,
    *,
    reducer,
    window,
    sliding_axis,
    out_dtype,
    keepdims,
    window_axis,
):
    """Combine one block's suffix scan, covered-block totals, and the prefix
    scan of the right-edge band into that block's windowed reductions.

    Output position ``t`` reduces the window ``[t, t + window)`` counted from
    this block's start: a suffix of this block, all of each middle block
    (``totals``), and a prefix of the band that starts ``band_offset`` into
    the concatenated ``right_parts``.  The per-block values (``out_len``,
    ``band_offset``) are positional so the Rust records layer can pass them as
    plain scalar slots.
    """
    ufunc, nan_identity, with_count = NATIVE_SLIDING_REDUCERS[reducer]

    def axis_slice(a, start, stop):
        index = [slice(None)] * a.ndim
        index[sliding_axis] = slice(start, stop)
        return a[tuple(index)]

    def suffix_scan(a, scan_ufunc):
        return np.flip(scan_ufunc.accumulate(np.flip(a, sliding_axis), axis=sliding_axis), sliding_axis)

    values, count = _prepared_values(block, nan_identity, with_count, out_dtype)
    out = axis_slice(suffix_scan(values, ufunc), 0, out_len)
    if with_count:
        out_count = axis_slice(suffix_scan(count, np.add), 0, out_len)

    for total in totals:
        if with_count:
            out_count = out_count + total["count"]
            total = total["total"]
        out = ufunc(out, total)

    prepared = [_prepared_values(part, nan_identity, with_count, out_dtype) for part in right_parts]
    right_values = [v for v, _ in prepared]
    band = right_values[0] if len(right_values) == 1 else np.concatenate(right_values, axis=sliding_axis)
    prefix = ufunc.accumulate(axis_slice(band, 0, band_offset + out_len), axis=sliding_axis)
    out = ufunc(out, axis_slice(prefix, band_offset, band_offset + out_len))
    if with_count:
        right_counts = [c for _, c in prepared]
        band_count = right_counts[0] if len(right_counts) == 1 else np.concatenate(right_counts, axis=sliding_axis)
        prefix_count = np.add.accumulate(axis_slice(band_count, 0, band_offset + out_len), axis=sliding_axis)
        out_count = out_count + axis_slice(prefix_count, band_offset, band_offset + out_len)

    out = np.asarray(out, dtype=out_dtype)
    if reducer == "mean":
        np.divide(out, window, out=out, casting="unsafe")
    elif reducer == "nanmean":
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(out, out_count, out=out, casting="unsafe")
    if keepdims:
        out = np.expand_dims(out, axis=window_axis)
    return out


def supports_native_moving_window(chunks, window):
    """Whether ``MovingWindowReduction`` can replace the overlap-based plan
    for a trailing-window (bottleneck ``move_*``) reduction.

    True when chunk sizes are known and positive, there is more than one
    chunk (with one the overlap path is already native), the array holds at
    least one full window, and every chunk is smaller than the window so each
    window reaches past its block's own start.  Under these conditions the
    overlap path would always coarsen: its boundary-"none" edge rule merges a
    first chunk no larger than the depth into its neighbor.
    """
    if window <= 1:
        return False
    if any(math.isnan(c) for c in chunks) or min(chunks) <= 0:
        return False
    if len(chunks) < 2 or sum(chunks) < window:
        return False
    return max(chunks) <= window - 1


def _moving_window_banded_reduce(
    block,
    totals,
    left_parts,
    n_trunc,
    band_offset,
    *,
    reducer,
    window,
    min_count,
    sliding_axis,
    out_dtype,
):
    """Trailing-window (bottleneck ``move_*``) reduction for one block on
    native chunks.

    Output position ``t`` reduces the window ``[t - window + 1, t]`` counted
    from this block's start, clipped at the array's start: a suffix of the
    concatenated ``left_parts`` (the blocks the window's left edge sweeps,
    starting ``band_offset`` in; the first ``n_trunc`` positions clip to the
    band start), all of each middle block (``totals``), and a prefix of this
    block.  Windows with fewer than ``min_count`` valid values yield NaN
    (``None`` means ``window``, like bottleneck).  The per-block values
    (``n_trunc``, ``band_offset``) are positional so the Rust records layer
    can pass them as plain scalar slots.
    """
    ufunc, nan_identity, _ = NATIVE_SLIDING_REDUCERS[reducer]

    def axis_slice(a, start, stop):
        index = [slice(None)] * a.ndim
        index[sliding_axis] = slice(start, stop)
        return a[tuple(index)]

    values, count = _prepared_values(block, nan_identity, True, out_dtype)
    out = ufunc.accumulate(values, axis=sliding_axis)
    out_count = np.add.accumulate(count, axis=sliding_axis)
    out_len = block.shape[sliding_axis]

    for total in totals:
        out_count = out_count + total["count"]
        out = ufunc(out, total["total"])

    if left_parts:
        prepared = [_prepared_values(part, nan_identity, True, out_dtype) for part in left_parts]

        def band_suffixes(parts, scan_ufunc):
            band = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=sliding_axis)
            scan = np.flip(scan_ufunc.accumulate(np.flip(band, sliding_axis), axis=sliding_axis), sliding_axis)
            seg = axis_slice(scan, band_offset, band_offset + out_len - n_trunc)
            if n_trunc:
                first = axis_slice(scan, band_offset, band_offset + 1)
                seg = np.concatenate([np.repeat(first, n_trunc, axis=sliding_axis), seg], axis=sliding_axis)
            return seg

        out = ufunc(out, band_suffixes([v for v, _ in prepared], ufunc))
        out_count = out_count + band_suffixes([c for _, c in prepared], np.add)

    out = np.asarray(out, dtype=out_dtype)
    if reducer == "nanmean":
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(out, out_count, out=out, casting="unsafe")
    limit = window if min_count is None else min_count
    out[out_count < limit] = np.nan
    return out


class MovingWindowReduction(ArrayExpr):
    """A trailing-window reduction (bottleneck ``move_*`` semantics) computed
    on the input's native chunks.

    Same shape and chunks as the input: output position ``j`` reduces the
    valid values in ``[j - window + 1, j]`` (clipped at the array start) and
    is NaN when fewer than ``min_count`` are valid.  Each output block
    combines a prefix scan of its own input block, the keepdims totals of the
    blocks its windows cover whole, and a suffix scan of the block(s) its
    windows' left edges sweep.  Only valid under
    ``supports_native_moving_window``.
    """

    _parameters = ["array", "window", "min_count", "sliding_axis", "reducer", "dtype"]

    @cached_property
    def _name(self):
        return f"moving-window-{self.reducer}-{self.deterministic_token}"

    @cached_property
    def dtype(self):
        return np.dtype(self.operand("dtype"))

    @cached_property
    def _meta(self):
        return np.empty((0,) * self.array.ndim, dtype=self.dtype)

    @cached_property
    def chunks(self):
        return self.array.chunks

    @cached_property
    def transfer_bytes(self):
        # See ArrayExpr.transfer_bytes.  Each output block reads its own input
        # block (co-located under min), one hyperplane per fully covered
        # middle block, and the left-edge band; under max the own block and
        # the whole band blocks are fetched.
        from dask_array._expr import TransferBytes

        x = self.array
        axis = self.sliding_axis
        itemsize = x.dtype.itemsize
        cross = itemsize * math.prod(s for i, s in enumerate(x.shape) if i != axis)
        lo = 0.0
        hi = 0.0
        for _, c, band_start, g, h, middle in self._block_plan:
            lo += (len(middle) + (sum(x.chunks[axis][g : h + 1]) - band_start if g is not None else 0)) * cross
            hi += (len(middle) + c + (sum(x.chunks[axis][g : h + 1]) if g is not None else 0)) * cross
        return TransferBytes(lo, hi)

    @cached_property
    def _block_plan(self):
        """Per block along the sliding axis: (start, size, band offset
        within block ``g``, band block ``g``, band block ``h``, middle-block
        range).  ``g``/``h`` are None when the block starts the array (no
        band); the middle blocks ``h+1..i`` are covered whole."""
        chunks = self.array.chunks[self.sliding_axis]
        window = self.window
        starts = [0]
        for c in chunks:
            starts.append(starts[-1] + c)
        plan = []
        for i, c in enumerate(chunks):
            start = starts[i]
            if start == 0:
                plan.append((start, c, 0, None, None, range(0)))
                continue
            band_first = max(0, start - window + 1)  # left edge of this block's first window
            band_last = max(band_first, start + c - window)  # left edge of its last window
            g = bisect_right(starts, band_first) - 1
            h = bisect_right(starts, band_last) - 1
            plan.append((start, c, band_first - starts[g], g, h, range(h + 1, i)))
        return plan

    @cached_property
    def _reduce_func(self):
        return partial(
            _moving_window_banded_reduce,
            reducer=self.reducer,
            window=self.window,
            min_count=self.min_count,
            sliding_axis=self.sliding_axis,
            out_dtype=self.operand("dtype"),
        )

    @cached_property
    def _total_func(self):
        return partial(
            _sliding_window_block_total,
            reducer=self.reducer,
            sliding_axis=self.sliding_axis,
            out_dtype=self.operand("dtype"),
            with_count=True,
        )

    def _layer(self):
        x = self.array
        axis = self.sliding_axis
        window = self.window

        total_name = f"{self._name}-total"
        other_ranges = [range(nb) for j, nb in enumerate(x.numblocks) if j != axis]

        def block_key(q, other):
            index = list(other)
            index.insert(axis, q)
            return tuple(index)

        dsk = {}
        needed_totals = set()
        for i, (start, c, band_offset, g, h, middle) in enumerate(self._block_plan):
            needed_totals.update(middle)
            n_trunc = max(0, min(c, window - 1 - start))
            for other in product(*other_ranges):
                index = block_key(i, other)
                dsk[(self._name,) + index] = (
                    self._reduce_func,
                    (x._name,) + index,
                    [(total_name,) + block_key(q, other) for q in middle],
                    [(x._name,) + block_key(q, other) for q in range(g, h + 1)] if g is not None else [],
                    n_trunc,
                    band_offset,
                )

        for q in sorted(needed_totals):
            for other in product(*other_ranges):
                index = block_key(q, other)
                dsk[(total_name,) + index] = (self._total_func, (x._name,) + index)

        return dsk

    def _frisky_layer(self):
        from dask_array._frisky.sliding_window import MovingWindowReductionLayer

        window = self.window
        plan = []
        for start, c, band_offset, g, h, _middle in self._block_plan:
            n_trunc = max(0, min(c, window - 1 - start))
            band_lo = -1 if g is None else g
            band_hi = -1 if h is None else h
            plan.append([n_trunc, band_offset, band_lo, band_hi])
        # A `-total` task is one keepdims hyperplane of running totals plus
        # (always, for the moving variant) a same-shaped count plane.
        return MovingWindowReductionLayer(
            self._name,
            self.array._name,
            self._reduce_func,
            self._total_func,
            self.sliding_axis,
            self.array.numblocks,
            plan,
            self.array.chunks,
            np.dtype(self.dtype).itemsize + np.dtype(np.int64).itemsize,
        )


class SlidingWindowReduction(ArrayExpr):
    """``reduction(sliding_window_view(x, window), axis=window_axis)`` computed
    on the input's native chunks.

    Output chunks equal the input's, trimmed by ``window - 1`` at the end of
    the sliding axis.  Each output block combines a suffix scan of its own
    input block, the keepdims totals of the blocks its windows cover whole,
    and a prefix scan of the block(s) its windows' right edges sweep — so no
    chunk ever needs to hold a full window.  Only valid under
    ``supports_native_sliding_window``.
    """

    _parameters = ["array", "window", "sliding_axis", "window_axis", "keepdims", "reducer", "dtype"]

    @cached_property
    def _name(self):
        return f"sliding-window-{self.reducer}-{self.deterministic_token}"

    @cached_property
    def dtype(self):
        return np.dtype(self.operand("dtype"))

    @cached_property
    def _meta(self):
        return np.empty((0,) * len(self.chunks), dtype=self.dtype)

    @cached_property
    def chunks(self):
        chunks = list(self.array.chunks)
        remaining = sum(chunks[self.sliding_axis]) - self.window + 1
        trimmed = []
        for c in chunks[self.sliding_axis]:
            if remaining <= 0:
                break
            take = min(c, remaining)
            trimmed.append(take)
            remaining -= take
        chunks[self.sliding_axis] = tuple(trimmed)
        if self.keepdims:
            chunks.insert(self.window_axis, (1,))
        return tuple(chunks)

    @cached_property
    def transfer_bytes(self):
        # See ArrayExpr.transfer_bytes.  Each output block reads its own input
        # block (co-located under min), one hyperplane per fully covered
        # middle block, and the right-edge band; under max the own block and
        # the whole right-edge blocks are fetched.
        from dask_array._expr import TransferBytes

        x = self.array
        axis = self.sliding_axis
        itemsize = x.dtype.itemsize
        cross = itemsize * math.prod(s for i, s in enumerate(x.shape) if i != axis)
        chunks = x.chunks[axis]
        lo = 0.0
        hi = 0.0
        for i, (out_len, band_offset, b, e) in enumerate(self._block_plan):
            if out_len <= 0:
                break
            middles = (b - i - 1) * cross
            lo += middles + (band_offset + out_len) * cross
            hi += middles + (chunks[i] + sum(chunks[b : e + 1])) * cross
        return TransferBytes(lo, hi)

    @cached_property
    def _block_plan(self):
        """Per block along the sliding axis: (output length, band offset
        within block ``b``, and the band blocks ``b..e`` its windows' right
        edges sweep).  Blocks the ``window - 1`` trim consumes entirely have
        output length 0; middle blocks (covered whole) are ``i+1..b``."""
        chunks = self.array.chunks[self.sliding_axis]
        window = self.window
        starts = [0]
        for c in chunks:
            starts.append(starts[-1] + c)
        remaining = sum(chunks) - window + 1
        plan = []
        for i, c in enumerate(chunks):
            out_len = max(0, min(c, remaining))
            remaining -= out_len
            if out_len <= 0:
                plan.append((0, 0, i, i))
                continue
            edge = starts[i] + window - 1  # right edge of this block's first window
            b = bisect_right(starts, edge) - 1
            e = bisect_right(starts, edge + out_len - 1) - 1
            plan.append((out_len, edge - starts[b], b, e))
        return plan

    @cached_property
    def _reduce_func(self):
        return partial(
            _sliding_window_banded_reduce,
            reducer=self.reducer,
            window=self.window,
            sliding_axis=self.sliding_axis,
            out_dtype=self.operand("dtype"),
            keepdims=self.keepdims,
            window_axis=self.window_axis,
        )

    @cached_property
    def _total_func(self):
        return partial(
            _sliding_window_block_total,
            reducer=self.reducer,
            sliding_axis=self.sliding_axis,
            out_dtype=self.operand("dtype"),
        )

    def _layer(self):
        x = self.array
        axis = self.sliding_axis

        total_name = f"{self._name}-total"
        other_ranges = [range(nb) for j, nb in enumerate(x.numblocks) if j != axis]

        def block_key(q, other):
            index = list(other)
            index.insert(axis, q)
            return tuple(index)

        dsk = {}
        needed_totals = set()
        for i, (out_len, band_offset, b, e) in enumerate(self._block_plan):
            if out_len <= 0:
                break
            needed_totals.update(range(i + 1, b))
            for other in product(*other_ranges):
                index = block_key(i, other)
                out_index = list(index)
                if self.keepdims:
                    out_index.insert(self.window_axis, 0)
                dsk[(self._name,) + tuple(out_index)] = (
                    self._reduce_func,
                    (x._name,) + index,
                    [(total_name,) + block_key(q, other) for q in range(i + 1, b)],
                    [(x._name,) + block_key(q, other) for q in range(b, e + 1)],
                    out_len,
                    band_offset,
                )

        for q in sorted(needed_totals):
            for other in product(*other_ranges):
                index = block_key(q, other)
                dsk[(total_name,) + index] = (self._total_func, (x._name,) + index)

        return dsk

    def _frisky_layer(self):
        from dask_array._frisky.sliding_window import SlidingWindowReductionLayer

        # A `-total` task is one keepdims hyperplane of running totals, plus a
        # same-shaped count plane for reducers that track counts.
        with_count = NATIVE_SLIDING_REDUCERS[self.reducer][2]
        total_itemsize = np.dtype(self.dtype).itemsize + (np.dtype(np.int64).itemsize if with_count else 0)
        return SlidingWindowReductionLayer(
            self._name,
            self.array._name,
            self._reduce_func,
            self._total_func,
            self.sliding_axis,
            self.array.numblocks,
            self.keepdims,
            self.window_axis,
            [list(row) for row in self._block_plan],
            self.array.chunks,
            total_itemsize,
        )
