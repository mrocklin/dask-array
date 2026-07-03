from __future__ import annotations

import builtins
import math
import warnings
from functools import partial
from itertools import product
from numbers import Integral

import numpy as np
from tlz import compose, get, partition_all

from dask import config
from dask_array._new_collection import new_collection
from dask_array._expr import ArrayExpr
from dask_array._utils import compute_meta
from dask_array._core_utils import _concatenate2
from dask_array._numpy_compat import ComplexWarning
from dask_array._utils import is_arraylike, validate_axis
from dask.blockwise import lol_tuples
from dask.tokenize import _tokenize_deterministic
from dask.utils import cached_property, funcname, getargspec, is_series_like


class Reduction(ArrayExpr):
    """Logical reduction expression that captures reduction intent.

    This expression represents a reduction operation conceptually,
    without immediately materializing the physical Blockwise + PartialReduce
    cascade. The physical implementation is deferred to _lower().
    """

    sliding_window_reducer = None
    sliding_window_binop = None
    sliding_window_binop_kwargs = {}

    _parameters = [
        "array",
        "chunk",
        "aggregate",
        "axis",
        "keepdims",
        "dtype",
        "split_every",
        "combine",
        "name",
        "concatenate",
        "output_size",
        "meta",
        "weights",
    ]
    _defaults = {
        "axis": None,
        "keepdims": False,
        "dtype": None,
        "split_every": None,
        "combine": None,
        "name": None,
        "concatenate": True,
        "output_size": 1,
        "meta": None,
        "weights": None,
    }

    def __dask_tokenize__(self):
        if not self._determ_token:
            self._determ_token = _tokenize_deterministic(
                type(self),
                self.chunk,
                self.aggregate,
                self.array,
                self.axis,
                self.keepdims,
                self.operand("dtype"),
                self.split_every,
                self.combine,
                self.concatenate,
                self.output_size,
                self.weights,
            )
        return self._determ_token

    @cached_property
    def _name(self):
        prefix = self.operand("name") or funcname(self.chunk)
        return f"{prefix}-{self.deterministic_token}"

    @cached_property
    def chunks(self):
        """Output chunks after reduction."""
        axis = self.axis
        if self.keepdims:
            return tuple((self.output_size,) if i in axis else c for i, c in enumerate(self.array.chunks))
        else:
            return tuple(c for i, c in enumerate(self.array.chunks) if i not in axis)

    @cached_property
    def dtype(self):
        if self.operand("dtype") is not None:
            return np.dtype(self.operand("dtype"))
        return self.array.dtype

    @property
    def _meta(self):
        # Compute a minimal metadata array with correct dtype and ndim
        dtype = self.dtype
        ndim = len(self.chunks)
        return np.empty((0,) * ndim, dtype=dtype)

    def _layer(self):
        """Generate the task layer by lowering first.

        Reduction should always be lowered before graph generation,
        but we need to support direct _layer() calls for is_dask_collection().
        """
        return self.lower_completely()._layer()

    def _simplify_up(self, parent, dependents):
        """Allow slice operations to push through Reduction."""
        from dask_array.slicing import SliceSlicesIntegers

        if isinstance(parent, SliceSlicesIntegers):
            return self._accept_slice(parent)
        return None

    def _accept_slice(self, slice_expr):
        """Accept a slice being pushed through this Reduction."""
        reduced_axes = set(self.axis)

        def make_result(sliced_input, input_index):
            # Handle sliced weights if present
            sliced_weights = None
            if self.weights is not None:
                sliced_weights = new_collection(self.weights)[input_index].expr

            return type(self)(
                sliced_input.expr,
                self.chunk,
                self.aggregate,
                self.axis,
                self.keepdims,
                self.operand("dtype"),
                self.split_every,
                self.combine,
                self.operand("name"),
                self.concatenate,
                self.output_size,
                self.meta,
                sliced_weights,
            )

        return _accept_slice_impl(slice_expr, self.array, reduced_axes, self.keepdims, make_result)

    def _lower(self):
        """Lower to Blockwise + PartialReduce cascade."""
        from dask_array._collection import blockwise

        axis = self.axis
        dtype = self.operand("dtype") or float
        name = self.operand("name")
        output_size = self.output_size

        # Prepare chunk function with dtype if needed
        chunk_func = self.chunk
        if "dtype" in getargspec(chunk_func).args:
            chunk_func = partial(chunk_func, dtype=dtype)

        aggregate_func = self.aggregate
        if "dtype" in getargspec(aggregate_func).args:
            aggregate_func = partial(aggregate_func, dtype=dtype)

        # Build args for blockwise
        inds = tuple(range(self.array.ndim))
        args = (self.array, inds)

        if self.weights is not None:
            args += (self.weights, inds)

        # Create Blockwise for per-chunk reduction
        adjust_chunks = {i: output_size for i in axis}
        tmp = blockwise(
            chunk_func,
            inds,
            *args,
            axis=axis,
            keepdims=True,
            token=name,
            dtype=dtype,
            adjust_chunks=adjust_chunks,
        )

        # Compute reduced_meta for PartialReduce
        if self.meta is None and hasattr(self.array, "_meta"):
            try:
                reduced_meta = compute_meta(
                    chunk_func, self.array.dtype, self.array._meta, axis=axis, keepdims=True, computing_meta=True
                )
            except TypeError:
                reduced_meta = compute_meta(chunk_func, self.array.dtype, self.array._meta, axis=axis, keepdims=True)
            except ValueError:
                reduced_meta = None
        else:
            reduced_meta = self.meta

        # Build tree reduction with PartialReduce
        result = _build_tree_reduce_expr(
            tmp.expr,
            aggregate_func,
            axis,
            self.keepdims,
            dtype,
            self.split_every,
            self.combine,
            name,
            self.concatenate,
            reduced_meta,
        )

        # Override final chunks for output_size != 1
        if self.keepdims and output_size != 1:
            from dask_array._expr import ChunksOverride

            final_chunks = tuple((output_size,) if i in axis else c for i, c in enumerate(result.chunks))
            result = ChunksOverride(result, final_chunks)

        return result

    @classmethod
    def sliding_window_reduce_block(
        cls,
        block,
        *,
        window,
        sliding_axis,
        out_dtype,
        ddof=0,
        implicit_complex_dtype=False,
    ):
        if cls.sliding_window_binop is None:
            raise NotImplementedError(cls.__name__)

        out_len = block.shape[sliding_axis] - window + 1
        index = [slice(None)] * block.ndim
        index[sliding_axis] = slice(0, out_len)
        out = np.array(block[tuple(index)], dtype=out_dtype, copy=True)

        for offset in range(1, window):
            index[sliding_axis] = slice(offset, offset + out_len)
            other = block[tuple(index)]
            cls.sliding_window_binop(out, other, out=out, **cls.sliding_window_binop_kwargs)

        return cls.sliding_window_finalize(out, window=window)

    @classmethod
    def sliding_window_finalize(cls, out, *, window):
        return out


def reduction(
    x,
    chunk,
    aggregate,
    axis=None,
    keepdims=False,
    dtype=None,
    split_every=None,
    combine=None,
    name=None,
    out=None,
    concatenate=True,
    output_size=1,
    meta=None,
    weights=None,
    reduction_cls=Reduction,
):
    """General version of reductions

    Parameters
    ----------
    x: Array
        Data being reduced along one or more axes
    chunk: callable(x_chunk, [weights_chunk=None], axis, keepdims)
        First function to be executed when resolving the dask graph.
        This function is applied in parallel to all original chunks of x.
        See below for function parameters.
    combine: callable(x_chunk, axis, keepdims), optional
        Function used for intermediate recursive aggregation (see
        split_every below). If omitted, it defaults to aggregate.
        If the reduction can be performed in less than 3 steps, it will not
        be invoked at all.
    aggregate: callable(x_chunk, axis, keepdims)
        Last function to be executed when resolving the dask graph,
        producing the final output. It is always invoked, even when the reduced
        Array counts a single chunk along the reduced axes.
    axis: int or sequence of ints, optional
        Axis or axes to aggregate upon. If omitted, aggregate along all axes.
    keepdims: boolean, optional
        Whether the reduction function should preserve the reduced axes,
        leaving them at size ``output_size``, or remove them.
    dtype: np.dtype
        data type of output. This argument was previously optional, but
        leaving as ``None`` will now raise an exception.
    split_every: int >= 2 or dict(axis: int), optional
        Determines the depth of the recursive aggregation. If set to or more
        than the number of input chunks, the aggregation will be performed in
        two steps, one ``chunk`` function per input chunk and a single
        ``aggregate`` function at the end. If set to less than that, an
        intermediate ``combine`` function will be used, so that any one
        ``combine`` or ``aggregate`` function has no more than ``split_every``
        inputs. The depth of the aggregation graph will be
        :math:`log_{split_every}(input chunks along reduced axes)`. Setting to
        a low value can reduce cache size and network transfers, at the cost of
        more CPU and a larger dask graph.

        Omit to let dask heuristically decide a good default. A default can
        also be set globally with the ``split_every`` key in
        :mod:`dask.config`.
    name: str, optional
        Prefix of the keys of the intermediate and output nodes. If omitted it
        defaults to the function names.
    out: Array, optional
        Another dask array whose contents will be replaced. Omit to create a
        new one. Note that, unlike in numpy, this setting gives no performance
        benefits whatsoever, but can still be useful  if one needs to preserve
        the references to a previously existing Array.
    concatenate: bool, optional
        If True (the default), the outputs of the ``chunk``/``combine``
        functions are concatenated into a single np.array before being passed
        to the ``combine``/``aggregate`` functions. If False, the input of
        ``combine`` and ``aggregate`` will be either a list of the raw outputs
        of the previous step or a single output, and the function will have to
        concatenate it itself. It can be useful to set this to False if the
        chunk and/or combine steps do not produce np.arrays.
    output_size: int >= 1, optional
        Size of the output of the ``aggregate`` function along the reduced
        axes. Ignored if keepdims is False.
    weights : array_like, optional
        Weights to be used in the reduction of `x`. Will be
        automatically broadcast to the shape of `x`, and so must have
        a compatible shape. For instance, if `x` has shape ``(3, 4)``
        then acceptable shapes for `weights` are ``(3, 4)``, ``(4,)``,
        ``(3, 1)``, ``(1, 1)``, ``(1)``, and ``()``.

    Returns
    -------
    dask array

    **Function Parameters**

    x_chunk: numpy.ndarray
        Individual input chunk. For ``chunk`` functions, it is one of the
        original chunks of x. For ``combine`` and ``aggregate`` functions, it's
        the concatenation of the outputs produced by the previous ``chunk`` or
        ``combine`` functions. If concatenate=False, it's a list of the raw
        outputs from the previous functions.
    weights_chunk: numpy.ndarray, optional
        Only applicable to the ``chunk`` function. Weights, with the
        same shape as `x_chunk`, to be applied during the reduction of
        the individual input chunk. If ``weights`` have not been
        provided then the function may omit this parameter. When
        `weights_chunk` is included then it must occur immediately
        after the `x_chunk` parameter, and must also have a default
        value for cases when ``weights`` are not provided.
    axis: tuple
        Normalized list of axes to reduce upon, e.g. ``(0, )``
        Scalar, negative, and None axes have been normalized away.
        Note that some numpy reduction functions cannot reduce along multiple
        axes at once and strictly require an int in input. Such functions have
        to be wrapped to cope.
    keepdims: bool
        Whether the reduction function should preserve the reduced axes or
        remove them.

    """
    # Convert non-dask arrays to dask arrays
    from dask_array._collection import Array

    if not isinstance(x, Array):
        from dask_array.core._conversion import asanyarray

        x = asanyarray(x)

    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, Integral):
        axis = (axis,)
    axis = validate_axis(axis, x.ndim)

    if dtype is None:
        raise ValueError("Must specify dtype")

    if is_series_like(x):
        x = x.values

    # Handle weights broadcasting
    weights_expr = None
    if weights is not None:
        from dask_array._broadcast import broadcast_to
        from dask_array.core._conversion import asanyarray

        wgt = asanyarray(weights)
        try:
            wgt = broadcast_to(wgt, x.shape)
        except ValueError:
            raise ValueError(f"Weights with shape {wgt.shape} are not broadcastable to x with shape {x.shape}")
        weights_expr = wgt.expr

    # Create the Reduction expression
    result = new_collection(
        reduction_cls(
            x.expr,
            chunk,
            aggregate,
            axis,
            keepdims,
            dtype,
            _normalize_split_every(split_every, axis),
            combine,
            name,
            concatenate,
            output_size,
            meta,
            weights_expr,
        )
    )

    # Handle out= parameter
    if out is not None:
        from dask_array.core._blockwise_funcs import _handle_out

        return _handle_out(out, result)
    return result


def _sliding_window_nan_skip_reduce_block(block, *, window, sliding_axis, out_dtype, identity, binop):
    out_len = block.shape[sliding_axis] - window + 1
    index = [slice(None)] * block.ndim
    index[sliding_axis] = slice(0, out_len)
    first = block[tuple(index)]
    out = np.full(first.shape, identity, dtype=out_dtype)

    for offset in range(window):
        index[sliding_axis] = slice(offset, offset + out_len)
        other = block[tuple(index)]
        valid = ~np.isnan(other)
        value = np.where(valid, other, identity)
        binop(out, value, out=out, casting="unsafe")

    return out


def _sliding_window_variance_mean_dtype(input_dtype, out_dtype):
    out_dtype = np.dtype(out_dtype)
    if np.issubdtype(out_dtype, np.inexact):
        return np.result_type(input_dtype, out_dtype)
    return np.result_type(input_dtype, np.float64)


def _sliding_window_variance_m2_dtype(out_dtype):
    out_dtype = np.dtype(out_dtype)
    if np.issubdtype(out_dtype, np.inexact):
        return out_dtype
    return np.result_type(out_dtype, np.float64)


class Sum(Reduction):
    sliding_window_reducer = "sum"
    sliding_window_binop = np.add
    sliding_window_binop_kwargs = {"casting": "unsafe"}


class Prod(Reduction):
    sliding_window_reducer = "prod"
    sliding_window_binop = np.multiply
    sliding_window_binop_kwargs = {"casting": "unsafe"}


class Min(Reduction):
    sliding_window_reducer = "min"
    sliding_window_binop = np.minimum


class Max(Reduction):
    sliding_window_reducer = "max"
    sliding_window_binop = np.maximum


class Any(Reduction):
    sliding_window_reducer = "any"
    sliding_window_binop = np.logical_or


class All(Reduction):
    sliding_window_reducer = "all"
    sliding_window_binop = np.logical_and


class Mean(Reduction):
    sliding_window_reducer = "mean"
    sliding_window_binop = np.add
    sliding_window_binop_kwargs = {"casting": "unsafe"}

    @classmethod
    def sliding_window_finalize(cls, out, *, window):
        np.divide(out, window, out=out, casting="unsafe")
        return out


class Var(Reduction):
    sliding_window_reducer = "var"

    @classmethod
    def sliding_window_reduce_block(
        cls,
        block,
        *,
        window,
        sliding_axis,
        out_dtype,
        ddof=0,
        implicit_complex_dtype=False,
    ):
        out_len = block.shape[sliding_axis] - window + 1
        index = [slice(None)] * block.ndim
        index[sliding_axis] = slice(0, out_len)
        input_is_complex = np.issubdtype(block.dtype, np.complexfloating)
        explicit_real_complex = (
            input_is_complex and not implicit_complex_dtype and not np.issubdtype(out_dtype, np.complexfloating)
        )
        if explicit_real_complex:
            first = block[tuple(index)]
            mean_dtype = _sliding_window_variance_mean_dtype(first.real.dtype, out_dtype)
            m2_dtype = _sliding_window_variance_m2_dtype(out_dtype)
            mean = np.array(first.real, dtype=mean_dtype, copy=True)
            m2 = np.zeros(mean.shape, dtype=m2_dtype)
            imag_m2 = np.array(first.imag * first.imag, dtype=m2_dtype, copy=True)
        else:
            mean_dtype = _sliding_window_variance_mean_dtype(block.dtype, out_dtype)
            m2_dtype = _sliding_window_variance_m2_dtype(out_dtype)
            mean = np.array(block[tuple(index)], dtype=mean_dtype, copy=True)
            m2 = np.zeros(mean.shape, dtype=m2_dtype)
            imag_m2 = None
        count = 1

        for offset in range(1, window):
            index[sliding_axis] = slice(offset, offset + out_len)
            raw_value = block[tuple(index)]
            if explicit_real_complex:
                value = np.asarray(raw_value.real, dtype=mean.dtype)
                imag = np.asarray(raw_value.imag, dtype=m2.dtype)
                imag_m2 += imag * imag
            else:
                value = np.asarray(raw_value, dtype=mean.dtype)
            count += 1
            delta = value - mean
            mean += delta / count
            delta2 = value - mean
            if not explicit_real_complex and np.issubdtype(mean.dtype, np.complexfloating):
                m2 += np.real(np.conj(delta) * delta2)
            else:
                m2 += delta * delta2

        denominator = count - ddof
        if denominator < 0:
            return np.full_like(m2, np.nan, dtype=out_dtype)

        with np.errstate(divide="ignore", invalid="ignore"):
            out = (m2 + imag_m2 if imag_m2 is not None else m2) / denominator
        return np.asarray(out, dtype=out_dtype)


class NanSum(Reduction):
    sliding_window_reducer = "nansum"

    @classmethod
    def sliding_window_reduce_block(
        cls,
        block,
        *,
        window,
        sliding_axis,
        out_dtype,
        ddof=0,
        implicit_complex_dtype=False,
    ):
        return _sliding_window_nan_skip_reduce_block(
            block, window=window, sliding_axis=sliding_axis, out_dtype=out_dtype, identity=0, binop=np.add
        )


class NanProd(Reduction):
    sliding_window_reducer = "nanprod"

    @classmethod
    def sliding_window_reduce_block(
        cls,
        block,
        *,
        window,
        sliding_axis,
        out_dtype,
        ddof=0,
        implicit_complex_dtype=False,
    ):
        return _sliding_window_nan_skip_reduce_block(
            block, window=window, sliding_axis=sliding_axis, out_dtype=out_dtype, identity=1, binop=np.multiply
        )


class NanMin(Reduction):
    sliding_window_reducer = "nanmin"
    sliding_window_binop = np.fmin


class NanMax(Reduction):
    sliding_window_reducer = "nanmax"
    sliding_window_binop = np.fmax


class NanMean(Reduction):
    sliding_window_reducer = "nanmean"

    @classmethod
    def sliding_window_reduce_block(
        cls,
        block,
        *,
        window,
        sliding_axis,
        out_dtype,
        ddof=0,
        implicit_complex_dtype=False,
    ):
        out_len = block.shape[sliding_axis] - window + 1
        index = [slice(None)] * block.ndim
        index[sliding_axis] = slice(0, out_len)
        first = block[tuple(index)]
        out = np.zeros(first.shape, dtype=out_dtype)
        count = np.zeros(first.shape, dtype=np.int64)

        for offset in range(window):
            index[sliding_axis] = slice(offset, offset + out_len)
            other = block[tuple(index)]
            valid = ~np.isnan(other)
            value = np.where(valid, other, 0)
            np.add(out, value, out=out, casting="unsafe")
            count += valid

        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(out, count, out=out, casting="unsafe")
        return out


class NanVar(Reduction):
    sliding_window_reducer = "nanvar"

    @classmethod
    def sliding_window_reduce_block(
        cls,
        block,
        *,
        window,
        sliding_axis,
        out_dtype,
        ddof=0,
        implicit_complex_dtype=False,
    ):
        out_len = block.shape[sliding_axis] - window + 1
        index = [slice(None)] * block.ndim
        index[sliding_axis] = slice(0, out_len)
        input_is_complex = np.issubdtype(block.dtype, np.complexfloating)
        explicit_real_complex = (
            input_is_complex and not implicit_complex_dtype and not np.issubdtype(out_dtype, np.complexfloating)
        )
        first = block[tuple(index)]
        if explicit_real_complex:
            mean_dtype = _sliding_window_variance_mean_dtype(first.real.dtype, out_dtype)
            m2_dtype = _sliding_window_variance_m2_dtype(out_dtype)
            mean = np.zeros(first.shape, dtype=mean_dtype)
            m2 = np.zeros(first.shape, dtype=m2_dtype)
            imag_m2 = np.zeros(first.shape, dtype=m2_dtype)
        else:
            mean_dtype = _sliding_window_variance_mean_dtype(block.dtype, out_dtype)
            m2_dtype = _sliding_window_variance_m2_dtype(out_dtype)
            mean = np.zeros(first.shape, dtype=mean_dtype)
            m2 = np.zeros(first.shape, dtype=m2_dtype)
            imag_m2 = None
        count = np.zeros(first.shape, dtype=np.int64)

        for offset in range(window):
            index[sliding_axis] = slice(offset, offset + out_len)
            raw_value = block[tuple(index)]
            valid = ~np.isnan(raw_value)
            if explicit_real_complex:
                value = np.asarray(raw_value.real, dtype=mean.dtype)
                imag = np.asarray(raw_value.imag, dtype=m2.dtype)
                imag_m2 += np.where(valid, imag * imag, 0)
            else:
                value = np.asarray(raw_value, dtype=mean.dtype)
            new_count = count + valid
            delta = value - mean
            divisor = np.where(valid, new_count, 1)
            with np.errstate(invalid="ignore"):
                mean += np.where(valid, delta / divisor, 0)
            delta2 = value - mean
            if not explicit_real_complex and np.issubdtype(mean.dtype, np.complexfloating):
                update = np.real(np.conj(delta) * delta2)
            else:
                update = delta * delta2
            m2 += np.where(valid, update, 0)
            count = new_count

        denominator = count - ddof
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (m2 + imag_m2 if imag_m2 is not None else m2) / denominator
        out = np.where(denominator < 0, np.nan, out)
        return np.asarray(out, dtype=out_dtype)


def _normalize_split_every(split_every, axis):
    """Canonical ``{axis: n}`` form. Applied at expression construction so
    equivalent spellings (``2`` vs ``{0: 2}``) tokenize — and so share keys —
    identically; the lowering re-applies it idempotently for direct callers."""
    split_every = split_every or config.get("split_every", 16)
    if isinstance(split_every, dict):
        return {k: split_every.get(k, 2) for k in axis}
    if isinstance(split_every, Integral):
        n = builtins.max(int(split_every ** (1 / (len(axis) or 1))), 2)
        return dict.fromkeys(axis, n)
    raise ValueError("split_every must be a int or a dict")


def _tree_reduce(
    x,
    aggregate,
    axis,
    keepdims,
    dtype,
    split_every=None,
    combine=None,
    name=None,
    concatenate=True,
    reduced_meta=None,
):
    """Perform the tree reduction step of a reduction.

    Lower level, users should use ``reduction`` or ``arg_reduction`` directly.
    """
    return new_collection(
        _build_tree_reduce_expr(
            x, aggregate, axis, keepdims, dtype, split_every, combine, name, concatenate, reduced_meta
        )
    )


def _build_tree_reduce_expr(
    x,
    aggregate,
    axis,
    keepdims,
    dtype,
    split_every,
    combine,
    name,
    concatenate,
    reduced_meta,
):
    """Build tree reduction cascade of PartialReduce expressions.

    Shared implementation used by both Reduction._build_tree_reduce and _tree_reduce.
    """
    split_every = _normalize_split_every(split_every, axis)

    # Compute tree depth
    depth = 1
    for i, n in enumerate(x.numblocks):
        if i in split_every and split_every[i] != 1:
            depth = int(builtins.max(depth, math.ceil(math.log(n, split_every[i]))))

    # Build combine function
    func = partial(combine or aggregate, axis=axis, keepdims=True)
    if concatenate:
        func = compose(func, partial(_concatenate2, axes=sorted(axis)))

    # Build intermediate PartialReduce layers
    for _ in range(depth - 1):
        x = PartialReduce(
            x,
            func,
            split_every,
            True,
            dtype=dtype,
            name=(name or funcname(combine or aggregate)) + "-partial",
            reduced_meta=reduced_meta,
        )

    # Build final aggregate function
    agg_func = partial(aggregate, axis=axis, keepdims=keepdims)
    if concatenate:
        agg_func = compose(agg_func, partial(_concatenate2, axes=sorted(axis)))

    # Final aggregation layer
    return PartialReduce(
        x,
        agg_func,
        split_every,
        keepdims=keepdims,
        dtype=dtype,
        name=(name or funcname(aggregate)) + "-aggregate",
        reduced_meta=reduced_meta,
    )


def _accept_slice_impl(slice_expr, input_array, reduced_axes, keepdims, make_result):
    """Shared implementation for slice pushdown through reductions.

    Parameters
    ----------
    slice_expr : SliceSlicesIntegers
        The slice expression being pushed through
    input_array : ArrayExpr
        The input array to the reduction
    reduced_axes : set
        Set of axes being reduced
    keepdims : bool
        Whether the reduction keeps dimensions
    make_result : callable(sliced_input, input_index) -> expr
        Factory function to create the result expression

    Returns
    -------
    expr or None
        The transformed expression, or None if slice cannot be pushed through
    """
    from dask_array.slicing import SliceSlicesIntegers

    index = slice_expr.index

    # Don't handle None/newaxis
    if any(idx is None for idx in index):
        return None

    input_ndim = input_array.ndim

    if keepdims:
        # With keepdims, output has same ndim as input
        full_index = index + (slice(None),) * (input_ndim - len(index))
    else:
        # Without keepdims, reduced axes are removed from output
        out_axis = [i for i in range(input_ndim) if i not in reduced_axes]
        output_ndim = len(out_axis)
        full_index = index + (slice(None),) * (output_ndim - len(index))

    # Convert integers to size-1 slices to preserve dimensions
    slice_index = tuple(slice(idx, idx + 1) if isinstance(idx, Integral) else idx for idx in full_index)
    has_integers = any(isinstance(idx, Integral) for idx in full_index)

    # Build input index mapping output axes to input axes
    if keepdims:
        input_index = slice_index
    else:
        input_index = []
        out_pos = 0
        for in_ax in range(input_ndim):
            if in_ax in reduced_axes:
                input_index.append(slice(None))
            else:
                input_index.append(slice_index[out_pos])
                out_pos += 1
        input_index = tuple(input_index)

    # Apply the slice to the input
    sliced_input = new_collection(input_array)[input_index]

    # Don't push slice through if it would create empty arrays on non-reduced axes
    for ax in range(input_ndim):
        if ax not in reduced_axes and sliced_input.shape[ax] == 0:
            return None

    result = make_result(sliced_input, input_index)

    # If we converted integers to slices, extract with [0] to restore dimensions
    if has_integers:
        extract_index = tuple(0 if isinstance(idx, Integral) else slice(None) for idx in full_index)
        return SliceSlicesIntegers(result, extract_index, slice_expr.allow_getitem_optimization)

    return result


class PartialReduce(ArrayExpr):
    _parameters = [
        "array",
        "func",
        "split_every",
        "keepdims",
        "dtype",
        "name",
        "reduced_meta",
    ]
    _defaults = {
        "keepdims": False,
        "dtype": None,
        "name": None,
        "reduced_meta": None,
    }

    def __dask_tokenize__(self):
        if not self._determ_token:
            # TODO: Is there an actual need to overwrite this?
            self._determ_token = _tokenize_deterministic(
                self.func, self.array, self.split_every, self.keepdims, self.dtype
            )
        return self._determ_token

    @cached_property
    def _name(self):
        return (self.operand("name") or funcname(self.func)) + "-" + self.deterministic_token

    @cached_property
    def dtype(self):
        # Use the explicitly passed dtype parameter instead of inferring from meta
        if self.operand("dtype") is not None:
            return np.dtype(self.operand("dtype"))
        return super().dtype

    @cached_property
    def chunks(self):
        chunks = [
            (tuple(1 for p in partition_all(self.split_every[i], c)) if i in self.split_every else c)
            for (i, c) in enumerate(self.array.chunks)
        ]

        if not self.keepdims:
            out_axis = [i for i in range(self.array.ndim) if i not in self.split_every]
            getter = lambda k: get(out_axis, k)
            chunks = list(getter(chunks))
        return tuple(chunks)

    @cached_property
    def transfer_bytes(self):
        # See ArrayExpr.transfer_bytes.  Each output block combines a group of
        # split_every input blocks; the largest of each group hosts the
        # combine under min, everything is fetched remotely under max.
        from dask_array._expr import TransferBytes

        x = self.array
        if any(math.isnan(c) for dim in x.chunks for c in dim):
            return TransferBytes(math.nan, math.nan)
        largest = 1.0
        for i, chunks in enumerate(x.chunks):
            if i in self.split_every:
                largest *= sum(max(group) for group in partition_all(self.split_every[i], chunks))
            else:
                largest *= sum(chunks)
        nbytes = x.nbytes
        return TransferBytes(nbytes - x.dtype.itemsize * largest, nbytes)

    def _layer(self):
        x = self.array
        parts = [list(partition_all(self.split_every.get(i, 1), range(n))) for (i, n) in enumerate(x.numblocks)]
        keys = product(*map(range, map(len, parts)))
        if not self.keepdims:
            out_axis = [i for i in range(x.ndim) if i not in self.split_every]
            getter = lambda k: get(out_axis, k)
            keys = map(getter, keys)
        dsk = {}
        for k, p in zip(keys, product(*parts)):
            free = {i: j[0] for (i, j) in enumerate(p) if len(j) == 1 and i not in self.split_every}
            dummy = dict(i for i in enumerate(p) if i[0] in self.split_every)
            g = lol_tuples((x.name,), range(x.ndim), free, dummy)
            dsk[(self._name,) + k] = (self.func, g)

        return dsk

    def _frisky_layer(self):
        """Describe this PartialReduce as a PartialReduceLayer for direct task
        emission. The Rust layer reproduces dask's lol_tuples nesting, so the
        structure matches ``_layer`` exactly."""
        from dask_array._frisky import PartialReduceLayer

        x = self.array
        steps = [self.split_every.get(i, 0) for i in range(x.ndim)]
        return PartialReduceLayer(self._name, self.func, x.name, list(x.numblocks), steps, self.keepdims)

    @property
    def _meta(self):
        meta = self.array._meta
        original_dtype = getattr(self.reduced_meta, "dtype", None) or getattr(meta, "dtype", None)

        if self.reduced_meta is not None:
            try:
                meta = self.func(self.reduced_meta, computing_meta=True)
            except TypeError:
                # No computing_meta kwarg, try without it
                try:
                    meta = self.func(self.reduced_meta)
                except ValueError as e:
                    if "zero-size array to reduction operation" in str(e):
                        meta = self.reduced_meta
                except IndexError:
                    meta = self.reduced_meta
            except (ValueError, IndexError):
                # Can't compute on empty array (ufunc, argtopk, etc.)
                meta = self.reduced_meta

        # Ensure meta is array-like (func can return Python scalars for object dtype)
        if not is_arraylike(meta) and meta is not None:
            meta = np.array(meta, dtype=original_dtype or object)

        # Reshape meta to match output dimensions
        if is_arraylike(meta) and meta.ndim != len(self.chunks):
            if len(self.chunks) == 0:
                # 0D output - reduce to scalar
                try:
                    meta = meta.sum()
                    if not hasattr(meta, "dtype"):
                        meta = np.array(meta, dtype=original_dtype)
                except TypeError:
                    # dtype doesn't support sum (e.g., datetime64)
                    meta = np.empty((), dtype=meta.dtype)
            else:
                target_shape = (0,) * len(self.chunks)
                # Use np.prod(shape) for array-likes that don't expose .size
                meta_size = getattr(meta, "size", None)
                if meta_size is None:
                    meta_size = np.prod(meta.shape)
                if meta_size != 0:
                    # Can't reshape non-empty array to empty shape (e.g., scalar)
                    meta = np.empty(target_shape, dtype=meta.dtype)
                else:
                    meta = meta.reshape(target_shape)

        # Ensure meta has the correct dtype if dtype is explicitly specified
        if self.operand("dtype") is not None and hasattr(meta, "dtype"):
            target_dtype = np.dtype(self.operand("dtype"))
            if meta.dtype != target_dtype:
                with warnings.catch_warnings():
                    # Suppress ComplexWarning when casting complex to real (e.g., var)
                    warnings.filterwarnings("ignore", category=ComplexWarning)
                    meta = meta.astype(target_dtype)

        # Convert MaskedConstant (np.ma.masked) to a proper MaskedArray
        # since the singleton cannot be tokenized
        if isinstance(meta, np.ma.core.MaskedConstant):
            meta = np.ma.array(meta, ndmin=0)

        return meta

    def _simplify_up(self, parent, dependents):
        """Allow slice operations to push through PartialReduce."""
        from dask_array.slicing import SliceSlicesIntegers

        if isinstance(parent, SliceSlicesIntegers):
            return self._accept_slice(parent)
        return None

    def _accept_slice(self, slice_expr):
        """Accept a slice being pushed through this PartialReduce."""
        reduced_axes = set(self.split_every.keys())

        def make_result(sliced_input, input_index):
            return PartialReduce(
                sliced_input.expr,
                self.func,
                self.split_every,
                self.keepdims,
                self.operand("dtype"),
                self.operand("name"),
                self.reduced_meta,
            )

        return _accept_slice_impl(slice_expr, self.array, reduced_axes, self.keepdims, make_result)
