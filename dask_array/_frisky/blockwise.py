"""Same-grid / broadcast elementwise blockwise layer.

``Blockwise._frisky_layer`` normalizes the operands to the
``("literal", value)`` / ``("array", dep_name, ind, numblocks)`` form, index
labels as ints aligned with ``out_ind``, and rejects anything outside the
supported subset. The Rust ``BlockwiseLayer`` holds the func/kwargs/literals and
does the block-id math.
"""

from __future__ import annotations

from dask_array import _rust
from dask_array._frisky.base import Layer


class _BlockwiseBoundArgs:
    __slots__ = ("func", "bound", "nargs")

    def __init__(self, func, bound, nargs):
        self.func = func
        self.bound = dict(bound)
        self.nargs = nargs

    def __call__(self, *args, **kwargs):
        out = []
        j = 0
        for i in range(self.nargs):
            if i in self.bound:
                out.append(self.bound[i])
            else:
                out.append(args[j])
                j += 1
        return self.func(*out, **kwargs)

    def __reduce__(self):
        return (_BlockwiseBoundArgs, (self.func, self.bound, self.nargs))


class BlockwiseLayer(Layer):
    def __init__(self, name, func, numblocks, out_ind, args, kwargs=None):
        bound = {}
        remaining = []
        for pos, item in enumerate(args):
            if item[0] == "literal":
                value = item[1]
                fits_scalar = (
                    (type(value) is int and -(1 << 63) <= value < (1 << 63))
                    or type(value) is float
                )
                if not fits_scalar:
                    bound[pos] = value
                    continue
            remaining.append(item)
        if bound:
            func = _BlockwiseBoundArgs(func, bound, len(args))
            args = tuple(remaining)
        self._rust = _rust.BlockwiseLayer(name, func, kwargs or {}, list(numblocks), list(out_ind), list(args))
