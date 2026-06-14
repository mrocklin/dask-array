"""Distinct-data differential for the records path (`to_task_records`).

For each collection, compares the Frisky records path against the dask path
*per key* with distinct (arange) data — so a mis-sliced / mis-ordered block
assembly is caught (which `da.ones`-based checks can't do). The records are run
through a local resolver that mirrors the Frisky worker (`resolve_futures`):
`TaskRef` -> data, recursing into nested lists/tuples/dicts. No cluster needed,
so it's a fast (<1s) feedback loop, and it's layer-agnostic — reuse it as new
layers land.

    PYTHONPATH=/Users/mrocklin/workspace/dask-array \
      MATURIN_IMPORT_HOOK_ENABLED=0 .venv/bin/python bench/diff_layers.py
"""

import numpy as np
from dask.core import flatten
from dask.local import get_sync
from dask._task_spec import TaskRef

import dask_array as da


def _resolve(arg, cache):
    if isinstance(arg, TaskRef):
        return cache[str(arg.key)]
    if isinstance(arg, list):
        return [_resolve(a, cache) for a in arg]
    if isinstance(arg, tuple):
        return tuple(_resolve(a, cache) for a in arg)
    if isinstance(arg, dict):
        return {k: _resolve(v, cache) for k, v in arg.items()}
    return arg


def records_per_key(collection, output_keys):
    """Collect records over the lowered tree; materialize uncovered leaves (e.g.
    from_array) via the dask path; run the records with a worker-style resolver.
    Returns {key_str: value} for output_keys."""
    expr = collection.expr.lower_completely()
    records, cache = [], {}
    stack, seen = [expr], set()
    while stack:
        e = stack.pop()
        if e._name in seen:
            continue
        seen.add(e._name)
        layer = getattr(e, "_frisky_layer", None)
        covered = False
        if layer is not None:
            try:
                records.extend(layer().to_task_records())
                covered = True
            except NotImplementedError:
                covered = False
        if covered:
            stack.extend(e.dependencies())
        else:
            # Leaf (no frisky layer): provide its blocks as inputs via dask.
            g = dict(e._layer())
            for k, v in zip(g, get_sync(g, list(g))):
                cache[str(k)] = v

    produced = {r[0]: r for r in records}

    def compute(key):
        if key in cache:
            return cache[key]
        _k, func, args, kwargs, deps = produced[key]
        for d in deps:
            compute(d)
        cache[key] = func(*[_resolve(a, cache) for a in args], **kwargs)
        return cache[key]

    return {k: compute(k) for k in output_keys}


def cases():
    def arr(shape, chunks):
        base = np.arange(int(np.prod(shape)), dtype="f8").reshape(shape)
        return da.from_array(base, chunks=chunks)

    # rechunk (records path needs from_array leaf -> materialized via dask)
    yield "rechunk 1d split", arr((12,), (4,)).rechunk((3,))
    yield "rechunk 1d merge", arr((20,), (5,)).rechunk((7,))
    yield "rechunk 2d irregular", arr((10, 10), (3, 3)).rechunk((4, 5))
    yield "rechunk 2d transpose-ish", arr((8, 8), (1, 8)).rechunk((8, 1))
    yield "rechunk multistep", arr((16, 50), (16, 1)).rechunk((3, 10))
    yield "rechunk 3d", arr((9, 8, 7), (2, 3, 4)).rechunk((3, 2, 5))
    yield "rechunk to single", arr((12, 12), (4, 4)).rechunk((12, 12))
    # reductions
    yield "sum axis0", arr((9, 6), (2, 2)).sum(axis=0)
    yield "sum axis1", arr((9, 6), (2, 2)).sum(axis=1)
    yield "sum all", arr((9, 6), (2, 2)).sum()
    yield "prod axis (0,2)", arr((4, 3, 5), (2, 3, 2)).prod(axis=(0, 2))
    yield "max keepdims", arr((7, 5), (3, 2)).max(axis=1, keepdims=True)
    # chained
    yield "rechunk then sum", arr((12, 12), (4, 4)).rechunk((6, 3)).sum(axis=0)
    yield "sum of rechunk 3d", arr((6, 6, 6), (2, 2, 2)).rechunk((3, 3, 3)).sum(axis=1)
    # squeeze
    yield "squeeze axis=0", arr((1, 6, 4), (1, 3, 2)).squeeze(axis=0)
    yield "squeeze axis=1", arr((4, 1, 6), (2, 1, 3)).squeeze(axis=1)
    yield "squeeze multi-axis", arr((1, 4, 1, 6), (1, 2, 1, 3)).squeeze(axis=(0, 2))
    yield "squeeze all size-1", arr((1, 4, 1), (1, 2, 1)).squeeze()
    # broadcast_to
    yield "broadcast_to new leading dim", da.broadcast_to(arr((4, 6), (2, 3)), (3, 4, 6))
    yield "broadcast_to size-1 dim", da.broadcast_to(arr((1, 6), (1, 3)), (4, 6))
    yield "broadcast_to both new and size-1", da.broadcast_to(arr((1, 6), (1, 3)), (5, 4, 6))
    yield "broadcast_to passthrough", da.broadcast_to(arr((4, 6), (2, 3)), (4, 6))
    # expand_dims
    yield "expand_dims axis0", da.expand_dims(arr((6, 4), (3, 2)), axis=0)
    yield "expand_dims axis1", da.expand_dims(arr((6, 4), (3, 2)), axis=1)
    yield "expand_dims axis-1", da.expand_dims(arr((6, 4), (3, 2)), axis=-1)
    yield "expand_dims multi", da.expand_dims(arr((6, 4), (3, 2)), axis=(0, 2))
    yield "expand_dims 3d", da.expand_dims(arr((4, 3, 5), (2, 3, 2)), axis=1)
    # concatenate
    yield "concatenate axis=0", da.concatenate([arr((6, 4), (3, 2)), arr((9, 4), (3, 2))], axis=0)
    yield "concatenate axis=1", da.concatenate([arr((4, 6), (2, 3)), arr((4, 9), (2, 3))], axis=1)
    yield (
        "concatenate 3 arrays",
        da.concatenate([arr((4, 6), (2, 3)), arr((4, 3), (2, 3)), arr((4, 9), (2, 3))], axis=1),
    )
    yield "concatenate 3d axis0", da.concatenate([arr((6, 4, 5), (3, 2, 5)), arr((9, 4, 5), (3, 2, 5))], axis=0)
    # stack
    yield "stack axis=0", da.stack([arr((4, 6), (2, 3)), arr((4, 6), (2, 3))], axis=0)
    yield "stack axis=1", da.stack([arr((4, 6), (2, 3)), arr((4, 6), (2, 3))], axis=1)
    yield "stack axis=2", da.stack([arr((4, 6), (2, 3)), arr((4, 6), (2, 3))], axis=2)
    yield "stack 3 arrays", da.stack([arr((4, 6), (2, 3)), arr((4, 6), (2, 3)), arr((4, 6), (2, 3))], axis=0)
    # basic slicing / getitem
    yield "slice 1d basic", arr((12,), (4,))[2:10]
    yield "slice 1d step", arr((20,), (5,))[1:18:2]
    yield "slice 1d full step", arr((20,), (5,))[::3]
    yield "slice 1d neg step full", arr((20,), (5,))[::-1]
    yield "slice 1d neg step partial", arr((20,), (5,))[15:3:-2]
    yield "slice 1d neg bounds", arr((12,), (4,))[-8:-2]
    yield "slice 2d", arr((10, 10), (3, 3))[2:8, 1:9]
    yield "slice 2d step", arr((12, 12), (4, 4))[::2, 1::3]
    yield "slice 2d neg step", arr((10, 10), (3, 3))[::-1, 2:9]
    yield "integer index 1d", arr((12,), (4,))[5]
    yield "integer index 1d neg", arr((12,), (4,))[-3]
    yield "integer index 2d row", arr((10, 10), (3, 3))[4]
    yield "integer + slice", arr((10, 10), (3, 3))[3, 2:8]
    yield "slice + integer", arr((10, 10), (3, 3))[2:8, 3]
    yield "scalar index 2d", arr((10, 10), (3, 3))[3, 4]
    yield "slice 3d mixed", arr((9, 8, 7), (2, 3, 4))[1:8, 5, ::2]
    yield "slice then slice (fusion)", arr((20,), (5,))[2:18][1:10]
    yield "slice partial alias", arr((10, 10), (3, 3))[:, 1:9]
    # blocks (x.blocks[...] — pure block-index alias)
    yield "blocks single", arr((12,), (4,)).blocks[1]
    yield "blocks slice", arr((20,), (5,)).blocks[1:3]
    yield "blocks reorder", arr((20,), (5,)).blocks[[3, 0, 2, 1]]
    yield "blocks repeat", arr((20,), (5,)).blocks[[0, 0, 2, 2]]
    yield "blocks 2d", arr((12, 12), (4, 4)).blocks[1:, ::2]
    yield "blocks 2d single+slice", arr((12, 12), (4, 4)).blocks[2, 0:2]
    # coarsen (per-block reduction over fixed neighborhoods)
    yield "coarsen 1d sum f2", da.coarsen(np.sum, arr((12,), (4,)), {0: 2})
    yield "coarsen 1d max f3", da.coarsen(np.max, arr((12,), (6,)), {0: 3})
    yield "coarsen 2d both axes", da.coarsen(np.sum, arr((12, 8), (4, 4)), {0: 2, 1: 2})
    yield "coarsen 2d one axis", da.coarsen(np.mean, arr((12, 8), (6, 4)), {0: 3})
    yield "coarsen 2d multichunk", da.coarsen(np.sum, arr((24, 16), (4, 8)), {0: 4, 1: 2})
    yield "coarsen 3d", da.coarsen(np.sum, arr((8, 6, 4), (4, 3, 2)), {0: 2, 2: 2})
    # arange (1-D indexed creation; distinct values are inherent)
    yield "arange basic", da.arange(20, chunks=6)
    yield "arange int step", da.arange(5, 50, 3, chunks=7)
    yield "arange float step", da.arange(0, 10, 0.5, chunks=4)
    yield "arange single chunk", da.arange(8, chunks=8)
    yield "arange neg start", da.arange(-7, 7, 2, chunks=3)
    # linspace (1-D indexed creation)
    yield "linspace endpoint", da.linspace(0, 10, 20, chunks=6)
    yield "linspace no endpoint", da.linspace(0, 10, 20, endpoint=False, chunks=6)
    yield "linspace float range", da.linspace(-2.5, 7.5, 17, chunks=5)
    yield "linspace single chunk", da.linspace(0, 1, 8, chunks=8)
    yield "linspace many chunks", da.linspace(1, 100, 50, chunks=7)
    # eye (2-D indexed creation; per-block np.eye / np.zeros choice)
    yield "eye main diag", da.eye(8, chunks=3)
    yield "eye pos k", da.eye(10, chunks=4, k=3)
    yield "eye neg k", da.eye(10, chunks=4, k=-5)
    yield "eye off-square M>N", da.eye(6, chunks=2, M=12, k=2)
    yield "eye off-square M<N", da.eye(12, chunks=4, M=6, k=-3)
    yield "eye k beyond matrix", da.eye(9, chunks=3, k=50)
    # diag (1-D -> 2-D matrix uses per-task-kwargs zeros_like; 2-D -> 1-D extract)
    yield "diag 1d->2d", da.diag(arr((12,), (4,)))
    yield "diag 1d->2d irregular", da.diag(arr((10,), (3,)))
    yield "diag 1d->2d single", da.diag(arr((5,), (5,)))
    yield "diag 2d->1d", da.diag(arr((12, 12), (4, 4)))
    yield "diag 2d->1d irregular", da.diag(arr((10, 10), (3, 3)))
    # reshape (per-block M.reshape; 1:1 C-order block mapping)
    yield "reshape merge 2d->1d", arr((6, 8), (3, 4)).reshape(48)
    yield "reshape split 1d->2d", arr((24,), (6,)).reshape(4, 6)
    yield "reshape ravel", arr((6, 8), (3, 4)).ravel()
    yield "reshape 3d merge", arr((4, 3, 5), (2, 3, 5)).reshape(12, 5)
    yield "reshape single block", arr((6,), (6,)).reshape(2, 3)
    yield "reshape_blockwise collapse", da.reshape_blockwise(arr((3, 3, 3), (3, 2, (2, 1))), (3, 9))
    # arg-reduction chunk step (argmin/argmax, non-ravel; combine = PartialReduce)
    yield "argmin axis0", arr((9, 6), (2, 2)).argmin(axis=0)
    yield "argmax axis1", arr((9, 6), (2, 2)).argmax(axis=1)
    yield "argmin 3d axis1", arr((8, 6, 4), (4, 3, 2)).argmin(axis=1)
    yield "argmin irregular", arr((10, 7), (3, 4)).argmin(axis=0)
    yield "argmax 3d axis2", arr((6, 5, 8), (3, 5, 2)).argmax(axis=2)
    # cumulative (sequential carry: chunk + identity + tail-getitem + binop)
    yield "cumsum axis0", arr((9, 6), (2, 2)).cumsum(axis=0)
    yield "cumsum axis1", arr((9, 6), (2, 2)).cumsum(axis=1)
    yield "cumsum single block axis", arr((6, 9), (6, 2)).cumsum(axis=0)
    yield "cumprod axis0", (arr((8, 5), (3, 2)) % 3 + 1).cumprod(axis=0)
    yield "cumsum 3d axis1", arr((4, 6, 5), (2, 3, 5)).cumsum(axis=1)
    yield "cumsum irregular", arr((10, 7), (3, 4)).cumsum(axis=0)


def main():
    failures = 0
    for label, x in cases():
        tkeys = list(flatten(x.__dask_keys__()))
        output_keys = [str(k) for k in tkeys]
        # Reference: the dask path (suite-validated vs numpy), keyed by str so it
        # lines up with the records' string keys.
        ref = dict(zip(output_keys, get_sync(dict(x.__dask_graph__()), tkeys)))
        try:
            got = records_per_key(x, output_keys)
            ok = all(np.array_equal(np.asarray(got[k]), np.asarray(ref[k])) for k in output_keys)
        except Exception as e:  # noqa: BLE001
            ok = False
            label = f"{label}  [{type(e).__name__}: {str(e)[:50]}]"
        failures += not ok
        print(f"  {'OK ' if ok else 'BAD'} {label}")
    print("\ndistinct-data records diff:", "all good" if not failures else f"{failures} FAILURES")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
