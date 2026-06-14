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
