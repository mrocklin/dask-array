"""Transparent end-to-end: the user calls plain ``dask.compute`` /
``dask.persist`` / ``x.compute()`` — no Frisky API — and Frisky handles the
graph, preferring scheduler-side expression submission and falling back to
client-side task records.

Constructing a Frisky ``Client`` points ``dask.config['scheduler']`` at it and
installs the compute/persist patch. We spy on both Frisky paths (see
``_spy.py``) to confirm which calls Frisky actually handled, and check every
result against numpy.

    PYTHONPATH=$PWD MATURIN_IMPORT_HOOK_ENABLED=0 \
      /Users/mrocklin/workspace/frisky/.venv/bin/python bench/e2e_transparent.py
"""

import numpy as np

import dask
import dask_array as da
from _spy import frisky_spy
from frisky import Client, LocalCluster


def main():
    failures = 0
    with frisky_spy() as spy:
        with LocalCluster(n_workers=2) as cluster:
            with Client(cluster.scheduler):
                # dask.compute free function on the RAW (unfused) collection
                before = spy.snapshot()
                (got,) = dask.compute(da.ones((7, 5), chunks=(3, 2)) + 1)
                path = spy.path_since(before)
                ok = np.allclose(np.asarray(got), np.full((7, 5), 2.0))
                bad = not (ok and path != "none")
                failures += bad
                print(f"  {'BAD' if bad else 'OK '} dask.compute(x)     match={ok} frisky={path}")

                # dask.persist free function, then compute the result
                before = spy.snapshot()
                p = dask.persist(da.ones((6, 6), chunks=(2, 3)) * 3)[0]
                path = spy.path_since(before)
                got = np.asarray(p.compute())
                ok = np.allclose(got, np.full((6, 6), 3.0))
                bad = not (ok and path != "none")
                failures += bad
                print(f"  {'BAD' if bad else 'OK '} dask.persist(x)     match={ok} frisky={path}")

                # two deps + a literal
                before = spy.snapshot()
                (got,) = dask.compute(da.ones((6, 6), chunks=(2, 3)) + da.full((6, 6), 3.0, chunks=(2, 3)))
                path = spy.path_since(before)
                ok = np.allclose(np.asarray(got), np.full((6, 6), 4.0))
                bad = not (ok and path != "none")
                failures += bad
                print(f"  {'BAD' if bad else 'OK '} dask.compute(a+b)   match={ok} frisky={path}")

                # x.compute() optimizes (fuses) first; FusedBlockwise is covered,
                # so this engages Frisky too
                before = spy.snapshot()
                got = np.asarray((da.ones((4, 4), chunks=2) + 1).compute())
                path = spy.path_since(before)
                ok = np.allclose(got, np.full((4, 4), 2.0))
                bad = not (ok and path != "none")
                failures += bad
                print(f"  {'BAD' if bad else 'OK '} x.compute() (fused) match={ok} frisky={path}")

    print(f"\nengaged: {spy.counts['expression']} expression, {spy.counts['records']} records")
    print("transparent frisky path:", "all good" if not failures else f"{failures} FAILURES")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
