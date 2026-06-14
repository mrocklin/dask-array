"""Transparent end-to-end: the user calls plain ``dask.compute`` /
``dask.persist`` / ``x.compute()`` — no Frisky API — and (when the graph is
representable) Frisky submits the Rust-generated task records via
``Client.submit_tasks``.

Constructing a Frisky ``Client`` points ``dask.config['scheduler']`` at it and
installs the compute/persist patch. We spy on the Frisky-path helpers to confirm
which calls actually took the records path, and check every result against numpy.

    PYTHONPATH=/Users/mrocklin/workspace/dask-array \
      /Users/mrocklin/workspace/frisky/.venv/bin/python bench/e2e_transparent.py
"""

import numpy as np

import dask
import dask_array as da
import frisky.dask as fdask
from frisky import Client, LocalCluster


def main():
    calls = {"compute": 0, "persist": 0}
    orig_c = fdask._frisky_compute_collections
    orig_p = fdask._frisky_persist_collections

    def spy_c(client, collections):
        out = orig_c(client, collections)
        calls["compute"] += out is not None
        return out

    def spy_p(client, collections):
        out = orig_p(client, collections)
        calls["persist"] += out is not None
        return out

    fdask._frisky_compute_collections = spy_c
    fdask._frisky_persist_collections = spy_p

    failures = 0
    with LocalCluster(n_workers=2) as cluster:
        with Client(cluster.scheduler):
            # dask.compute free function on the RAW (unfused) collection -> records
            before = calls["compute"]
            (got,) = dask.compute(da.ones((7, 5), chunks=(3, 2)) + 1)
            used = calls["compute"] > before
            ok = np.allclose(np.asarray(got), np.full((7, 5), 2.0))
            failures += not (ok and used)
            print(f"  {'OK ' if ok and used else 'BAD'} dask.compute(x)     match={ok} records={used}")

            # dask.persist free function -> records, then compute the result
            before = calls["persist"]
            p = dask.persist(da.ones((6, 6), chunks=(2, 3)) * 3)[0]
            used = calls["persist"] > before
            got = np.asarray(p.compute())
            ok = np.allclose(got, np.full((6, 6), 3.0))
            failures += not (ok and used)
            print(f"  {'OK ' if ok and used else 'BAD'} dask.persist(x)     match={ok} records={used}")

            # two deps + a literal
            (got,) = dask.compute(da.ones((6, 6), chunks=(2, 3)) + da.full((6, 6), 3.0, chunks=(2, 3)))
            ok = np.allclose(np.asarray(got), np.full((6, 6), 4.0))
            failures += not ok
            print(f"  {'OK ' if ok else 'BAD'} dask.compute(a+b)   match={ok}")

            # x.compute() optimizes (fuses) first -> falls back, still correct
            got = np.asarray((da.ones((4, 4), chunks=2) + 1).compute())
            ok = np.allclose(got, np.full((4, 4), 2.0))
            failures += not ok
            print(f"  {'OK ' if ok else 'BAD'} x.compute() (fused) match={ok} (fallback expected)")

    fdask._frisky_compute_collections = orig_c
    fdask._frisky_persist_collections = orig_p
    print("\ntransparent records path:", "all good" if not failures else f"{failures} FAILURES")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
