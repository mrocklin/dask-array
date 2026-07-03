from __future__ import annotations

import importlib.util

import dask
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked slow",
    )
    parser.addoption(
        "--scheduler",
        choices=("dask", "frisky", "both"),
        default="dask",
        help="scheduler backend for the dask-array test suite",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_local_scheduler: behavior needs a local (in-process, by-reference) "
        "scheduler; skipped under Frisky, which serializes task data.",
    )
    config.addinivalue_line(
        "markers",
        "xfail_frisky: known Frisky scheduler gap; xfailed only on the Frisky scheduler variant.",
    )
    config.addinivalue_line(
        "markers",
        "skip_frisky: known Frisky scheduler gap; skipped only on the Frisky scheduler variant.",
    )
    scheduler = config.getoption("--scheduler")
    if scheduler in ("frisky", "both") and importlib.util.find_spec("frisky") is None:
        raise pytest.UsageError(
            "--scheduler=frisky requires frisky to be installed in this pytest environment. "
            "Use Frisky's venv, or install Frisky into the dask-array test env."
        )


def pytest_generate_tests(metafunc):
    if "array_scheduler" not in metafunc.fixturenames:
        return
    option = metafunc.config.getoption("--scheduler")
    params = ["dask", "frisky"] if option == "both" else [option]
    metafunc.parametrize("array_scheduler", params, scope="session", indirect=True)


@pytest.fixture(scope="session", autouse=True)
def array_scheduler(request):
    if request.param == "dask":
        with dask.config.set({"scheduler": "sync"}):
            yield "dask"
        return

    import frisky

    with frisky.LocalCluster(n_workers=2, processes=False, dashboard_address="127.0.0.1:0") as cluster:
        with frisky.Client(cluster.scheduler) as client:
            with dask.config.set({"scheduler": client}):
                yield "frisky"


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Tests that need a local scheduler (e.g. da.store mutating a client-side
    # target in place) can't be observed under Frisky, which serializes task
    # data. Skip them only on the frisky variant.
    skip_local_scheduler_on_frisky = pytest.mark.skip(reason="requires a local scheduler; Frisky serializes task data")
    for item in items:
        if "requires_local_scheduler" not in item.keywords:
            continue
        callspec = getattr(item, "callspec", None)
        if callspec and callspec.params.get("array_scheduler") == "frisky":
            item.add_marker(skip_local_scheduler_on_frisky)

    issue = "https://github.com/mrocklin/dask-array/issues/10"
    skip_frisky = pytest.mark.skip(reason=f"known Frisky scheduler gap; see {issue}")
    for item in items:
        if "skip_frisky" not in item.keywords:
            continue
        callspec = getattr(item, "callspec", None)
        if callspec and callspec.params.get("array_scheduler") == "frisky":
            item.add_marker(skip_frisky)

    xfail_frisky = pytest.mark.xfail(reason=f"known Frisky scheduler gap; see {issue}", strict=True)
    for item in items:
        if "xfail_frisky" not in item.keywords:
            continue
        callspec = getattr(item, "callspec", None)
        if callspec and callspec.params.get("array_scheduler") == "frisky":
            item.add_marker(xfail_frisky)
