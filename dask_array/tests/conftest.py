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
            with dask.config.set({"scheduler": client, "array.rechunk.method": "tasks"}):
                yield "frisky"


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
