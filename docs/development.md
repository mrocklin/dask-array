# Development

How to build dask-array locally, run the tests, and cut a release.

## Building

dask-array is **pure Python by default** — you do not need a Rust toolchain to
work on it:

```bash
git clone https://github.com/mrocklin/dask-array
cd dask-array
pip install -e .          # or: uv run <cmd>, uv sync
```

The Python/dask path is fully functional without any native extension. There is
an optional Rust accelerator (`dask_array._rust`) that the Frisky scheduler path
uses; when it is absent the Frisky path falls back to a pure-Python records layer
(`dask_array/_frisky/graph_records.py`), so results are identical — just without
the native fast path.

The build backend is hatchling. `[tool.maturin]` in `pyproject.toml` is read only
by `maturin develop` (below) and is ignored by `pip`/`uv`.

### Building the Rust accelerator (optional)

If you have a Rust toolchain and want the native layers, build the extension into
the source tree with maturin:

```bash
MATURIN_IMPORT_HOOK_ENABLED=0 maturin develop --release
```

This drops `dask_array/_rust.*.so` next to the package; an editable install then
picks it up. Re-run it after changing anything under `crates/dask-array-python/`,
and bump `PROTOCOL_REVISION` (kept in sync between `dask_array/_frisky/base.py`
and the Rust `lib.rs`) when the record surface changes. See AGENTS.md →
"Frisky task-graph support" for the architecture.

## Testing

Default (dask scheduler):

```bash
uv run --extra test --extra complete --extra sparse python -m pytest dask_array/tests/ -q
```

The suite takes a `--scheduler={dask,frisky,both}` option
(`dask_array/tests/conftest.py`). Exercising the Frisky path needs both `frisky`
and `dask_array` (with the native extension) importable in the same interpreter —
i.e. run from Frisky's virtualenv:

```bash
path/to/frisky/.venv/bin/python -m pytest dask_array/tests/ --scheduler=frisky -q \
  --deselect dask_array/tests/test_reductions.py::test_weighted_reduction \
  --deselect dask_array/tests/test_slice_pushdown.py::test_masked_array \
  --deselect dask_array/tests/test_slicing.py::test_slice_masked_arrays
```

The three deselects are pre-existing numpy-masked-array tokenize flakes unrelated
to Frisky. Tests that need a local scheduler — e.g. in-place `da.store`, which a
serializing scheduler can't observe — are marked `requires_local_scheduler` and
skip automatically on the frisky variant.

## Cutting a release

The project publishes to PyPI as `dask-array`. The release workflow
(`.github/workflows/publish.yml`) triggers on an `x.y.z` tag push: it builds a
**universal pure-Python wheel** plus an sdist, smoke-tests both on every supported
Python version, and publishes via PyPI Trusted Publishing.

The version is derived from the git tag (hatch-vcs) — there is no version string
to edit by hand.

> **Today the PyPI wheel is pure Python** (`py3-none-any`) — the native
> accelerator is not yet shipped, so install-from-PyPI users get the pure-Python
> path (Frisky falls back to `GraphRecordsLayer`). To use the native layers,
> build from source with maturin (see above). Shipping native wheels is planned —
> see [Future work](#future-work-ship-native-wheels) below.

### Steps

1. Run the test suite:

   ```bash
   uv run --extra test --extra complete --extra sparse pytest dask_array/tests/ -q
   ```

2. Push the release commit to `main`.
3. Create the version tag and check the artifacts locally:

   ```bash
   git tag 0.1.1
   uv build --sdist --wheel
   uvx twine check dist/*
   ```

4. Push the tag to trigger the publish workflow:

   ```bash
   git push mrocklin 0.1.1
   ```

PyPI files are immutable. If an upload succeeds with bad artifacts or metadata,
release a new version.

### Future work: ship native wheels

Goal: PyPI users get the Rust accelerator without a Rust toolchain, while source
installs on unsupported platforms still work (pip falls back to the pure-Python
sdist).

Sketch:

- Add a per-platform build matrix to `publish.yml` that builds **abi3** wheels
  with maturin (one wheel per OS/arch covers Python ≥ 3.10) for the targets we
  support — Linux x86_64/aarch64, macOS arm64/x86_64, Windows x86_64 — alongside
  the existing pure-Python sdist. `scripts/build-wheels.sh` already drives
  per-platform maturin builds via Docker and is the natural seed (it is currently
  untracked; track it if it becomes part of the pipeline).
- Reconcile the version. The pure-Python wheel/sdist take their version from git
  tags (hatch-vcs); maturin reads it from `crates/dask-array-python/Cargo.toml`.
  The native-wheel build must stamp the Cargo version to match the tag so all
  artifacts agree (`build-wheels.sh` already does this kind of stamping).
- Keep the sdist pure-Python-buildable (it is) so a toolchain-less
  `pip install` from source still succeeds — that is the universal fallback when
  no matching native wheel exists.

Result: `pip install dask-array` gets the native fast path on common platforms and
the pure-Python path everywhere else, with no behavior change — only performance.
