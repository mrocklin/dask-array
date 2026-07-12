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

If you have Rust >= 1.83 and want the native layers, build the extension into the
source tree with maturin:

```bash
MATURIN_IMPORT_HOOK_ENABLED=0 maturin develop --release
```

On PowerShell:

```powershell
$env:MATURIN_IMPORT_HOOK_ENABLED = "0"
maturin develop --release
```

This drops the platform-native `dask_array/_rust.*` extension next to the
package; an editable install then picks it up. Re-run it after changing anything
under `crates/dask-array-python/`, and bump `NATIVE_BUILD_GENERATION` (kept in
sync between `dask_array/_frisky/base.py` and the Rust `lib.rs`) after any Rust
change — a local build-freshness check, not the Frisky wire protocol. See
AGENTS.md → "Frisky task-graph support" for the architecture.

## Testing

This section is the canonical home for test invocation — AGENTS.md and
`.ai-docs/testing.md` defer to it, so update commands here first.

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
(`.github/workflows/publish.yml`) triggers on an `x.y.z` tag push and builds:

- a pure-Python **sdist** + **`py3-none-any` wheel** (hatchling/hatch-vcs), and
- native **abi3 wheels** (maturin) for **linux x86_64/aarch64** and
  **macOS arm64** and **Windows x86_64** — one wheel per platform covers
  Python ≥ 3.10.

It smoke-tests the Linux x86_64 native wheel across supported Python versions,
the Windows x86_64 native wheel on Python 3.12, and the sdist (which must stay
pure-Python), then publishes everything via PyPI Trusted Publishing.
`pip install dask-array` then resolves the native wheel where the platform
matches, the `py3-none-any` wheel elsewhere, and the sdist (pure-Python, no Rust
toolchain) as a last resort.

The version comes from the git tag (hatch-vcs); the native-wheel job stamps the
same tag version into `pyproject.toml` before building (maturin doesn't read
hatch-vcs), and the publish job re-checks every artifact against the tag. There is
no version string to edit by hand.

### Steps

1. Run the test suite (and, if touching the native path, the Frisky suite — see
   Testing above):

   ```bash
   uv run --extra test --extra complete --extra sparse pytest dask_array/tests/ -q
   ```

2. Push the release commit to `main`.
3. **Build-only trial.** From the Actions tab, run the *Publish to PyPI* workflow
   manually (`workflow_dispatch`). It builds the full wheel matrix and runs the
   smoke tests **without publishing** (publish is gated on a tag push). Confirm
   all four native build jobs, the pure-Python build job, and the smoke-test
   jobs are green before tagging.
4. Tag and push to publish:

   ```bash
   git tag 0.1.1
   git push mrocklin 0.1.1
   ```

PyPI files are immutable. If an upload succeeds with bad artifacts or metadata,
release a new version.

### Not yet covered

- **Windows ARM64** wheels are not built; those users fall back to the pure-Python
  wheel.
- **Coiled package-sync** does not yet match abi3 wheels, so it won't pick these
  up — that dev flow stays on the private index until it gets its own pipeline.
