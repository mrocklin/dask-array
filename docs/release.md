# Release

This project publishes to PyPI as `dask-array`.

The project builds a universal pure-Python wheel. The release workflow builds
once and smoke-tests the wheel and sdist on each supported Python version.
The package version is derived from Git tags.

## One-time PyPI setup

Use PyPI Trusted Publishing rather than a long-lived API token.

Create a pending publisher on PyPI with:

- PyPI project name: `dask-array`
- Owner: `mrocklin`
- Repository: `dask-array`
- Workflow: `publish.yml`
- Environment: `pypi`

Create the matching `pypi` environment in GitHub. Requiring reviewer approval on
that environment is recommended.

## Publish

1. Run the test suite:

   ```bash
   uv run --extra test --extra complete --extra sparse pytest dask_array/tests/ -q
   ```

2. Push the release commit to `main`.
3. Create a local version tag and check the release artifacts:

   ```bash
   git tag 0.1.0
   uv build --sdist --wheel
   uvx twine check dist/*
   ```

4. Push the version tag:

   ```bash
   git push mrocklin 0.1.0
   ```

PyPI files are immutable. If an upload succeeds with bad artifacts or metadata,
release a new version.
