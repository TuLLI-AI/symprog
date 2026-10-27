# Publishing symprog

How to release `symprog` to [PyPI](https://pypi.org/project/symprog/) so that
`pip install symprog` works.

## How packaging is set up

- `pyproject.toml` holds all package metadata. It uses the
  [hatchling](https://hatch.pypa.io/) build backend.
- The version is read from `symprog/version.py` (`__version__`).
- Core dependencies are the packages imported when you `import symprog`. Things
  that are optional (the Streamlit console, Jupyter, JAX) are extras:
  `pip install "symprog[console]"`.
- Tests live in `tests/` and are not shipped in the wheel.

## Build and check locally

```bash
pip install -e ".[dev]"
pytest                 # all tests should pass
make build             # same as: python -m build  -> dist/*.whl, dist/*.tar.gz
twine check dist/*     # validates the README renders on PyPI
```

To check the built wheel in a clean environment:

```bash
python -m venv /tmp/symprog-check
/tmp/symprog-check/bin/pip install dist/symprog-*.whl
/tmp/symprog-check/bin/python -c "from symprog import Solver; print(Solver)"
```

## One-time setup: trusted publishing

Releases are uploaded by GitHub Actions (`.github/workflows/publish.yml`) using
PyPI [trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API
token is ever stored.

1. Create an account on <https://pypi.org> (and enable 2FA).
2. Go to *Your account → Publishing → Add a new pending publisher* and enter:
   - PyPI project name: `symprog`
   - Owner: `TuLLI-AI`
   - Repository: `symprog`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. In the GitHub repo, go to *Settings → Environments* and create an environment
   named `pypi`. Optionally require a reviewer before it can run.

The first successful publish creates the `symprog` project on PyPI and links it
to the repository. You can then add other maintainers on PyPI.

Optional dry run: repeat steps 1–2 on <https://test.pypi.org>. Then add
`with: repository-url: https://test.pypi.org/legacy/` to the publish step.

## Cutting a release

1. Bump `__version__` in `symprog/version.py`. Use pre-release versions such as
   `0.1.0a1` and `0.1.0b1` while the API is still changing. While only
   pre-releases exist, `pip install symprog` installs the newest one. Once a
   final release such as `0.1.0` exists, pip skips pre-releases unless users
   pass `--pre` or pin the exact version.
2. Commit, then tag and push:
   ```bash
   git tag v0.1.0a1
   git push origin main --tags
   ```
3. On GitHub, create a *Release* from the tag. Publishing the release triggers
   the workflow, which builds the package and uploads it to PyPI.

## Manual upload (fallback)

If you can't use GitHub Actions, create a PyPI API token and run:

```bash
make build
twine upload dist/*
```
