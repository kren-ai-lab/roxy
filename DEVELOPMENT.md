# Development Guide

## Prerequisites

- Python 3.11–3.14
- [uv](https://docs.astral.sh/uv/) (package manager)

## Setup

```bash
git clone https://github.com/ProteinEngineering-PESB2/roxy_library
cd roxy_library
uv sync --all-extras
```

## Common Tasks

```bash
uv run task format      # sort imports + ruff format
uv run task lint        # ruff check (no fixes)
uv run task lint-fix    # ruff check --fix
uv run task test        # pytest -q
uv run task test-v      # pytest -v
uv run task test-cov    # pytest + HTML coverage report
```

## Running the CLI

```bash
uv run roxy --version
uv run roxy --help
uv run roxy list
```

## Adding a Descriptor Family

1. Create `roxy/descriptors/<family_name>/<module>.py`.
2. Implement a class inheriting `BaseDescriptor` and decorate with `@register`.
3. Import the module in `roxy/descriptors/__init__.py` so the decorator fires.
4. Add tests in `tests/descriptors/<family_name>/`.

See `AGENTS.md` for the full list of families and their source notebooks in `dev_notebooks/`.

## Project Structure

See `AGENTS.md` for the full package layout.

## Branches

- `main` — stable releases
- `refactor/cleanup-base` — current base refactor (S1)
- `feat/descriptors-<family>` — individual descriptor families (S2/S3)
