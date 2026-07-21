# Contributing

Thanks for contributing to rl-flow. This guide covers local setup and the checks a change must pass.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python env + dependency manager)
- [pnpm](https://pnpm.io/) and Node 20+ (only needed for the web app)
- Python 3.11+

## Setup

```bash
make install          # uv sync (dev + analysis extras) + pnpm install
```

This creates `.venv` and installs the frontend dependencies. To run the CLI or tests without the
Makefile, prefix commands with `uv run` (e.g. `uv run pytest`, `uv run rlflow --help`).

## Development loop

```bash
make test             # run the Python test suite
make lint             # ruff check
make format           # auto-format + apply safe lint fixes
make check            # lint + format-check + tests — run this before opening a PR
```

Frontend:

```bash
make web              # dev server at http://localhost:5173
make web-build        # type-check + production build
make web-test         # vitest
```

Tip: heavy JAX tests run fastest and most deterministically on CPU — `JAX_PLATFORMS=cpu uv run pytest`.

## Style

- **Python** is linted and formatted with [ruff](https://docs.astral.sh/ruff/); configuration lives in
  `pyproject.toml`. The lint baseline is intentionally lenient (see the `ignore` list) and is being
  ratcheted tighter over time — don't add new violations of already-enabled rules.
- Match the surrounding code's style, naming, and comment density.

## Adding a component

Components (agents, environments, policies, buffers, intrinsic rewards) are `ComponentSpec` objects
(`rlflow/schemas/component.py`) exposed via the `rlflow.components` entry-point group. Adding one that
only renders in the UI requires no React changes; making a new *compute* component actually execute
currently also requires wiring in the runner (`rlflow_builtin/runners/tabular_jax.py`) — see the
roadmap for the planned registry-driven dispatch.

## Pull requests

- Keep changes small and focused; one logical change per PR.
- `make check` must pass, and CI (`.github/workflows/ci.yml`) runs the same gates plus the web build.
- Include tests for behavior changes. For refactors that must preserve behavior, add or run a
  fixed-seed numeric-equivalence check.
- Never commit generated artifacts (`runs/`, `site/`, `*.egg-info/`); they are gitignored.
