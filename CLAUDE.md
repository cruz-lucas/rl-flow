# Repository guide

rl-flow is a JAX-based framework for composing reinforcement-learning experiments as node-graph
workflows, then validating, compiling, and running them into reproducible run directories. It is a
uv + pnpm monorepo.

## Layout

| Path | What it is |
|------|------------|
| `packages/core/rlflow/` | Core package `rlflow`: graph model, compiler, registry, schemas, execution, storage, tracking, analysis, CLI |
| `packages/builtin_components/rlflow_builtin/` | Built-in components `rlflow_builtin`: tabular + DQN agents, environments (gridworld/riverswim/sixarms/navix), policies, buffers, runners |
| `apps/api/rlflow_api/` | FastAPI backend (`create_app` in `main.py`) |
| `apps/web/` | React + Vite frontend (`@rl-flow/web`) |
| `configs/` | Workflow / sweep / cluster YAML configs |
| `docs/` | MkDocs Material sources |
| `tests/` | pytest suite (`unit/`, `integration/`, `analysis/`, `e2e/`) |

## Commands

Everything is wrapped in the `Makefile` — run `make help`. The common ones:

```bash
make install        # uv sync (dev + analysis) and pnpm install
make test           # uv run pytest
make lint           # ruff check
make format         # ruff format + ruff check --fix
make check          # lint + format-check + tests (the pre-PR gate)
make docs           # generate reference pages + mkdocs build --strict
make web            # start the web dev server
```

Run an experiment end-to-end:

```bash
uv run rlflow workflow validate configs/workflows/tabular_q_learning_riverswim.yaml
uv run rlflow run configs/workflows/tabular_q_learning_riverswim.yaml --backend local
```

## Conventions

- Invoke the CLI as `uv run rlflow ...` (a `rlflow` console script is declared in `pyproject.toml`).
- Python is linted and formatted with **ruff** (config in `pyproject.toml`); keep `make check` green.
- CI (`.github/workflows/ci.yml`) runs ruff + pytest and the web build/tests on every PR.
- Reproducibility is a first-class feature: every run writes a `manifest.json`
  (`rlflow/tracking/manifest.py`) capturing git SHA, seeds, dependency versions, and config hashes.
  Don't regress it.

## Gotchas

- The pinned JAX is newer than any stable `tensorflow-probability` release, so navix (via rlax/distrax)
  requires a **tfp nightly** — pinned as `tfp-nightly` in `pyproject.toml`. If `uv sync` can't find that
  nightly (nightlies are eventually purged from PyPI), bump it to a newer nightly compatible with the
  pinned JAX rather than removing it.
- Run heavy JAX tests on CPU with `JAX_PLATFORMS=cpu` for determinism/speed.
