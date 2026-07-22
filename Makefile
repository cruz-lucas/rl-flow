# rl-flow developer task runner.
#
# Common workflows are wrapped here so contributors don't have to remember the
# underlying uv / pnpm invocations. Run `make help` to see everything.

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Python packages that ship with the repo (used by lint / typecheck targets).
PY_PATHS := packages apps tests scripts

.PHONY: help install install-py install-web \
        test test-cov lint format format-check typecheck check \
        docs docs-serve web web-build web-test bench clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: install-py install-web ## Install all Python + web dependencies

install-py: ## Install Python dependencies (dev extras)
	uv sync --extra dev --extra analysis

install-web: ## Install web dependencies
	pnpm install

test: ## Run the Python test suite
	uv run pytest

test-cov: ## Run the Python test suite with coverage
	uv run pytest --cov=rlflow --cov=rlflow_builtin --cov=rlflow_api --cov-report=term-missing

lint: ## Lint Python sources with ruff
	uv run ruff check $(PY_PATHS)

format: ## Auto-format Python sources with ruff
	uv run ruff format $(PY_PATHS)
	uv run ruff check --fix $(PY_PATHS)

format-check: ## Check formatting without modifying files
	uv run ruff format --check $(PY_PATHS)

typecheck: ## Type-check the core package with mypy
	uv run mypy packages/core

check: lint format-check typecheck test ## Run lint + format-check + typecheck + tests (the pre-PR gate)

docs: ## Build the documentation site (strict)
	uv run python scripts/generate_docs_reference.py
	uv run mkdocs build --strict

docs-serve: ## Serve the documentation site locally
	uv run mkdocs serve

web: ## Start the web dev server
	pnpm --filter @rl-flow/web dev

web-build: ## Build the web app
	pnpm --filter @rl-flow/web build

web-test: ## Run the web test suite
	pnpm --filter @rl-flow/web test

web-lint: ## Lint + format-check the web app
	pnpm --filter @rl-flow/web lint
	pnpm --filter @rl-flow/web format:check

bench: ## Run the performance benchmark guard
	uv run python scripts/bench.py

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache site apps/web/dist
	find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
