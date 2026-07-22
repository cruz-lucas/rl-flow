"""Smoke test for the performance benchmark harness.

Confirms the benchmark runs end-to-end and produces sane throughput numbers. It
is not a cross-machine regression gate — compare `scripts/bench.py` JSON output
across runs on the same hardware for that.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bench.py"
_spec = importlib.util.spec_from_file_location("rlflow_bench", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


@pytest.mark.bench
def test_bench_tabular_reports_positive_throughput() -> None:
    result = bench.bench_tabular(train_episodes=20, max_episode_steps=20)
    assert result["env_steps"] == 400
    assert result["run_seconds"] > 0
    assert result["steps_per_second"] is not None and result["steps_per_second"] > 0
    assert result["compile_seconds"] >= 0
