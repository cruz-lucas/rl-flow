"""Seed-parameter helpers.

Kept dependency-free (no pandas) so both the pandas-based analysis layer and the
FastAPI routes can share one implementation without pulling analysis extras into
the API import path.
"""

from __future__ import annotations

from typing import Any


def is_seed_parameter(key: str) -> bool:
    normalized = key.lower()
    return normalized == "seed" or normalized.endswith("_seed") or normalized.endswith(".seed")


def non_seed_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in parameters.items() if not is_seed_parameter(key)}
