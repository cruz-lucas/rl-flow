from __future__ import annotations


def component_schema(properties: dict, *, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required or [],
        "properties": properties,
    }
