# Component Authoring

A component is a `ComponentSpec` with ports, defaults, a JSON schema, and optional compile targets.
Use the `source` field to group components by provider, for example `builtin`,
`navix`, or your plugin name. Component IDs should keep the same prefix.

Expose third-party components through the `rlflow.components` Python entry point:

```toml
[project.entry-points."rlflow.components"]
my_components = "my_package.components:components"
```

The callable should return `list[ComponentSpec]`.

```python
from rlflow.schemas.component import ComponentSpec, PortSpec

def components():
    return [
        ComponentSpec(
            id="my.logger",
            source="my_package",
            kind="logger",
            display_name="My Logger",
            output_ports=[PortSpec(name="logger", type="logger")],
            config_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"enabled": {"type": "boolean"}},
            },
            defaults={"enabled": True},
            compile_target={"gin": {"bindings": {"MyLogger.enabled": "enabled"}}},
        )
    ]
```

Do not add frontend code for new RL modules. If the JSON schema is complete, the UI can render the form.
