from rlflow.graph.gin_writer import GinWriter
from rlflow.schemas.component import ComponentSpec


def test_gin_writer_output_is_deterministic() -> None:
    components = {
        "b": ComponentSpec(
            id="b",
            kind="logger",
            display_name="B",
            compile_target={"gin": {"bindings": {"Z.value": "z"}, "base_files": ["base_b.gin"]}},
        ),
        "a": ComponentSpec(
            id="a",
            kind="agent",
            display_name="A",
            compile_target={"gin": {"bindings": {"A.value": "a"}, "base_files": ["base_a.gin"]}},
        ),
    }
    resolved = {"b": {"z": 2}, "a": {"a": "x"}}

    first = GinWriter().render(components, resolved)
    second = GinWriter().render(components, resolved)

    assert first == second
    assert first.index("base_a.gin") < first.index("base_b.gin")
    assert first.index("A.value") < first.index("Z.value")


def test_gin_writer_renders_imports_and_raw_references() -> None:
    components = {
        "agent": ComponentSpec(
            id="agent",
            kind="agent",
            display_name="Agent",
            compile_target={
                "gin": {
                    "imports": ["example_pkg.networks"],
                    "bindings": {
                        "Agent.network": "network",
                        "Agent.observation_dtype": "observation_dtype",
                    },
                }
            },
        )
    }
    resolved = {
        "agent": {
            "network": "@networks.ClassicControlDQNNetwork",
            "observation_dtype": "%jax_networks.CARTPOLE_OBSERVATION_DTYPE",
        }
    }

    rendered = GinWriter().render(components, resolved)

    assert "import example_pkg.networks" in rendered
    assert "Agent.network = @networks.ClassicControlDQNNetwork" in rendered
    assert "Agent.observation_dtype = %jax_networks.CARTPOLE_OBSERVATION_DTYPE" in rendered
