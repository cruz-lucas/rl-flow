import json

from typer.testing import CliRunner

from rlflow.cli import app

runner = CliRunner()


def test_components_describe_lists_config_keys() -> None:
    result = runner.invoke(app, ["components", "describe", "builtin.env.riverswim"])
    assert result.exit_code == 0
    assert "builtin.env.riverswim" in result.stdout
    # Config keys a user needs to author the YAML should be listed.
    assert "num_states" in result.stdout
    assert "p_right" in result.stdout


def test_components_describe_json_is_machine_readable() -> None:
    result = runner.invoke(
        app, ["components", "describe", "builtin.agent.q_learning_tabular", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["id"] == "builtin.agent.q_learning_tabular"
    assert "config_schema" in payload
    assert "defaults" in payload


def test_components_describe_unknown_component_exits_nonzero() -> None:
    result = runner.invoke(app, ["components", "describe", "no.such.component"])
    assert result.exit_code == 1
