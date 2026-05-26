import jax
import pytest

from rlflow.registry.builtin import create_default_registry
from rlflow_builtin.environments.navix import NavixWrapper, create_navix_environment


def test_navix_component_is_registered_under_navix_source() -> None:
    component = create_default_registry(discover=False).get("navix.env.grid")

    assert component.source == "navix"
    assert component.defaults["env_name"] == "empty_room"
    assert component.defaults["observation_mode"] == "tabular"
    assert component.config_schema["properties"]["observation_mode"]["enum"] == [
        "tabular",
        "one_hot",
        "state_features",
        "symbolic",
        "rgb",
    ]


def test_empty_room_cardinal_tabular_observation() -> None:
    env = NavixWrapper(
        env_name="empty_room",
        size=5,
        layout="fixed",
        observation_mode="tabular",
        action_set="cardinal",
    )

    _timestep, observation = env.reset(jax.random.PRNGKey(0))

    assert observation.shape == ()
    assert observation.dtype.name == "int32"
    assert int(env.env.action_space.n) == 4
    assert int(env.env.observation_space.n) == 9


def test_empty_room_cardinal_actions_do_not_rotate_player() -> None:
    env = NavixWrapper(
        env_name="empty_room",
        size=5,
        layout="fixed",
        observation_mode="tabular",
        action_set="cardinal",
    )
    timestep, _observation = env.reset(jax.random.PRNGKey(0))
    initial_direction = int(timestep.state.get_player().direction)

    timestep, *_ = env.step(timestep, 3)

    assert int(timestep.state.get_player().direction) == initial_direction


def test_empty_room_state_features_include_direction_only_for_default_actions() -> None:
    key = jax.random.PRNGKey(0)
    cardinal_env = NavixWrapper(
        env_name="empty_room",
        size=5,
        layout="fixed",
        observation_mode="state_features",
        action_set="cardinal",
    )
    default_env = NavixWrapper(
        env_name="empty_room",
        size=5,
        layout="fixed",
        observation_mode="state_features",
        action_set="default",
    )

    _timestep, cardinal_observation = cardinal_env.reset(key)
    _timestep, default_observation = default_env.reset(key)

    assert cardinal_observation.shape == (9,)
    assert default_observation.shape == (13,)


def test_doorkey_fixed_layout_feature_and_tabular_spaces() -> None:
    key = jax.random.PRNGKey(0)
    feature_env = NavixWrapper(
        env_name="doorkey",
        size=5,
        layout="layout1",
        observation_mode="state_features",
        action_set="default",
    )
    one_hot_env = NavixWrapper(
        env_name="doorkey",
        size=5,
        layout="layout1",
        observation_mode="one_hot",
        action_set="default",
    )

    _timestep, feature_observation = feature_env.reset(key)
    _timestep, one_hot_observation = one_hot_env.reset(key)

    assert feature_observation.shape == (25,)
    assert one_hot_observation.shape == (720,)
    assert int(one_hot_env.env.observation_space.n) == 2


def test_doorkey_random_layout_features_include_door_position() -> None:
    env = NavixWrapper(
        env_name="doorkey",
        size=5,
        layout="random",
        observation_mode="state_features",
        action_set="default",
    )

    _timestep, observation = env.reset(jax.random.PRNGKey(0))

    assert observation.shape == (34,)


@pytest.mark.parametrize("observation_mode", ["symbolic", "rgb"])
def test_original_navix_observation_modes(observation_mode: str) -> None:
    env = NavixWrapper(
        env_name="empty_room",
        size=5,
        layout="fixed",
        observation_mode=observation_mode,
        action_set="default",
    )

    _timestep, observation = env.reset(jax.random.PRNGKey(0))

    assert observation.shape[-1] == 3
    if observation_mode == "symbolic":
        assert observation.shape == (5, 5, 3)
    else:
        assert observation.ndim == 3


def test_cardinal_action_set_is_only_for_empty_room() -> None:
    with pytest.raises(ValueError, match="only supported for empty_room"):
        create_navix_environment(env_name="doorkey", action_set="cardinal")
