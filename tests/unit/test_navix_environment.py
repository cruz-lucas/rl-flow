import jax
import numpy as np
import pytest

from rlflow.registry.builtin import create_default_registry
from rlflow_builtin.dqn.training import _coerce_navix_settings
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
    assert component.config_schema["properties"]["env_name"]["enum"] == [
        "empty_room",
        "doorkey",
        "four_rooms",
    ]
    assert component.config_schema["properties"]["size"]["enum"] == [5, 6, 8, 16, 19]
    assert component.config_schema["properties"]["symbolic_distractor"]["enum"] == [
        "none",
        "corner_wall_color",
        "shared_wall_color",
        "independent_wall_color",
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


def test_four_rooms_fixed_layout_has_fixed_goal_walls_and_cardinal_actions() -> None:
    env = NavixWrapper(
        env_name="four_rooms",
        size=19,
        layout="fixed",
        observation_mode="symbolic",
        action_set="cardinal",
    )

    timestep, observation = env.reset(jax.random.PRNGKey(0))
    entities = np.asarray(observation[..., 0])

    assert observation.shape == (19, 19, 3)
    assert int(env.env.action_space.n) == 4
    assert timestep.state.get_player().position.tolist() == [1, 1]
    assert timestep.state.get_goals().position.tolist() == [[17, 17]]
    for row, col in [(4, 9), (14, 9), (9, 4), (9, 14)]:
        assert int(entities[row, col]) != 2
    for row, col in [(1, 9), (9, 1), (9, 9), (17, 9), (9, 17)]:
        assert int(entities[row, col]) == 2


def test_four_rooms_random_layout_resets() -> None:
    env = NavixWrapper(
        env_name="four_rooms",
        size=19,
        layout="random",
        observation_mode="symbolic",
        action_set="cardinal",
    )

    _timestep, observation = env.reset(jax.random.PRNGKey(0))

    assert observation.shape == (19, 19, 3)
    assert int(env.env.action_space.n) == 4


def test_four_rooms_cardinal_actions_do_not_rotate_player() -> None:
    env = NavixWrapper(
        env_name="four_rooms",
        size=19,
        layout="fixed",
        observation_mode="tabular",
        action_set="cardinal",
    )
    timestep, _observation = env.reset(jax.random.PRNGKey(0))
    initial_direction = int(timestep.state.get_player().direction)

    timestep, *_ = env.step(timestep, 3)

    assert int(timestep.state.get_player().direction) == initial_direction
    assert timestep.state.get_player().position.tolist() == [1, 2]


def test_four_rooms_cardinal_tabular_observation() -> None:
    env = NavixWrapper(
        env_name="four_rooms",
        size=19,
        layout="fixed",
        observation_mode="tabular",
        action_set="cardinal",
    )

    _timestep, observation = env.reset(jax.random.PRNGKey(0))

    assert observation.shape == ()
    assert observation.dtype.name == "int32"
    assert int(observation) == 0
    assert int(env.env.action_space.n) == 4
    assert int(env.env.observation_space.n) == 289


def test_four_rooms_state_features_include_direction_only_for_default_actions() -> None:
    key = jax.random.PRNGKey(0)
    cardinal_env = NavixWrapper(
        env_name="four_rooms",
        size=19,
        layout="fixed",
        observation_mode="state_features",
        action_set="cardinal",
    )
    default_env = NavixWrapper(
        env_name="four_rooms",
        size=19,
        layout="fixed",
        observation_mode="state_features",
        action_set="default",
    )

    _timestep, cardinal_observation = cardinal_env.reset(key)
    _timestep, default_observation = default_env.reset(key)

    assert cardinal_observation.shape == (289,)
    assert default_observation.shape == (293,)


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


def test_symbolic_corner_wall_distractor_changes_over_time() -> None:
    env = create_navix_environment(
        env_name="empty_room",
        size=5,
        layout="fixed",
        observation_mode="symbolic",
        action_set="cardinal",
        symbolic_distractor="corner_wall_color",
    )
    timestep = env.reset(jax.random.PRNGKey(0))
    values = [int(timestep.observation[0, -1, 1])]
    for _ in range(4):
        timestep = env.step(timestep, 3)
        values.append(int(timestep.observation[0, -1, 1]))

    assert len(set(values)) > 1


def test_four_rooms_symbolic_corner_wall_distractor_changes_over_time() -> None:
    env = create_navix_environment(
        env_name="four_rooms",
        size=19,
        layout="fixed",
        observation_mode="symbolic",
        action_set="cardinal",
        symbolic_distractor="corner_wall_color",
    )
    timestep = env.reset(jax.random.PRNGKey(0))
    values = [int(timestep.observation[0, -1, 1])]
    for _ in range(4):
        timestep = env.step(timestep, 3)
        values.append(int(timestep.observation[0, -1, 1]))

    assert len(set(values)) > 1


def test_shared_wall_distractor_uses_one_value_for_all_walls() -> None:
    env = create_navix_environment(
        env_name="empty_room",
        size=5,
        layout="fixed",
        observation_mode="symbolic",
        action_set="cardinal",
        symbolic_distractor="shared_wall_color",
    )

    timestep = env.reset(jax.random.PRNGKey(0))
    observation = timestep.observation
    wall_values = observation[observation[..., 0] == 2, 1]

    assert len(set(map(int, wall_values))) == 1


def test_four_rooms_shared_wall_distractor_uses_one_value_for_all_walls() -> None:
    env = create_navix_environment(
        env_name="four_rooms",
        size=19,
        layout="fixed",
        observation_mode="symbolic",
        action_set="cardinal",
        symbolic_distractor="shared_wall_color",
    )

    timestep = env.reset(jax.random.PRNGKey(0))
    observation = timestep.observation
    wall_values = observation[observation[..., 0] == 2, 1]

    assert len(set(map(int, wall_values))) == 1


def test_dqn_navix_settings_preserve_symbolic_distractor() -> None:
    settings = _coerce_navix_settings(
        {
            "env_name": "empty_room",
            "size": 16,
            "layout": "fixed",
            "observation_mode": "symbolic",
            "action_set": "cardinal",
            "max_steps": 2048,
            "symbolic_distractor": "independent_wall_color",
        }
    )

    assert settings["symbolic_distractor"] == "independent_wall_color"


def test_symbolic_distractors_require_symbolic_supported_room() -> None:
    with pytest.raises(ValueError, match="require observation_mode='symbolic'"):
        create_navix_environment(
            env_name="empty_room",
            observation_mode="tabular",
            symbolic_distractor="corner_wall_color",
        )
    with pytest.raises(ValueError, match="require observation_mode='symbolic'"):
        create_navix_environment(
            env_name="four_rooms",
            size=19,
            observation_mode="tabular",
            symbolic_distractor="corner_wall_color",
        )
    with pytest.raises(ValueError, match="only supported for empty_room"):
        create_navix_environment(
            env_name="doorkey",
            observation_mode="symbolic",
            symbolic_distractor="corner_wall_color",
        )


def test_cardinal_action_set_is_only_for_empty_room() -> None:
    with pytest.raises(ValueError, match="only supported for empty_room and four_rooms"):
        create_navix_environment(env_name="doorkey", action_set="cardinal")


def test_four_rooms_rejects_noncanonical_sizes_and_named_layouts() -> None:
    with pytest.raises(ValueError, match="19x19 only"):
        create_navix_environment(env_name="four_rooms", size=16)
    with pytest.raises(ValueError, match="fixed or random"):
        create_navix_environment(env_name="four_rooms", size=19, layout="layout1")
