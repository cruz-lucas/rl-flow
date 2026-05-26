from __future__ import annotations

from rlflow.schemas.component import ComponentSpec
from rlflow_builtin.dqn.components import dqn_agent_components, intrinsic_reward_components
from rlflow_builtin.environments import environment_components
from rlflow_builtin.policies import policy_components
from rlflow_builtin.replay_buffers import replay_buffer_components
from rlflow_builtin.runners import runner_components
from rlflow_builtin.tabular.components import agent_components


def components() -> list[ComponentSpec]:
    return [
        *agent_components(),
        *dqn_agent_components(),
        *intrinsic_reward_components(),
        *environment_components(),
        *policy_components(),
        *replay_buffer_components(),
        *runner_components(),
    ]
