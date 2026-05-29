# Builtin Components

Builtin components use the `builtin.*` ID prefix and `source: builtin`.
Provider-backed components can use their own source group; Navix uses
`source: navix` and appears as a separate collapsible group in the UI.

Current builtin components:

- `builtin.agent.q_learning_tabular`
- `builtin.agent.sarsa_tabular`
- `builtin.agent.dqn_jax`
- `builtin.agent.dqn_rmax_jax`
- `builtin.intrinsic.rnd`
- `builtin.intrinsic.cfn`
- `builtin.intrinsic.count`
- `builtin.intrinsic.simhash`
- `builtin.policy.epsilon_greedy`
- `builtin.policy.ucb`
- `builtin.policy.softmax`
- `builtin.env.gridworld`
- `builtin.env.riverswim`
- `builtin.env.sixarms`
- `builtin.replay.tabular_uniform`
- `builtin.replay.uniform`
- `builtin.runner.tabular_jax`
- `navix.env.grid`

The builtin JAX runner compiles to `rlflow_builtin.runners.tabular_jax`. With
`builtin.agent.q_learning_tabular` or `builtin.agent.sarsa_tabular`, it runs a
JIT-compiled JAX scan over fixed-length episodes, writes `q_table.npy`,
`action_counts.npy`, `metrics.json`, JSONL train/eval histories, and optionally
a final `.npz` checkpoint.

Examples:

- `configs/workflows/tabular_q_learning_riverswim.yaml`
- `configs/workflows/tabular_q_learning_sixarms.yaml`
- `configs/workflows/tabular_sarsa_gridworld.yaml`
- `configs/workflows/tabular_q_learning_navix_empty_room.yaml`
- `configs/workflows/tabular_collect_riverswim_dataset.yaml`
- `configs/workflows/tabular_offline_q_learning_riverswim.yaml`
- `configs/workflows/navix_dqn_empty_room.yaml`

Navix `navix.env.grid` supports `empty_room` and `doorkey`. Observation modes
are `tabular`, `one_hot`, `state_features`, `symbolic`, and `rgb`. The
`cardinal` action set is available for `empty_room`; DoorKey uses the default
Navix/Minigrid action set.

For value learning on Navix vector, symbolic, RGB, or scalar tabular
observations, connect `builtin.agent.dqn_jax` to `builtin.runner.tabular_jax`
with a `builtin.replay.uniform` replay buffer. Optional exploration bonuses are
separate intrinsic reward modules: connect `builtin.intrinsic.rnd`,
`builtin.intrinsic.cfn`, `builtin.intrinsic.count`, or
`builtin.intrinsic.simhash` to the runner's
`intrinsic_reward` port. The DQN path uses JAX scans over fixed-length episodes
and replay updates, with scalar tabular observations converted to one-hot
vectors before entering the network. Set `normalize_observations: true` to
scale integer vector observations, such as Navix symbolic or RGB observations,
before they enter the network. Saved DQN replay datasets keep the original
environment observations, not the network-encoded vectors. CFN replay stores
the coin-flip targets needed by the intrinsic reward update.

`builtin.agent.dqn_rmax_jax` uses the same network and replay path as DQN, but
uses the connected intrinsic module as an unknown detector instead of
epsilon-greedy exploration. At action selection time, any action whose
normalized intrinsic bonus is above `rmax_bonus_threshold` is assigned
`rmax_decision_v_max`. During replay updates, unknown current state-action pairs
are masked out of the Q loss, and target bootstrapping uses `rmax_update_v_max`
whenever the next observation has any unknown action. Count bonuses are maintained in a
JIT-friendly hashed table; use `count_action_conditioning: input` for
state-action counts or `none` for state counts.

`builtin.intrinsic.simhash` implements the SimHash count bonus from
count-based exploration for deep RL. Set `simhash_mode: static` to hash the
conditioned observation with a fixed random projection, or `simhash_mode:
learned` to train an autoencoder and hash its latent representation. Use
`simhash_hidden_units` to set the learned autoencoder shape, for example
`"256,256"` or `[256, 256]`.

DQN train and eval histories include `discounted_return`. For sweep plots, run
`rlflow sweep export-learning-curves <sweep_manifest.yaml>` to write a CSV and
SVG with seed-averaged discounted-return curves and bootstrapped 95% confidence
bands.

The tabular replay buffer can also act as a scalar-transition dataset bridge:

- Set `save_dataset_path` to write collected replay transitions as a compressed
  `.npz` file under the run directory.
- Set `load_dataset_path` to seed the replay buffer from a saved dataset. Relative
  paths are resolved from the project root when they exist, otherwise from the
  current run directory.
- Set `offline_only: true` and `offline_updates` to train from the loaded replay
  dataset without environment interaction.
