"""Network primitives for the DQN components (MLP + autoencoder + optimizer).

Extracted from ``dqn.training`` so the neural-network building blocks live on
their own. Re-exported by ``dqn.training`` for backwards compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from rlflow_builtin.dqn.config import DqnAgentConfig


def _activation(x: jax.Array, name: str) -> jax.Array:
    if name == "tanh":
        return jnp.tanh(x)
    if name == "gelu":
        return jax.nn.gelu(x)
    if name == "elu":
        return jax.nn.elu(x)
    if name == "linear":
        return x
    return jax.nn.relu(x)


def _init_mlp(
    key: jax.Array,
    input_dim: int,
    hidden_units: tuple[int, ...],
    output_dim: int,
) -> tuple[dict[str, jax.Array], ...]:
    dims = (input_dim, *hidden_units, output_dim)
    keys = jax.random.split(key, len(dims) - 1)
    params = []
    for layer_key, in_dim, out_dim in zip(keys, dims[:-1], dims[1:], strict=True):
        scale = jnp.sqrt(2.0 / float(max(in_dim, 1)))
        params.append(
            {
                "w": jax.random.normal(layer_key, (in_dim, out_dim), dtype=jnp.float32) * scale,
                "b": jnp.zeros((out_dim,), dtype=jnp.float32),
            }
        )
    return tuple(params)


def _init_autoencoder(
    key: jax.Array,
    input_dim: int,
    hidden_units: tuple[int, ...],
    latent_dim: int,
) -> tuple[dict[str, jax.Array], ...]:
    decoder_units = tuple(reversed(hidden_units))
    dims = (input_dim, *hidden_units, latent_dim, *decoder_units, input_dim)
    keys = jax.random.split(key, len(dims) - 1)
    params = []
    for layer_key, in_dim, out_dim in zip(keys, dims[:-1], dims[1:], strict=True):
        scale = jnp.sqrt(2.0 / float(max(in_dim, 1)))
        params.append(
            {
                "w": jax.random.normal(layer_key, (in_dim, out_dim), dtype=jnp.float32) * scale,
                "b": jnp.zeros((out_dim,), dtype=jnp.float32),
            }
        )
    return tuple(params)


def _apply_mlp(
    params: tuple[dict[str, jax.Array], ...],
    observations: jax.Array,
    activation: str = "relu",
) -> jax.Array:
    x = jnp.asarray(observations, dtype=jnp.float32)
    for layer in params[:-1]:
        x = _activation(x @ layer["w"] + layer["b"], activation)
    output = params[-1]
    return x @ output["w"] + output["b"]


def _apply_autoencoder_encoder(
    params: tuple[dict[str, jax.Array], ...],
    observations: jax.Array,
    hidden_units: tuple[int, ...],
    activation: str = "relu",
) -> jax.Array:
    x = jnp.asarray(observations, dtype=jnp.float32)
    latent_layer_index = len(hidden_units)
    for index, layer in enumerate(params):
        x = x @ layer["w"] + layer["b"]
        if index == latent_layer_index:
            return x
        x = _activation(x, activation)
    return x


def _optimizer(
    agent: DqnAgentConfig,
    learning_rate: float,
    optimizer_name: str,
) -> optax.GradientTransformation:
    if optimizer_name == "sgd":
        base = optax.sgd(learning_rate, momentum=agent.optimizer_momentum)
    elif optimizer_name == "rmsprop":
        base = optax.rmsprop(
            learning_rate,
            decay=agent.optimizer_decay,
            eps=agent.optimizer_epsilon,
            momentum=agent.optimizer_momentum,
            centered=agent.optimizer_centered,
        )
    else:
        base = optax.adamw(
            learning_rate,
            b1=agent.optimizer_beta1,
            b2=agent.optimizer_beta2,
            eps=agent.optimizer_epsilon,
            weight_decay=agent.optimizer_weight_decay,
        )
    if agent.max_grad_norm > 0.0:
        return optax.chain(optax.clip_by_global_norm(agent.max_grad_norm), base)
    return base
