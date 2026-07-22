"""Pure network primitives for the DQN components (MLP + autoencoder).

Extracted from ``dqn.training`` so the neural-network math lives on its own. These
functions are dependency-free (jax only) and are re-exported by ``dqn.training``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


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
