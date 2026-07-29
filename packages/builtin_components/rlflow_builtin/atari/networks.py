"""Raw-JAX Nature-CNN primitives for the Atari agent.

Deliberately mirrors the hand-rolled param-dict + optax conventions in
``rlflow_builtin.dqn.networks`` (see ``_init_mlp`` / ``_apply_mlp``) rather than
pulling in flax — the conv stack is the only genuinely new network code, and the
dense head is reused verbatim. Observations are channels-first uint8 batches
``(N, C, H, W)`` (envpool's stacked-frame layout) and are scaled to ``[0, 1]``
inside the forward pass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rlflow_builtin.dqn.networks import _activation, _apply_mlp, _init_mlp

# (out_channels, kernel_size, stride) for the three Nature-DQN conv layers.
_NATURE_CONV: tuple[tuple[int, int, int], ...] = ((32, 8, 4), (64, 4, 2), (64, 3, 1))


def _conv_output_hw(height: int, width: int) -> tuple[int, int]:
    for _out_ch, kernel, stride in _NATURE_CONV:
        height = (height - kernel) // stride + 1
        width = (width - kernel) // stride + 1
    return height, width


def _conv_feature_dim(in_channels: int, height: int, width: int) -> int:
    del in_channels
    out_height, out_width = _conv_output_hw(height, width)
    return int(_NATURE_CONV[-1][0] * out_height * out_width)


def _apply_conv(layer: dict[str, jax.Array], x: jax.Array, stride: int) -> jax.Array:
    y = jax.lax.conv_general_dilated(
        x,
        layer["w"],
        window_strides=(stride, stride),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return y + layer["b"][None, :, None, None]


def nature_cnn_init(
    key: jax.Array,
    in_channels: int,
    height: int,
    width: int,
    output_dim: int,
    head_hidden: tuple[int, ...] = (512,),
) -> dict:
    """Initialise a Nature-DQN CNN: 3 conv layers + an MLP head.

    ``head_hidden=()`` yields a single linear projection head, which is how the
    (fixed, random) CFN encoder maps pixels to an embedding.
    """
    conv_keys = jax.random.split(key, len(_NATURE_CONV) + 1)
    conv_layers = []
    in_ch = in_channels
    for index, (out_ch, kernel, _stride) in enumerate(_NATURE_CONV):
        scale = jnp.sqrt(2.0 / float(max(in_ch * kernel * kernel, 1)))
        conv_layers.append(
            {
                "w": jax.random.normal(
                    conv_keys[index], (out_ch, in_ch, kernel, kernel), dtype=jnp.float32
                )
                * scale,
                "b": jnp.zeros((out_ch,), dtype=jnp.float32),
            }
        )
        in_ch = out_ch
    feature_dim = _conv_feature_dim(in_channels, height, width)
    head = _init_mlp(conv_keys[-1], feature_dim, head_hidden, output_dim)
    return {"conv": tuple(conv_layers), "head": head}


def nature_cnn_apply(params: dict, observations: jax.Array, activation: str = "relu") -> jax.Array:
    x = jnp.asarray(observations, dtype=jnp.float32) / 255.0
    for layer, (_out_ch, _kernel, stride) in zip(params["conv"], _NATURE_CONV, strict=True):
        x = _activation(_apply_conv(layer, x, stride), activation)
    x = x.reshape(x.shape[0], -1)
    return _apply_mlp(params["head"], x, activation)


def cfn_encoder_init(
    key: jax.Array,
    in_channels: int,
    height: int,
    width: int,
    embed_dim: int,
) -> dict:
    """Fixed random CNN encoder mapping pixels to a CFN feature embedding.

    Not trained: it provides a stable distinguishing representation on top of
    which the reused CFN coin-flip predictor/prior MLPs operate. Random CNN
    features are a well-established, cheap encoder for exploration bonuses.
    """
    return nature_cnn_init(key, in_channels, height, width, embed_dim, head_hidden=())


def cfn_encoder_apply(params: dict, observations: jax.Array) -> jax.Array:
    return jax.nn.relu(nature_cnn_apply(params, observations, "relu"))
