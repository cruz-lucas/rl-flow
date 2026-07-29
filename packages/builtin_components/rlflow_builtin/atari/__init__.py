"""Atari (ALE) DQN + R-Max + CFN integration.

Runs real Atari via envpool through the non-JAX runtime-delegate hook
(``compile_target={"runtime": ...}``) rather than the jitted, JAX-native DQN
loop. The training loop lives on the host (envpool steps a vector of envs on the
CPU) while the CNN Q-update, the CFN coin-flip update, and the R-Max knownness
masking are ``jax.jit``-compiled on the accelerator.

``components`` is import-safe (no envpool / no CNN imports at module load) so the
registry stays importable everywhere; the heavy imports live in
:mod:`rlflow_builtin.atari.training` and are pulled in lazily when the delegate
actually runs.
"""

from __future__ import annotations

from rlflow_builtin.atari.components import atari_components

__all__ = ["atari_components"]
