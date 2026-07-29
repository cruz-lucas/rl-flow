"""Host-side uint8 replay buffer for the Atari agent.

Stores stacked frames as ``uint8`` on host RAM (not accelerator memory); only
sampled minibatches are moved to the device, where the CNN forward scales them to
``[0, 1]``. Coin-flip targets for CFN are sampled at insertion time (Bernoulli
±1) and stored alongside each transition, exactly as the jitted DQN pipeline does
in ``dqn/replay.py``.

Note: frames are stored already-stacked (envpool emits ``(C, H, W)`` stacks). A
single-frame ring with on-sample stacking would cut memory ~4x; that is a
worthwhile future optimisation but is deliberately avoided here to keep the
episode-boundary handling simple and obviously correct.
"""

from __future__ import annotations

import numpy as np


class AtariReplay:
    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        target_dim: int,
        seed: int = 0,
    ) -> None:
        self.capacity = int(capacity)
        self.observations = np.zeros((self.capacity, *observation_shape), dtype=np.uint8)
        self.next_observations = np.zeros((self.capacity, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros((self.capacity,), dtype=np.int32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.terminals = np.zeros((self.capacity,), dtype=np.bool_)
        self.coin_targets = np.zeros((self.capacity, int(target_dim)), dtype=np.int8)
        self._pos = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add_batch(
        self,
        observations: np.ndarray,
        next_observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        coin_targets: np.ndarray,
    ) -> None:
        n = observations.shape[0]
        idx = (self._pos + np.arange(n)) % self.capacity
        self.observations[idx] = observations
        self.next_observations[idx] = next_observations
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.terminals[idx] = terminals
        self.coin_targets[idx] = coin_targets
        self._pos = int((self._pos + n) % self.capacity)
        self._size = int(min(self._size + n, self.capacity))

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self._rng.integers(0, self._size, size=int(batch_size))
        return {
            "observations": self.observations[idx],
            "next_observations": self.next_observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "terminals": self.terminals[idx].astype(np.float32),
            "intrinsic_targets": self.coin_targets[idx].astype(np.float32),
        }

    def sample_coin_targets(self, num: int, target_dim: int) -> np.ndarray:
        """Sample fresh ±1 coin-flip targets for a batch of new transitions."""
        return self._rng.integers(0, 2, size=(int(num), int(target_dim)), dtype=np.int8) * 2 - 1
