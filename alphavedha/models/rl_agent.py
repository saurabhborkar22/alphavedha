"""PPO-based reinforcement learning portfolio optimizer.

Learns optimal position sizing directly from market features,
accounting for transaction costs, slippage, and regime conditions.
Pure PyTorch implementation — no external RL library required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn as nn
from torch.distributions import Normal

logger = structlog.get_logger(__name__)


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic network for PPO."""

    def __init__(self, obs_size: int, action_size: int, hidden: int = 256) -> None:
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_size),
            nn.Tanh(),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))

        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        shared = self.shared(obs)
        mean = self.actor_mean(shared)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(shared)
        return dist, value

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(-1, 1), log_prob, value.squeeze(-1)

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value.squeeze(-1), entropy


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    hidden_size: int = 256


@dataclass
class RolloutBuffer:
    observations: list[np.ndarray]
    actions: list[np.ndarray]
    log_probs: list[float]
    rewards: list[float]
    values: list[float]
    dones: list[bool]

    @staticmethod
    def empty() -> RolloutBuffer:
        return RolloutBuffer([], [], [], [], [], [])

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns


class PPOAgent:
    """Proximal Policy Optimization agent for portfolio management."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        config: PPOConfig | None = None,
    ) -> None:
        self._config = config or PPOConfig()
        self._obs_size = obs_size
        self._action_size = action_size

        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_name)

        self._network = ActorCritic(
            obs_size, action_size, self._config.hidden_size,
        ).to(self._device)

        self._optimizer = torch.optim.Adam(
            self._network.parameters(), lr=self._config.lr,
        )

    def select_action(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Select action for a single observation.

        Returns: (action, log_prob, value)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self._device)
        with torch.no_grad():
            action, log_prob, value = self._network.get_action(obs_t)
        return (
            action.cpu().numpy().squeeze(0),
            float(log_prob.cpu()),
            float(value.cpu()),
        )

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run PPO update on collected rollout data."""
        advantages, returns = buffer.compute_returns(
            self._config.gamma, self._config.gae_lambda,
        )

        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 0:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        obs = torch.FloatTensor(np.array(buffer.observations)).to(self._device)
        actions = torch.FloatTensor(np.array(buffer.actions)).to(self._device)
        old_log_probs = torch.FloatTensor(buffer.log_probs).to(self._device)
        returns_t = torch.FloatTensor(returns).to(self._device)
        advantages_t = torch.FloatTensor(advantages).to(self._device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self._config.ppo_epochs):
            indices = np.random.permutation(len(buffer.observations))

            for start in range(0, len(indices), self._config.batch_size):
                batch_idx = indices[start : start + self._config.batch_size]
                batch_idx_t = torch.LongTensor(batch_idx).to(self._device)

                b_obs = obs[batch_idx_t]
                b_actions = actions[batch_idx_t]
                b_old_lp = old_log_probs[batch_idx_t]
                b_returns = returns_t[batch_idx_t]
                b_adv = advantages_t[batch_idx_t]

                new_log_probs, values, entropy = self._network.evaluate(b_obs, b_actions)

                ratio = (new_log_probs - b_old_lp).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self._config.clip_epsilon, 1 + self._config.clip_epsilon) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, b_returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self._config.value_coef * value_loss
                    + self._config.entropy_coef * entropy_loss
                )

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._network.parameters(), self._config.max_grad_norm)
                self._optimizer.step()

                total_policy_loss += float(policy_loss.detach())
                total_value_loss += float(value_loss.detach())
                total_entropy += float(entropy.mean().detach())
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def save(self, directory: Path) -> None:
        """Save agent weights and config."""
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self._network.state_dict(), directory / "ppo_weights.pt")

        import json
        meta = {
            "obs_size": self._obs_size,
            "action_size": self._action_size,
            "hidden_size": self._config.hidden_size,
            "lr": self._config.lr,
        }
        (directory / "ppo_config.json").write_text(json.dumps(meta, indent=2))
        logger.info("ppo_agent_saved", path=str(directory))

    @classmethod
    def load(cls, directory: Path) -> PPOAgent:
        """Load agent from saved weights."""
        import json

        meta = json.loads((directory / "ppo_config.json").read_text())
        config = PPOConfig(
            hidden_size=meta["hidden_size"],
            lr=meta["lr"],
        )
        agent = cls(meta["obs_size"], meta["action_size"], config)
        weights = torch.load(directory / "ppo_weights.pt", map_location=agent._device)
        agent._network.load_state_dict(weights)
        logger.info("ppo_agent_loaded", path=str(directory))
        return agent
