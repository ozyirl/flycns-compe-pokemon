"""Action-masked policy and replaceable battle feature networks."""

from __future__ import annotations

from typing import Any

import torch
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import CategoricalDistribution, Distribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs


class BaselineFeaturesExtractor(BaseFeaturesExtractor):
    """Plain MLP baseline; a fly network can later implement this same contract."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128) -> None:
        state_space = observation_space.spaces["observation"]
        if not isinstance(state_space, spaces.Box):
            raise TypeError("The 'observation' entry must be a Box")
        input_dim = int(state_space.shape[0])
        super().__init__(observation_space, features_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.network(observations["observation"])


class MaskedActorCriticPolicy(MultiInputActorCriticPolicy):
    """PPO policy that makes illegal Showdown actions impossible to sample."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("features_extractor_class", BaselineFeaturesExtractor)
        kwargs.setdefault("share_features_extractor", True)
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("MaskedActorCriticPolicy requires a Discrete action space")
        if not isinstance(self.action_dist, CategoricalDistribution):
            raise TypeError("MaskedActorCriticPolicy requires a categorical distribution")

    def _distribution(
        self, latent_pi: torch.Tensor, observation: PyTorchObs
    ) -> Distribution:
        if not isinstance(observation, dict):
            raise TypeError("Masked policy observations must be dictionaries")
        action_mask = observation["action_mask"].bool()
        if not torch.all(action_mask.any(dim=1)):
            raise RuntimeError("Received an observation with no legal action")
        logits = self.action_net(latent_pi)
        masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return self.action_dist.proba_distribution(action_logits=masked_logits)

    def _latents(self, observation: PyTorchObs) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(observation)
        if isinstance(features, tuple):
            policy_features, value_features = features
            return (
                self.mlp_extractor.forward_actor(policy_features),
                self.mlp_extractor.forward_critic(value_features),
            )
        return self.mlp_extractor(features)

    def forward(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_pi, latent_vf = self._latents(observation)
        values = self.value_net(latent_vf)
        distribution = self._distribution(latent_pi, observation)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probability = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_probability

    def evaluate_actions(
        self, observation: PyTorchObs, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        latent_pi, latent_vf = self._latents(observation)
        distribution = self._distribution(latent_pi, observation)
        return (
            self.value_net(latent_vf),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def get_distribution(self, observation: PyTorchObs) -> Distribution:
        latent_pi, _ = self._latents(observation)
        return self._distribution(latent_pi, observation)

