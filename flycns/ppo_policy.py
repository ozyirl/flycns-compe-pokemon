"""Read-only one-decision policy wrapper over the fixed Fly CNS simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from encoding import OBSERVATION_DIM
from flycns.action_decoder import ACTION_COUNT
from flycns.actor_critic import FlyCNSActorCritic
from flycns.graph import FlyGraph
from flycns.stimulation import BattleSensoryAdapter, FlySpikeSimulator


DEFAULT_GRAPH_PATH = Path(__file__).resolve().parents[1] / "artifacts/fly_connectome.npz"


@dataclass(frozen=True)
class PolicyDecision:
    selected_action_index: int
    action_log_probability: float
    state_value: float
    policy_logits: tuple[float, ...]


@dataclass(frozen=True)
class BatchPolicyEvaluation:
    action_log_probabilities: np.ndarray
    entropies: np.ndarray
    state_values: np.ndarray


class FlyCNSPPOPolicy:
    """Evaluate one observation without updating weights or executing an action.

    ``policy_logits`` in the result are masked: illegal indices have ``-inf``.
    Deterministic evaluation chooses the highest legal logit, with the lowest
    action index breaking ties. Sampling uses the same masked categorical law.
    """

    def __init__(
        self,
        *,
        graph_path: str | Path = DEFAULT_GRAPH_PATH,
        actor_critic: FlyCNSActorCritic | None = None,
    ) -> None:
        graph = FlyGraph.load(graph_path)
        self.adapter = BattleSensoryAdapter(graph)
        self.simulator = FlySpikeSimulator(graph)
        if actor_critic is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(7)
                actor_critic = FlyCNSActorCritic()
        self.actor_critic = actor_critic

    @staticmethod
    def _validated_masks(
        action_masks: Sequence[int] | np.ndarray,
        expected_shape: tuple[int, ...],
    ) -> np.ndarray:
        masks = np.asarray(action_masks)
        if masks.shape != expected_shape:
            raise ValueError(f"Expected action mask shape {expected_shape}, received {masks.shape}")
        if not np.isin(masks, (0, 1)).all():
            raise ValueError("action_mask values must be 0 or 1")
        if not np.all(np.any(masks, axis=-1)):
            raise ValueError("Each action_mask must contain at least one legal action")
        return masks.astype(bool, copy=False)

    def _descending_activity(self, observation: np.ndarray) -> np.ndarray:
        stimulation = self.adapter.adapt(observation)
        report = self.simulator.run(stimulation)
        return report.spike_counts[report.descending_indices].astype(np.float32)

    def predict(
        self,
        observation: Sequence[float] | np.ndarray,
        action_mask: Sequence[int] | np.ndarray,
        *,
        deterministic: bool = True,
    ) -> PolicyDecision:
        mask = self._validated_masks(action_mask, (ACTION_COUNT,))
        descending = self._descending_activity(np.asarray(observation, dtype=np.float32))

        parameter = next(self.actor_critic.parameters())
        activity = torch.from_numpy(descending).to(device=parameter.device, dtype=parameter.dtype)
        legal = torch.as_tensor(mask, device=parameter.device, dtype=torch.bool)
        with torch.inference_mode():
            raw_logits, value = self.actor_critic(activity)
            masked_logits = raw_logits.masked_fill(~legal, float("-inf"))
            distribution = Categorical(logits=masked_logits)
            action = torch.argmax(masked_logits) if deterministic else distribution.sample()
            log_probability = distribution.log_prob(action)

        return PolicyDecision(
            selected_action_index=int(action.item()),
            action_log_probability=float(log_probability.item()),
            state_value=float(value.item()),
            policy_logits=tuple(float(logit) for logit in masked_logits.cpu().tolist()),
        )

    def sample_action(
        self,
        observation: Sequence[float] | np.ndarray,
        action_mask: Sequence[int] | np.ndarray,
    ) -> PolicyDecision:
        """Sample one legal action from the categorical policy without executing it."""

        return self.predict(observation, action_mask, deterministic=False)

    def evaluate_batch(
        self,
        observations: Sequence[Sequence[float]] | np.ndarray,
        action_masks: Sequence[Sequence[int]] | np.ndarray,
        selected_action_indices: Sequence[int] | np.ndarray,
    ) -> BatchPolicyEvaluation:
        """Evaluate supplied legal actions without sampling or updating weights."""

        observation_array = np.asarray(observations, dtype=np.float32)
        if observation_array.ndim != 2 or observation_array.shape[1] != OBSERVATION_DIM:
            raise ValueError(
                f"Expected observation batch shape (batch, {OBSERVATION_DIM}), "
                f"received {observation_array.shape}"
            )
        batch_size = observation_array.shape[0]
        if batch_size == 0:
            raise ValueError("Observation batch must not be empty")
        masks = self._validated_masks(action_masks, (batch_size, ACTION_COUNT))
        actions = np.asarray(selected_action_indices)
        if actions.shape != (batch_size,) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError(f"Expected integer action indices with shape ({batch_size},)")
        if np.any((actions < 0) | (actions >= ACTION_COUNT)):
            raise ValueError(f"Selected action indices must be in [0, {ACTION_COUNT})")
        if not np.all(masks[np.arange(batch_size), actions]):
            raise ValueError("Selected action indices must be legal under their action_masks")

        descending = np.stack(
            [self._descending_activity(observation) for observation in observation_array]
        )
        parameter = next(self.actor_critic.parameters())
        activity = torch.from_numpy(descending).to(device=parameter.device, dtype=parameter.dtype)
        legal = torch.as_tensor(masks, device=parameter.device, dtype=torch.bool)
        selected = torch.as_tensor(actions, device=parameter.device, dtype=torch.long)
        with torch.inference_mode():
            raw_logits, values = self.actor_critic(activity)
            masked_logits = raw_logits.masked_fill(~legal, float("-inf"))
            distribution = Categorical(logits=masked_logits)
            log_probabilities = distribution.log_prob(selected)
            entropies = distribution.entropy()

        return BatchPolicyEvaluation(
            action_log_probabilities=log_probabilities.cpu().numpy().copy(),
            entropies=entropies.cpu().numpy().copy(),
            state_values=values.squeeze(-1).cpu().numpy().copy(),
        )
