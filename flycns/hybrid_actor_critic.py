"""Hybrid actor-critic: fixed Fly CNS activity plus raw per-slot battle features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from encoding import (
    GLOBAL_FEATURES,
    MOVES_PER_POKEMON,
    OBSERVATION_DIM,
    TEAM_SIZE,
)
from flycns.action_decoder import ACTION_COUNT
from flycns.actor_critic import FlyCNSActorCritic
from flycns.trainable_decoder import DESCENDING_ACTIVITY_SIZE, TrainableActionDecoder


ARCHITECTURE = "fly-hybrid"
WEIGHTS_FORMAT_VERSION = 1

MOVE_FEATURE_SIZE = 8
MOVE_FEATURES_OFFSET = OBSERVATION_DIM - MOVES_PER_POKEMON * MOVE_FEATURE_SIZE
SWITCH_FEATURE_SIZE = 6
SWITCH_FEATURES_OFFSET = GLOBAL_FEATURES  # The trainer's own team comes first.
SWITCH_ACTIONS = TEAM_SIZE
# Actions 6-9 are plain moves; 10-25 repeat the same four slots with a gimmick
# (mega, z-move, dynamax, tera), so each slot score applies to all five groups.
MOVE_ACTION_GROUPS = (ACTION_COUNT - SWITCH_ACTIONS) // MOVES_PER_POKEMON
SCORER_INPUT_SIZE = MOVE_FEATURE_SIZE + 1


# One scorer unit per 20 base power: typical gaps between a Pokemon's moves.
BASE_POWER_UNIT = 20.0


def move_scorer_inputs(move_features: torch.Tensor) -> torch.Tensor:
    """Undo the encoder's compression so move gaps are O(1) for a small linear scorer.

    The extra final input is base power x accuracy x effectiveness x STAB, in the
    same units, which lets one weight express a damage-aware preference.
    """

    base_power = move_features[..., 0] * 250.0 / BASE_POWER_UNIT
    accuracy = move_features[..., 1]
    stab = 1.0 + 0.5 * move_features[..., 6]
    effectiveness = move_features[..., 7] * 4.0
    rescaled = torch.cat(
        [base_power.unsqueeze(-1), move_features[..., 1:7], effectiveness.unsqueeze(-1)], dim=-1
    )
    return torch.cat([rescaled, (base_power * accuracy * stab * effectiveness).unsqueeze(-1)], dim=-1)


class HybridPolicyHead(nn.Module):
    """Fly logits plus shared scorers that read each legal slot's own raw features."""

    def __init__(self) -> None:
        super().__init__()
        self.fly = TrainableActionDecoder()
        self.move_scorer = nn.Linear(SCORER_INPUT_SIZE, 1)
        self.switch_scorer = nn.Linear(SWITCH_FEATURE_SIZE, 1)
        # Zero-initialized raw scorers keep a warm-started model identical to its source.
        for scorer in (self.move_scorer, self.switch_scorer):
            nn.init.zeros_(scorer.weight)
            nn.init.zeros_(scorer.bias)

    def forward(self, normalized_activity: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        moves = observation[..., MOVE_FEATURES_OFFSET:].unflatten(-1, (MOVES_PER_POKEMON, MOVE_FEATURE_SIZE))
        move_scores = self.move_scorer(move_scorer_inputs(moves)).squeeze(-1)
        switches = observation[
            ..., SWITCH_FEATURES_OFFSET:SWITCH_FEATURES_OFFSET + SWITCH_ACTIONS * SWITCH_FEATURE_SIZE
        ].unflatten(-1, (SWITCH_ACTIONS, SWITCH_FEATURE_SIZE))
        switch_scores = self.switch_scorer(switches).squeeze(-1)
        slot_scores = torch.cat([switch_scores, move_scores.tile((MOVE_ACTION_GROUPS,))], dim=-1)
        return self.fly(normalized_activity) + slot_scores

    def raw_parameters(self) -> list[nn.Parameter]:
        return [*self.move_scorer.parameters(), *self.switch_scorer.parameters()]


class HybridValueHead(nn.Module):
    """Linear value over normalized Fly activity and the raw observation."""

    def __init__(self) -> None:
        super().__init__()
        self.fly = nn.Linear(DESCENDING_ACTIVITY_SIZE, 1)
        self.raw = nn.Linear(OBSERVATION_DIM, 1, bias=False)
        nn.init.zeros_(self.raw.weight)

    def forward(self, normalized_activity: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        return self.fly(normalized_activity) + self.raw(observation)

    def raw_parameters(self) -> list[nn.Parameter]:
        return list(self.raw.parameters())


class FlyCNSHybridActorCritic(nn.Module):
    """Same Fly pathway as :class:`FlyCNSActorCritic`, with raw slot features beside it."""

    def __init__(self) -> None:
        super().__init__()
        self.input_normalization = nn.LayerNorm(
            DESCENDING_ACTIVITY_SIZE, eps=1e-5, elementwise_affine=False
        )
        self.policy_head = HybridPolicyHead()
        self.value_head = HybridValueHead()

    def forward(
        self, descending_activity: torch.Tensor, observation: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observation is None or observation.shape[-1] != OBSERVATION_DIM:
            raise ValueError(f"Hybrid actor-critic requires the {OBSERVATION_DIM}-value observation")
        normalized_activity = self.input_normalization(descending_activity)
        return (
            self.policy_head(normalized_activity, observation),
            self.value_head(normalized_activity, observation),
        )

    def raw_parameters(self) -> list[nn.Parameter]:
        """Parameters that read raw observation features (new relative to Fly-only)."""

        return [*self.policy_head.raw_parameters(), *self.value_head.raw_parameters()]

    def fly_parameters(self) -> list[nn.Parameter]:
        raw = {id(parameter) for parameter in self.raw_parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in raw]

    @classmethod
    def from_fly(cls, source: FlyCNSActorCritic) -> "FlyCNSHybridActorCritic":
        """Copy Fly-only heads; zeroed raw pathways make the outputs initially identical."""

        model = cls()
        model.policy_head.fly.load_state_dict(source.policy_head.state_dict())
        model.value_head.fly.load_state_dict(source.value_head.state_dict())
        return model

    def save_weights(self, path: str | Path) -> None:
        torch.save(
            {
                "architecture": ARCHITECTURE,
                "format_version": WEIGHTS_FORMAT_VERSION,
                "descending_activity_size": DESCENDING_ACTIVITY_SIZE,
                "observation_size": OBSERVATION_DIM,
                "action_count": ACTION_COUNT,
                "state_dict": self.state_dict(),
            },
            Path(path),
        )

    @classmethod
    def load_weights(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> "FlyCNSHybridActorCritic":
        payload: dict[str, Any] = torch.load(Path(path), map_location=map_location, weights_only=True)
        expected_metadata = {
            "architecture": ARCHITECTURE,
            "format_version": WEIGHTS_FORMAT_VERSION,
            "descending_activity_size": DESCENDING_ACTIVITY_SIZE,
            "observation_size": OBSERVATION_DIM,
            "action_count": ACTION_COUNT,
        }
        actual_metadata = {key: payload.get(key) for key in expected_metadata}
        if actual_metadata != expected_metadata:
            raise ValueError(
                "Hybrid actor-critic weight metadata does not match this model: "
                f"expected {expected_metadata}, received {actual_metadata}"
            )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Hybrid actor-critic weights are missing a state_dict")
        model = cls()
        model.load_state_dict(state_dict)
        return model


def load_actor_critic(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> FlyCNSActorCritic | FlyCNSHybridActorCritic:
    """Load either checkpoint family, dispatching on the saved architecture tag."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if isinstance(payload, dict) and payload.get("architecture") == ARCHITECTURE:
        return FlyCNSHybridActorCritic.load_weights(path, map_location=map_location)
    return FlyCNSActorCritic.load_weights(path, map_location=map_location)
