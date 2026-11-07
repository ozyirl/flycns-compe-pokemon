"""Hybrid actor-critic: fixed Fly CNS activity plus raw per-slot battle features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from encoding import (
    ACTIVE_FEATURES,
    GLOBAL_FEATURES,
    MOVES_PER_POKEMON,
    OBSERVATION_DIM,
    TEAM_FEATURES,
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

    architecture = ARCHITECTURE
    format_version = WEIGHTS_FORMAT_VERSION

    def __init__(self) -> None:
        super().__init__()
        self.input_normalization = nn.LayerNorm(
            DESCENDING_ACTIVITY_SIZE, eps=1e-5, elementwise_affine=False
        )
        self.policy_head = self._make_policy_head()
        self.value_head = self._make_value_head()

    def _make_policy_head(self) -> HybridPolicyHead:
        return HybridPolicyHead()

    def _make_value_head(self) -> HybridValueHead:
        return HybridValueHead()

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
                "architecture": self.architecture,
                "format_version": self.format_version,
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
            "architecture": cls.architecture,
            "format_version": cls.format_version,
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


CONTEXT_ARCHITECTURE = "fly-hybrid-context"
CONTEXT_WEIGHTS_FORMAT_VERSION = 1
CONTEXT_HIDDEN_SIZE = 32
VALUE_HIDDEN_SIZE = 64
ACTIVE_FEATURES_OFFSET = SWITCH_FEATURES_OFFSET + TEAM_FEATURES
TEAM_SLOT_ACTIVE, TEAM_SLOT_HP = 1, 2
# Global flags, both active Pokemon (boosts, base stats, types), and both active HPs.
CONTEXT_SIZE = GLOBAL_FEATURES + ACTIVE_FEATURES + 2


def battle_context(observation: torch.Tensor) -> torch.Tensor:
    """Per-decision state that should modulate every slot's score."""

    teams = observation[..., SWITCH_FEATURES_OFFSET:ACTIVE_FEATURES_OFFSET].unflatten(
        -1, (2, TEAM_SIZE, SWITCH_FEATURE_SIZE)
    )
    active_hp = (teams[..., TEAM_SLOT_ACTIVE] * teams[..., TEAM_SLOT_HP]).sum(dim=-1)
    return torch.cat([
        observation[..., :GLOBAL_FEATURES],
        observation[..., ACTIVE_FEATURES_OFFSET:ACTIVE_FEATURES_OFFSET + ACTIVE_FEATURES],
        active_hp,
    ], dim=-1)


def _residual_mlp(input_size: int, hidden_size: int) -> nn.Sequential:
    """Small MLP whose zeroed output layer makes it a no-op until trained."""

    mlp = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
    nn.init.zeros_(mlp[-1].weight)
    nn.init.zeros_(mlp[-1].bias)
    return mlp


class ContextualPolicyHead(HybridPolicyHead):
    """Hybrid slot scores plus MLP corrections that see each slot beside the battle context."""

    def __init__(self) -> None:
        super().__init__()
        self.move_context = _residual_mlp(MOVE_FEATURE_SIZE + CONTEXT_SIZE, CONTEXT_HIDDEN_SIZE)
        self.switch_context = _residual_mlp(SWITCH_FEATURE_SIZE + CONTEXT_SIZE, CONTEXT_HIDDEN_SIZE)

    def forward(self, normalized_activity: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        context = battle_context(observation)
        moves = observation[..., MOVE_FEATURES_OFFSET:].unflatten(-1, (MOVES_PER_POKEMON, MOVE_FEATURE_SIZE))
        switches = observation[
            ..., SWITCH_FEATURES_OFFSET:SWITCH_FEATURES_OFFSET + SWITCH_ACTIONS * SWITCH_FEATURE_SIZE
        ].unflatten(-1, (SWITCH_ACTIONS, SWITCH_FEATURE_SIZE))

        def with_context(slots: torch.Tensor) -> torch.Tensor:
            return torch.cat([slots, context.unsqueeze(-2).expand(*slots.shape[:-1], CONTEXT_SIZE)], dim=-1)

        move_scores = self.move_context(with_context(moves)).squeeze(-1)
        switch_scores = self.switch_context(with_context(switches)).squeeze(-1)
        corrections = torch.cat([switch_scores, move_scores.tile((MOVE_ACTION_GROUPS,))], dim=-1)
        return super().forward(normalized_activity, observation) + corrections

    def raw_parameters(self) -> list[nn.Parameter]:
        return [*super().raw_parameters(), *self.move_context.parameters(), *self.switch_context.parameters()]


class ContextualValueHead(HybridValueHead):
    """Hybrid linear value plus a residual MLP over the raw observation."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_mlp = _residual_mlp(OBSERVATION_DIM, VALUE_HIDDEN_SIZE)

    def forward(self, normalized_activity: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        return super().forward(normalized_activity, observation) + self.raw_mlp(observation)

    def raw_parameters(self) -> list[nn.Parameter]:
        return [*super().raw_parameters(), *self.raw_mlp.parameters()]


class FlyCNSContextualActorCritic(FlyCNSHybridActorCritic):
    """Hybrid actor-critic whose slot scores and value also depend on battle context."""

    architecture = CONTEXT_ARCHITECTURE
    format_version = CONTEXT_WEIGHTS_FORMAT_VERSION

    def _make_policy_head(self) -> ContextualPolicyHead:
        return ContextualPolicyHead()

    def _make_value_head(self) -> ContextualValueHead:
        return ContextualValueHead()

    @classmethod
    def from_hybrid(cls, source: FlyCNSHybridActorCritic) -> "FlyCNSContextualActorCritic":
        """Copy every hybrid weight; zeroed MLP outputs keep the outputs initially identical."""

        if type(source) is not FlyCNSHybridActorCritic:
            raise TypeError("from_hybrid expects a plain hybrid actor-critic")
        model = cls()
        missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
        if unexpected or any(not key.split(".")[1].endswith(("context", "mlp")) for key in missing):
            raise ValueError(f"Unexpected hybrid weights: missing={missing}, unexpected={unexpected}")
        return model


_ARCHITECTURES: dict[str, type[FlyCNSHybridActorCritic]] = {
    ARCHITECTURE: FlyCNSHybridActorCritic,
    CONTEXT_ARCHITECTURE: FlyCNSContextualActorCritic,
}


def load_actor_critic(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> FlyCNSActorCritic | FlyCNSHybridActorCritic:
    """Load any checkpoint family, dispatching on the saved architecture tag."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    architecture = payload.get("architecture") if isinstance(payload, dict) else None
    if architecture in _ARCHITECTURES:
        return _ARCHITECTURES[architecture].load_weights(path, map_location=map_location)
    return FlyCNSActorCritic.load_weights(path, map_location=map_location)
