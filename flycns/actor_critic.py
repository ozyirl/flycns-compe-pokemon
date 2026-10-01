"""Standalone trainable actor-critic heads for Fly CNS descending activity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from flycns.action_decoder import ACTION_COUNT
from flycns.trainable_decoder import DESCENDING_ACTIVITY_SIZE, TrainableActionDecoder


WEIGHTS_FORMAT_VERSION = 1


class FlyCNSActorCritic(nn.Module):
    """Return action logits and a state value from 512 descending values."""

    def __init__(self) -> None:
        super().__init__()
        self.policy_head = TrainableActionDecoder()
        self.value_head = nn.Linear(DESCENDING_ACTIVITY_SIZE, 1)

    def forward(self, descending_activity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_logits = self.policy_head(descending_activity)
        state_value = self.value_head(descending_activity)
        return policy_logits, state_value

    def save_weights(self, path: str | Path) -> None:
        """Save both heads without any optimizer or simulator state."""

        torch.save(
            {
                "format_version": WEIGHTS_FORMAT_VERSION,
                "descending_activity_size": DESCENDING_ACTIVITY_SIZE,
                "action_count": ACTION_COUNT,
                "value_count": 1,
                "state_dict": self.state_dict(),
            },
            Path(path),
        )

    @classmethod
    def load_weights(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> "FlyCNSActorCritic":
        """Load weights saved by :meth:`save_weights` into a new wrapper."""

        payload: dict[str, Any] = torch.load(
            Path(path), map_location=map_location, weights_only=True
        )
        expected_metadata = {
            "format_version": WEIGHTS_FORMAT_VERSION,
            "descending_activity_size": DESCENDING_ACTIVITY_SIZE,
            "action_count": ACTION_COUNT,
            "value_count": 1,
        }
        actual_metadata = {key: payload.get(key) for key in expected_metadata}
        if actual_metadata != expected_metadata:
            raise ValueError(
                "Actor-critic weight metadata does not match this model: "
                f"expected {expected_metadata}, received {actual_metadata}"
            )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Actor-critic weights are missing a state_dict")

        model = cls()
        model.load_state_dict(state_dict)
        return model
