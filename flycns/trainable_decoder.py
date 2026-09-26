"""Isolated trainable decoder for Fly CNS descending-neuron activity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from flycns.action_decoder import ACTION_COUNT


DESCENDING_ACTIVITY_SIZE = 512
WEIGHTS_FORMAT_VERSION = 1


class TrainableActionDecoder(nn.Module):
    """Map 512 descending-neuron values to the existing 26 action logits."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(DESCENDING_ACTIVITY_SIZE, ACTION_COUNT)

    def forward(self, descending_activity: torch.Tensor) -> torch.Tensor:
        if descending_activity.ndim == 0 or descending_activity.shape[-1] != DESCENDING_ACTIVITY_SIZE:
            raise ValueError(
                "Expected descending activity with final dimension "
                f"{DESCENDING_ACTIVITY_SIZE}, received {tuple(descending_activity.shape)}"
            )
        return self.linear(descending_activity)

    def save_weights(self, path: str | Path) -> None:
        """Save decoder parameters and shape metadata without optimizer state."""

        torch.save(
            {
                "format_version": WEIGHTS_FORMAT_VERSION,
                "descending_activity_size": DESCENDING_ACTIVITY_SIZE,
                "action_count": ACTION_COUNT,
                "state_dict": self.state_dict(),
            },
            Path(path),
        )

    @classmethod
    def load_weights(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> "TrainableActionDecoder":
        """Construct a decoder from weights written by :meth:`save_weights`."""

        payload: dict[str, Any] = torch.load(
            Path(path), map_location=map_location, weights_only=True
        )
        expected_metadata = {
            "format_version": WEIGHTS_FORMAT_VERSION,
            "descending_activity_size": DESCENDING_ACTIVITY_SIZE,
            "action_count": ACTION_COUNT,
        }
        actual_metadata = {
            key: payload.get(key) for key in expected_metadata
        }
        if actual_metadata != expected_metadata:
            raise ValueError(
                "Decoder weight metadata does not match this model: "
                f"expected {expected_metadata}, received {actual_metadata}"
            )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Decoder weights are missing a state_dict")

        decoder = cls()
        decoder.load_state_dict(state_dict)
        return decoder
