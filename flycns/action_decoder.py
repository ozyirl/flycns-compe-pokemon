"""Read-only fixed projection from descending spikes to Showdown action scores."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from poke_env.environment import SinglesEnv


GENERATION = 9
ACTION_COUNT = SinglesEnv.get_action_space_size(GENERATION)


def default_action_labels() -> tuple[str, ...]:
    """Describe poke-env's existing SinglesEnv action indices without renumbering."""

    labels = [f"switch: slot {slot}" for slot in range(1, 7)]
    labels.extend(f"move: slot {slot}" for slot in range(1, 5))
    for gimmick in ("mega", "z-move", "dynamax", "terastallize"):
        labels.extend(
            f"move: slot {slot} + {gimmick}" for slot in range(1, 5)
        )
    if len(labels) != ACTION_COUNT:
        raise RuntimeError(
            f"Expected {ACTION_COUNT} poke-env actions, built {len(labels)} labels"
        )
    return tuple(labels)


@dataclass(frozen=True)
class HypotheticalActionScore:
    action: int
    label: str
    score: float


class DescendingActionDecoder:
    """Score existing poke-env actions with a deterministic, untrained projection."""

    def __init__(self, descending_root_ids: Sequence[int]) -> None:
        root_ids = np.asarray(descending_root_ids, dtype=np.int64)
        if root_ids.ndim != 1 or root_ids.size == 0:
            raise ValueError("descending_root_ids must be a non-empty one-dimensional list")
        if np.unique(root_ids).size != root_ids.size:
            raise ValueError("descending_root_ids must be unique")
        self.descending_root_ids = root_ids
        self.action_count = ACTION_COUNT
        self.projection = self._build_projection(root_ids)

    @staticmethod
    def _build_projection(root_ids: np.ndarray) -> np.ndarray:
        """Build stable signed weights from body IDs and existing action indices."""

        scale = 1.0 / math.sqrt(root_ids.size)
        projection = np.empty((root_ids.size, ACTION_COUNT), dtype=np.float64)
        for row, root_id in enumerate(root_ids):
            for action in range(ACTION_COUNT):
                key = f"fly-action-v1:{int(root_id)}:{action}".encode()
                value = int.from_bytes(
                    hashlib.blake2b(key, digest_size=8).digest(), "little"
                )
                unit = value / float((1 << 64) - 1)
                projection[row, action] = (2.0 * unit - 1.0) * scale
        return projection

    def score(self, descending_spike_counts: Sequence[int] | np.ndarray) -> np.ndarray:
        """Return all action scores; legality is deliberately not applied here."""

        spikes = np.asarray(descending_spike_counts, dtype=np.float64)
        expected_shape = (self.descending_root_ids.size,)
        if spikes.shape != expected_shape:
            raise ValueError(
                f"Expected descending spike shape {expected_shape}, received {spikes.shape}"
            )
        if not np.isfinite(spikes).all() or np.any(spikes < 0):
            raise ValueError("Descending spike counts must be finite and non-negative")
        return np.log1p(spikes) @ self.projection

    def rank_legal(
        self,
        scores: Sequence[float] | np.ndarray,
        action_mask: Sequence[int] | np.ndarray,
        labels: Sequence[str] | None = None,
    ) -> list[HypotheticalActionScore]:
        """Filter with the existing mask and rank legal actions without executing one."""

        score_array = np.asarray(scores, dtype=np.float64)
        mask = np.asarray(action_mask).astype(bool, copy=False)
        expected_shape = (self.action_count,)
        if score_array.shape != expected_shape:
            raise ValueError(
                f"Expected action score shape {expected_shape}, received {score_array.shape}"
            )
        if mask.shape != expected_shape:
            raise ValueError(
                f"Expected action mask shape {expected_shape}, received {mask.shape}"
            )
        action_labels = tuple(labels) if labels is not None else default_action_labels()
        if len(action_labels) != self.action_count:
            raise ValueError(
                f"Expected {self.action_count} action labels, received {len(action_labels)}"
            )

        legal_indices = np.flatnonzero(mask)
        ranked = sorted(
            legal_indices.tolist(), key=lambda action: (-score_array[action], action)
        )
        return [
            HypotheticalActionScore(
                action=action,
                label=str(action_labels[action]),
                score=float(score_array[action]),
            )
            for action in ranked
        ]

    def hypothetical_choice(
        self,
        scores: Sequence[float] | np.ndarray,
        action_mask: Sequence[int] | np.ndarray,
        labels: Sequence[str] | None = None,
    ) -> HypotheticalActionScore:
        ranked = self.rank_legal(scores, action_mask, labels)
        if not ranked:
            raise ValueError("The supplied action_mask contains no legal actions")
        return ranked[0]
