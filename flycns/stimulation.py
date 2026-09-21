"""Debug-only battle-observation adapter and fixed Fly CNS spike simulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from encoding import OBSERVATION_DIM
from flycns.graph import FlyGraph


class BattleSensoryAdapter:
    """Map the PPO battle vector to non-negative sensory-neuron stimulation.

    Each battle feature is repeated across the selected sensory population. Repeated
    copies alternate polarity, so both positive and negative feature values can drive
    sensory cells. The mapping is deterministic and has no learned parameters.
    """

    def __init__(self, graph: FlyGraph, observation_dim: int = OBSERVATION_DIM) -> None:
        self.observation_dim = observation_dim
        self.sensory_indices = graph.population_indices["sensory"]
        if self.sensory_indices.size == 0:
            raise ValueError("Fly graph has no sensory neurons")

        sensory_slots = np.arange(self.sensory_indices.size, dtype=np.int64)
        self.feature_indices = sensory_slots % observation_dim
        copies = sensory_slots // observation_dim
        self.polarities = np.where(copies % 2 == 0, 1.0, -1.0).astype(np.float32)

    def adapt(self, observation: Mapping[str, Any] | np.ndarray) -> np.ndarray:
        """Return one stimulation value in ``[0, 1]`` per sensory neuron."""

        raw = observation.get("observation") if isinstance(observation, Mapping) else observation
        if raw is None:
            raise KeyError("Observation dictionary is missing the 'observation' entry")
        vector = np.asarray(raw, dtype=np.float32)
        if vector.shape != (self.observation_dim,):
            raise ValueError(
                f"Expected battle observation shape ({self.observation_dim},), "
                f"received {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("Battle observation contains non-finite values")

        signed_features = vector[self.feature_indices] * self.polarities
        return np.clip(signed_features, 0.0, 1.0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class DescendingSpikeCount:
    root_id: int
    cell_type: str
    spikes: int


@dataclass(frozen=True)
class FlySpikeReport:
    sensory_stimulation: np.ndarray
    spike_counts: np.ndarray
    sensory_indices: np.ndarray
    central_indices: np.ndarray
    descending_indices: np.ndarray
    graph: FlyGraph

    @property
    def sensory_neurons_stimulated(self) -> int:
        return int(np.count_nonzero(self.sensory_stimulation > 0.0))

    @property
    def total_cns_spikes(self) -> int:
        return int(self.spike_counts.sum())

    @property
    def active_central_neurons(self) -> int:
        return int(np.count_nonzero(self.spike_counts[self.central_indices]))

    @property
    def active_descending_neurons(self) -> int:
        return int(np.count_nonzero(self.spike_counts[self.descending_indices]))

    def top_descending(self, limit: int = 10) -> list[DescendingSpikeCount]:
        """Return descending cells ordered by spike count, then stable body ID."""

        indices = sorted(
            self.descending_indices.tolist(),
            key=lambda index: (-int(self.spike_counts[index]), self.graph.neurons[index].root_id),
        )[:limit]
        return [
            DescendingSpikeCount(
                root_id=self.graph.neurons[index].root_id,
                cell_type=(
                    self.graph.neurons[index].cell_type
                    or self.graph.neurons[index].cell_class
                    or "untyped"
                ),
                spikes=int(self.spike_counts[index]),
            )
            for index in indices
        ]


class FlySpikeSimulator:
    """Small deterministic integrate-and-fire diagnostic over the fixed graph."""

    def __init__(
        self,
        graph: FlyGraph,
        *,
        steps: int = 32,
        threshold: float = 1.0,
        membrane_decay: float = 0.85,
        sensory_gain: float = 1.25,
        recurrent_gain: float = 2.0,
    ) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if not 0.0 <= membrane_decay < 1.0:
            raise ValueError("membrane_decay must be in [0, 1)")

        self.graph = graph
        self.steps = steps
        self.threshold = threshold
        self.membrane_decay = membrane_decay
        self.sensory_gain = sensory_gain
        self.recurrent_gain = recurrent_gain
        populations = graph.population_indices
        self.sensory_indices = populations["sensory"]
        self.central_indices = populations["central"]
        self.descending_indices = populations["descending"]

    def run(self, sensory_stimulation: np.ndarray) -> FlySpikeReport:
        stimulation = np.asarray(sensory_stimulation, dtype=np.float32)
        expected_shape = (self.sensory_indices.size,)
        if stimulation.shape != expected_shape:
            raise ValueError(
                f"Expected sensory stimulation shape {expected_shape}, "
                f"received {stimulation.shape}"
            )
        if not np.isfinite(stimulation).all():
            raise ValueError("Sensory stimulation contains non-finite values")

        neuron_count = len(self.graph.neurons)
        membrane = np.zeros(neuron_count, dtype=np.float32)
        recurrent = np.zeros(neuron_count, dtype=np.float32)
        previous_spikes = np.zeros(neuron_count, dtype=np.float32)
        spike_counts = np.zeros(neuron_count, dtype=np.int32)
        sensory_drive = np.zeros(neuron_count, dtype=np.float32)
        sensory_drive[self.sensory_indices] = (
            np.clip(stimulation, 0.0, 1.0) * self.sensory_gain
        )

        for _ in range(self.steps):
            recurrent.fill(0.0)
            if np.any(previous_spikes):
                edge_drive = (
                    self.graph.edge_weights
                    * previous_spikes[self.graph.edge_pre]
                    * self.recurrent_gain
                )
                np.add.at(recurrent, self.graph.edge_post, edge_drive)

            membrane = self.membrane_decay * membrane + sensory_drive + recurrent
            fired = membrane >= self.threshold
            spike_counts += fired
            membrane[fired] -= self.threshold
            membrane = np.maximum(membrane, -self.threshold)
            previous_spikes = fired.astype(np.float32)

        return FlySpikeReport(
            sensory_stimulation=stimulation.copy(),
            spike_counts=spike_counts,
            sensory_indices=self.sensory_indices.copy(),
            central_indices=self.central_indices.copy(),
            descending_indices=self.descending_indices.copy(),
            graph=self.graph,
        )
