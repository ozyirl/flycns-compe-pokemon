"""Run a fixed battle vector through Fly CNS and the untrained action decoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from encoding import OBSERVATION_DIM
from flycns.action_decoder import ACTION_COUNT
from flycns.graph import FlyGraph
from flycns.stimulation import BattleSensoryAdapter, FlySpikeSimulator
from flycns.trainable_decoder import DESCENDING_ACTIVITY_SIZE, TrainableActionDecoder


DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent / "artifacts/fly_connectome.npz"


@dataclass(frozen=True)
class IntegrationSmokeResult:
    sensory_neurons_stimulated: int
    active_descending_neurons: int
    descending_activity: np.ndarray
    logits: np.ndarray
    highest_scoring_action_index: int


def run_smoke(
    *,
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
    decoder: TrainableActionDecoder | None = None,
) -> IntegrationSmokeResult:
    """Exercise the complete forward path without gradients or parameter updates."""

    observation = np.linspace(-0.8, 1.0, OBSERVATION_DIM, dtype=np.float32)
    observation[15] = 1.0  # The encoder's constant feature.

    graph = FlyGraph.load(graph_path)
    stimulation = BattleSensoryAdapter(graph).adapt(observation)
    report = FlySpikeSimulator(graph).run(stimulation)
    descending_activity = report.spike_counts[report.descending_indices].astype(np.float32)
    if descending_activity.shape != (DESCENDING_ACTIVITY_SIZE,):
        raise ValueError(
            f"Expected {DESCENDING_ACTIVITY_SIZE} descending values, "
            f"received {descending_activity.shape}"
        )

    if decoder is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            decoder = TrainableActionDecoder()

    with torch.no_grad():
        logits = decoder(torch.from_numpy(descending_activity)).numpy().copy()
    if logits.shape != (ACTION_COUNT,):
        raise ValueError(f"Expected {ACTION_COUNT} action logits, received {logits.shape}")

    return IntegrationSmokeResult(
        sensory_neurons_stimulated=report.sensory_neurons_stimulated,
        active_descending_neurons=report.active_descending_neurons,
        descending_activity=descending_activity,
        logits=logits,
        highest_scoring_action_index=int(np.argmax(logits)),
    )


if __name__ == "__main__":
    result = run_smoke()
    print(f"sensory neurons stimulated: {result.sensory_neurons_stimulated}")
    print(f"active descending neurons: {result.active_descending_neurons}")
    print(f"shape of descending activity: {result.descending_activity.shape}")
    print(f"26 decoder logits: {result.logits.tolist()}")
    print(f"highest-scoring action index: {result.highest_scoring_action_index}")
