"""Run the Fly CNS spike diagnostic without selecting an action or training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from encoding import OBSERVATION_DIM
from flycns.action_decoder import ACTION_COUNT, DescendingActionDecoder
from flycns.graph import FlyGraph
from flycns.stimulation import BattleSensoryAdapter, FlySpikeSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path, default=Path("artifacts/fly_connectome.npz")
    )
    parser.add_argument(
        "--observation",
        type=Path,
        help="optional JSON PPO observation dict or raw NPY vector",
    )
    parser.add_argument(
        "--action-mask",
        type=Path,
        help="NPY mask required when --observation is a raw vector",
    )
    parser.add_argument(
        "--action-labels",
        type=Path,
        help="optional JSON list of 26 display labels from the current battle",
    )
    parser.add_argument("--steps", type=int, default=32)
    return parser.parse_args()


def smoke_observation() -> dict[str, np.ndarray]:
    """Create a deterministic in-range vector for an offline wiring smoke test."""

    vector = np.linspace(-0.8, 1.0, OBSERVATION_DIM, dtype=np.float32)
    vector[15] = 1.0  # The encoder's constant feature.
    action_mask = np.zeros(ACTION_COUNT, dtype=np.int8)
    action_mask[[0, 1, 2, 3, 4, 6, 7, 8, 9, 22, 23, 24, 25]] = 1
    return {
        "observation": vector,
        "action_mask": action_mask,
    }


def load_observation(
    path: Path | None, action_mask_path: Path | None
) -> dict[str, Any]:
    if path is None:
        return smoke_observation()
    if path.suffix.lower() == ".npy":
        if action_mask_path is None:
            raise ValueError("Raw NPY observations require --action-mask")
        return {
            "observation": np.load(path, allow_pickle=False),
            "action_mask": np.load(action_mask_path, allow_pickle=False),
        }
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if (
            isinstance(payload, dict)
            and "observation" in payload
            and "action_mask" in payload
        ):
            return payload
        if action_mask_path is None:
            raise ValueError(
                "JSON observations must include action_mask or use --action-mask"
            )
        return {
            "observation": np.asarray(payload, dtype=np.float32),
            "action_mask": np.load(action_mask_path, allow_pickle=False),
        }
    raise ValueError("--observation must be a .json or .npy file")


def load_action_labels(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    labels = json.loads(path.read_text())
    if not isinstance(labels, list) or not all(
        isinstance(label, str) for label in labels
    ):
        raise ValueError("--action-labels must contain a JSON list of strings")
    return labels


def main() -> None:
    args = parse_args()
    graph = FlyGraph.load(args.graph)
    observation = load_observation(args.observation, args.action_mask)
    labels = load_action_labels(args.action_labels)
    if labels is None and isinstance(observation.get("action_labels"), list):
        embedded_labels = observation["action_labels"]
        if not all(isinstance(label, str) for label in embedded_labels):
            raise ValueError("Embedded action_labels must be a list of strings")
        labels = embedded_labels
    adapter = BattleSensoryAdapter(graph)
    stimulation = adapter.adapt(observation)
    report = FlySpikeSimulator(graph, steps=args.steps).run(stimulation)

    print("Fly CNS debug only — no action selected, no PPO call, no training")
    print(f"sensory neurons stimulated: {report.sensory_neurons_stimulated}")
    print(f"total CNS spikes: {report.total_cns_spikes}")
    print(f"active central neurons: {report.active_central_neurons}")
    print(f"active descending neurons: {report.active_descending_neurons}")
    print("top 10 descending neurons by spike count:")
    for rank, neuron in enumerate(report.top_descending(10), start=1):
        print(
            f"  {rank:2}. {neuron.root_id:<8} "
            f"{neuron.cell_type:<24.24} {neuron.spikes:>4} spikes"
        )

    descending_ids = [
        graph.neurons[index].root_id for index in report.descending_indices
    ]
    descending_spikes = report.spike_counts[report.descending_indices]
    decoder = DescendingActionDecoder(descending_ids)
    action_scores = decoder.score(descending_spikes)
    legal_actions = decoder.rank_legal(
        action_scores, observation["action_mask"], labels
    )

    print("\nFly CNS hypothetical actions\n")
    for rank, action in enumerate(legal_actions, start=1):
        print(
            f"{rank:2}. {action.label:<31.31} "
            f"score={action.score:>7.3f}  [action {action.action}]"
        )
    choice = decoder.hypothetical_choice(
        action_scores, observation["action_mask"], labels
    )
    print(f"\nwould choose: {choice.label}")
    print("\nNOTE: action was not executed")


if __name__ == "__main__":
    main()
