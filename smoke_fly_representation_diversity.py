"""Compare Fly CNS descending activity for five synthetic battle feature profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from encoding import ACTIVE_FEATURES, GLOBAL_FEATURES, OBSERVATION_DIM, TEAM_SIZE
from flycns.graph import FlyGraph
from flycns.stimulation import BattleSensoryAdapter, FlySpikeSimulator


DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent / "artifacts/fly_connectome.npz"


@dataclass(frozen=True)
class Representation:
    name: str
    descending_spikes: np.ndarray
    active_descending_neurons: int
    total_descending_spikes: int
    differs_from_all_others: bool


@dataclass(frozen=True)
class DiversityResult:
    representations: tuple[Representation, ...]
    cosine_similarity: np.ndarray


def observation_profiles() -> dict[str, np.ndarray]:
    """Vary distinct encoder feature blocks without constructing a battle."""

    baseline = np.linspace(-0.8, 1.0, OBSERVATION_DIM, dtype=np.float32)
    baseline[15] = 1.0  # The encoder's constant feature.

    team_size = TEAM_SIZE * 6
    own_team = slice(GLOBAL_FEATURES, GLOBAL_FEATURES + team_size)
    opponent_team = slice(own_team.stop, own_team.stop + team_size)
    active_size = ACTIVE_FEATURES // 2
    own_active = slice(opponent_team.stop, opponent_team.stop + active_size)
    opponent_active = slice(own_active.stop, own_active.stop + active_size)

    profiles = {"baseline": baseline}
    for name, high_block, low_block in (
        ("own_team_features", own_team, opponent_team),
        ("opponent_team_features", opponent_team, own_team),
        ("own_active_features", own_active, opponent_active),
        ("opponent_active_features", opponent_active, own_active),
    ):
        observation = baseline.copy()
        observation[high_block] = 1.0
        observation[low_block] = 0.0
        profiles[name] = observation
    return profiles


def run_smoke(graph_path: str | Path = DEFAULT_GRAPH_PATH) -> DiversityResult:
    """Run the fixed simulator once per observation and compare output vectors."""

    graph = FlyGraph.load(graph_path)
    adapter = BattleSensoryAdapter(graph)
    simulator = FlySpikeSimulator(graph)
    profiles = observation_profiles()
    names = tuple(profiles)
    vectors: list[np.ndarray] = []
    for observation in profiles.values():
        stimulation = adapter.adapt(observation)
        report = simulator.run(stimulation)
        vectors.append(report.spike_counts[report.descending_indices].copy())

    activity = np.stack(vectors).astype(np.float64)
    norms = np.linalg.norm(activity, axis=1)
    denominators = np.outer(norms, norms)
    cosine_similarity = np.divide(
        activity @ activity.T,
        denominators,
        out=np.full(denominators.shape, np.nan, dtype=np.float64),
        where=denominators > 0,
    )
    representations = tuple(
        Representation(
            name=name,
            descending_spikes=vector,
            active_descending_neurons=int(np.count_nonzero(vector)),
            total_descending_spikes=int(vector.sum()),
            differs_from_all_others=all(
                not np.array_equal(vector, other)
                for other_index, other in enumerate(vectors)
                if other_index != index
            ),
        )
        for index, (name, vector) in enumerate(zip(names, vectors, strict=True))
    )
    return DiversityResult(representations, cosine_similarity)


if __name__ == "__main__":
    result = run_smoke()
    print("Fly CNS descending representation diversity (synthetic feature profiles)")
    print("profile                       active DNs  DN spikes  differs from all others")
    for item in result.representations:
        print(
            f"{item.name:<29} {item.active_descending_neurons:>10} "
            f"{item.total_descending_spikes:>10}  "
            f"{'yes' if item.differs_from_all_others else 'no'}"
        )

    print("\nCosine similarity between descending vectors:")
    print("index  profile")
    for index, item in enumerate(result.representations):
        print(f"{index:>5}  {item.name}")
    print("\n       " + " ".join(f"{index:>7}" for index in range(len(result.representations))))
    for index, row in enumerate(result.cosine_similarity):
        values = " ".join(
            f"{value:>7.4f}" if np.isfinite(value) else "    N/A"
            for value in row
        )
        print(f"{index:>5}  {values}")
