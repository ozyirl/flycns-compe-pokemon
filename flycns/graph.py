"""Build and load an inspectable sparse graph from the MaleCNS v1.0 CSVs."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from flycns.config import FlyGraphConfig
from flycns.receptors import receptor_for


POPULATION_ORDER = ("sensory", "kenyon", "modulatory", "central", "descending")


@dataclass(frozen=True)
class NeuronRecord:
    root_id: int
    population: str
    super_class: str
    cell_class: str
    cell_type: str
    neurotransmitter: str
    nt_confidence: float


@dataclass
class FlyGraph:
    """Compact graph artifact used by both the policy and visual inspection tools."""

    neurons: list[NeuronRecord]
    edge_pre: np.ndarray
    edge_post: np.ndarray
    edge_synapses: np.ndarray
    edge_weights: np.ndarray
    receptor_names: np.ndarray
    config: FlyGraphConfig

    @property
    def population_indices(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for population in POPULATION_ORDER:
            result[population] = np.asarray(
                [i for i, neuron in enumerate(self.neurons) if neuron.population == population],
                dtype=np.int64,
            )
        return result

    def summary(self, top_cell_types: int = 8) -> dict[str, object]:
        populations = Counter(neuron.population for neuron in self.neurons)
        transmitters = Counter(neuron.neurotransmitter for neuron in self.neurons)
        receptors = Counter(str(name) for name in self.receptor_names)
        types_by_population: dict[str, list[dict[str, object]]] = {}
        for population in POPULATION_ORDER:
            counts = Counter(
                neuron.cell_type or neuron.cell_class or "untyped"
                for neuron in self.neurons
                if neuron.population == population
            )
            types_by_population[population] = [
                {"name": name, "count": count}
                for name, count in counts.most_common(top_cell_types)
            ]
        return {
            "neurons": len(self.neurons),
            "edges": int(self.edge_pre.size),
            "synapses": int(self.edge_synapses.sum()),
            "populations": dict(populations),
            "neurotransmitters": dict(transmitters),
            "modeled_receptor_edges": dict(receptors),
            "top_cell_types": types_by_population,
            "config": self.config.to_dict(),
            "provenance": {
                "cell_identity": "Janelia MaleCNS v1.0 neuron annotations",
                "connectivity": "Janelia MaleCNS v1.0 synapse counts",
                "receptors": "modeled assumptions; not measured by the source CSVs",
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            root_ids=np.asarray([n.root_id for n in self.neurons], dtype=np.int64),
            populations=np.asarray([n.population for n in self.neurons], dtype="U16"),
            super_classes=np.asarray([n.super_class for n in self.neurons], dtype="U32"),
            cell_classes=np.asarray([n.cell_class for n in self.neurons], dtype="U48"),
            cell_types=np.asarray([n.cell_type for n in self.neurons], dtype="U96"),
            neurotransmitters=np.asarray(
                [n.neurotransmitter for n in self.neurons], dtype="U8"
            ),
            nt_confidences=np.asarray(
                [n.nt_confidence for n in self.neurons], dtype=np.float32
            ),
            edge_pre=self.edge_pre.astype(np.int64),
            edge_post=self.edge_post.astype(np.int64),
            edge_synapses=self.edge_synapses.astype(np.int32),
            edge_weights=self.edge_weights.astype(np.float32),
            receptor_names=self.receptor_names.astype("U48"),
            config_json=np.asarray(json.dumps(self.config.to_dict())),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FlyGraph":
        with np.load(Path(path), allow_pickle=False) as data:
            config = FlyGraphConfig(**json.loads(str(data["config_json"])))
            neurons = [
                NeuronRecord(
                    root_id=int(root_id),
                    population=str(population),
                    super_class=str(super_class),
                    cell_class=str(cell_class),
                    cell_type=str(cell_type),
                    neurotransmitter=str(neurotransmitter),
                    nt_confidence=float(confidence),
                )
                for root_id, population, super_class, cell_class, cell_type,
                neurotransmitter, confidence in zip(
                    data["root_ids"],
                    data["populations"],
                    data["super_classes"],
                    data["cell_classes"],
                    data["cell_types"],
                    data["neurotransmitters"],
                    data["nt_confidences"],
                    strict=True,
                )
            ]
            return cls(
                neurons=neurons,
                edge_pre=data["edge_pre"].copy(),
                edge_post=data["edge_post"].copy(),
                edge_synapses=data["edge_synapses"].copy(),
                edge_weights=data["edge_weights"].copy(),
                receptor_names=data["receptor_names"].copy(),
                config=config,
            )


def _stable_sample(rows: Iterable[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    def key(row: dict[str, str]) -> bytes:
        identity = f"{seed}:{row['Root ID']}".encode()
        return hashlib.blake2b(identity, digest_size=8).digest()

    return sorted(rows, key=key)[:count]


def _candidate_populations(row: dict[str, str]) -> tuple[str, ...]:
    super_class = row.get("Super Class", "")
    cell_class = row.get("Class", "")
    transmitter = row.get("Predicted NT type", "")
    result: list[str] = []
    if "sensory" in super_class or super_class == "visual_projection":
        result.append("sensory")
    if cell_class == "Kenyon_Cell":
        result.append("kenyon")
    # Use the curated DAN class for dopamine cells. The source predictor labels most
    # Kenyon cells as DA, so transmitter alone would incorrectly reclassify the
    # mushroom body as a modulatory population. SER/OCT lack an equivalent complete
    # class annotation and are included by transmitter.
    if cell_class == "DAN" or transmitter in {"SER", "OCT"}:
        result.append("modulatory")
    if super_class in {"cb_intrinsic", "ol_intrinsic", "vnc_intrinsic"}:
        result.append("central")
    if super_class == "descending_neuron":
        result.append("descending")
    return tuple(result)


def _read_neurons(path: Path, config: FlyGraphConfig) -> list[NeuronRecord]:
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            for population in _candidate_populations(row):
                candidates[population].append(row)

    caps = {
        "sensory": config.sensory_neurons,
        "kenyon": config.kenyon_neurons,
        "modulatory": config.modulatory_neurons,
        "central": config.central_neurons,
        "descending": config.descending_neurons,
    }
    selected: dict[int, tuple[str, dict[str, str]]] = {}
    # Generic central membership is applied first. More informative functional roles
    # then win overlaps, keeping Kenyon, modulatory, and descending labels visible.
    population_priority = ("central", "sensory", "kenyon", "modulatory", "descending")
    for population in population_priority:
        for row in _stable_sample(candidates[population], caps[population], config.seed):
            selected[int(row["Root ID"])] = (population, row)

    neurons: list[NeuronRecord] = []
    for root_id, (population, row) in sorted(selected.items()):
        confidence = row.get("Predicted NT confidence", "")
        neurons.append(
            NeuronRecord(
                root_id=root_id,
                population=population,
                super_class=row.get("Super Class", ""),
                cell_class=row.get("Class", ""),
                cell_type=row.get("Primary Cell Type", ""),
                neurotransmitter=row.get("Predicted NT type", "") or "UNKNOWN",
                nt_confidence=float(confidence) if confidence else 0.0,
            )
        )
    return neurons


def build_graph(
    neurons_path: Path,
    connections_path: Path,
    config: FlyGraphConfig | None = None,
) -> FlyGraph:
    """Create a selected induced graph and normalize incoming synaptic strength."""

    config = config or FlyGraphConfig()
    neurons = _read_neurons(neurons_path, config)
    index_by_id = {neuron.root_id: index for index, neuron in enumerate(neurons)}
    pair_synapses: dict[tuple[int, int], int] = defaultdict(int)
    with gzip.open(connections_path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            pre = index_by_id.get(int(row["pre_root_id"]))
            post = index_by_id.get(int(row["post_root_id"]))
            if pre is not None and post is not None and pre != post:
                pair_synapses[(pre, post)] += int(row["syn_count"])

    pairs = sorted(pair_synapses)
    edge_pre = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    edge_post = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    edge_synapses = np.asarray([pair_synapses[pair] for pair in pairs], dtype=np.int32)
    receptor_names = np.asarray(
        [receptor_for(neurons[pre].neurotransmitter).receptor for pre in edge_pre],
        dtype="U48",
    )
    raw_weights = np.asarray(
        [
            math.log1p(int(synapses))
            * receptor_for(neurons[int(pre)].neurotransmitter).gain
            * neurons[int(pre)].nt_confidence
            for pre, synapses in zip(edge_pre, edge_synapses, strict=True)
        ],
        dtype=np.float32,
    )
    incoming = np.zeros(len(neurons), dtype=np.float32)
    np.add.at(incoming, edge_post, np.abs(raw_weights))
    denominators = np.maximum(incoming[edge_post], 1.0)
    edge_weights = raw_weights / denominators
    return FlyGraph(
        neurons=neurons,
        edge_pre=edge_pre,
        edge_post=edge_post,
        edge_synapses=edge_synapses,
        edge_weights=edge_weights,
        receptor_names=receptor_names,
        config=config,
    )
