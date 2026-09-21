"""Create a browser catalog without importing policy or training code."""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


POPULATION_COLORS = {
    "sensory": "#41a5ff",
    "kenyon": "#ff7a38",
    "modulatory": "#b66cff",
    "central": "#39d39f",
    "descending": "#ffd166",
}

PUBLIC_SOURCES = {
    "morphology": "precomputed://gs://flyem-male-cns/v1.0/segmentation",
    "morphology_properties": [
        "precomputed://gs://flyem-male-cns/v1.0/segmentation/type_property",
        "precomputed://gs://flyem-male-cns/v1.0/segmentation/type_and_group_property",
        "precomputed://gs://flyem-male-cns/v1.0/segmentation/tags_property",
        "precomputed://gs://flyem-male-cns/v1.0/segmentation/numeric_properties",
        "precomputed://gs://flyem-male-cns/v1.0/segmentation/instance_property",
    ],
    "em": "precomputed://gs://flyem-male-cns/em/em-clahe-jpeg",
    "brain_outline": "precomputed://gs://flyem-male-cns/rois/fullbrain-major-shells",
    "vnc_outline": "precomputed://gs://flyem-male-cns/rois/vnc-neuropil-shell-v2",
    "brain_neuropils": "precomputed://gs://flyem-male-cns/rois/fullbrain-roi-v5",
    "vnc_neuropils": "precomputed://gs://flyem-male-cns/rois/malecns-vnc-neuropil-roi-v0",
    "synapses": "precomputed://gs://flyem-male-cns/v1.0/male-cns-v1.0-synapses-precomputed/",
    "official_scene": "gs://flyem-male-cns/v1.0/male-cns-v1.0.json",
}


def _strings(values: np.ndarray) -> list[str]:
    return [str(value) for value in values]


def build_catalog(graph_path: Path, connections_path: Path) -> dict[str, Any]:
    """Build visualization metadata and retain neuropil breakdowns for graph edges."""

    with np.load(graph_path, allow_pickle=False) as data:
        root_ids = [int(value) for value in data["root_ids"]]
        populations = _strings(data["populations"])
        super_classes = _strings(data["super_classes"])
        cell_classes = _strings(data["cell_classes"])
        cell_types = _strings(data["cell_types"])
        neurotransmitters = _strings(data["neurotransmitters"])
        confidences = [float(value) for value in data["nt_confidences"]]
        edge_pre = [int(value) for value in data["edge_pre"]]
        edge_post = [int(value) for value in data["edge_post"]]
        edge_synapses = [int(value) for value in data["edge_synapses"]]
        receptor_names = _strings(data["receptor_names"])
        graph_config = json.loads(str(data["config_json"]))

    neurons = [
        {
            "id": root_id,
            "population": population,
            "super_class": super_class,
            "cell_class": cell_class,
            "cell_type": cell_type,
            "neurotransmitter": neurotransmitter,
            "nt_confidence": round(confidence, 4),
        }
        for root_id, population, super_class, cell_class, cell_type,
        neurotransmitter, confidence in zip(
            root_ids,
            populations,
            super_classes,
            cell_classes,
            cell_types,
            neurotransmitters,
            confidences,
            strict=True,
        )
    ]

    selected_pairs = {
        (root_ids[pre], root_ids[post])
        for pre, post in zip(edge_pre, edge_post, strict=True)
    }
    neuropils: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    with gzip.open(connections_path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            pair = (int(row["pre_root_id"]), int(row["post_root_id"]))
            if pair in selected_pairs:
                region = row.get("neuropil", "") or "unassigned"
                neuropils[pair][region] += int(row["syn_count"])

    edges = []
    for pre, post, synapses, receptor in zip(
        edge_pre, edge_post, edge_synapses, receptor_names, strict=True
    ):
        pre_id = root_ids[pre]
        post_id = root_ids[post]
        regions = sorted(
            neuropils[(pre_id, post_id)].items(), key=lambda item: (-item[1], item[0])
        )
        edges.append(
            {
                "pre": pre_id,
                "post": post_id,
                "synapses": synapses,
                "receptor": receptor,
                "neuropils": regions,
            }
        )

    population_counts: dict[str, int] = defaultdict(int)
    for neuron in neurons:
        population_counts[neuron["population"]] += 1
    return {
        "schema_version": 1,
        "mode": "visualization-only",
        "neurons": neurons,
        "edges": edges,
        "summary": {
            "neurons": len(neurons),
            "edges": len(edges),
            "synapses": sum(edge_synapses),
            "populations": dict(population_counts),
        },
        "population_colors": POPULATION_COLORS,
        "sources": PUBLIC_SOURCES,
        "graph_config": graph_config,
        "capabilities": {
            "morphology": True,
            "em": True,
            "brain_outline": True,
            "neuropils": True,
            "synapse_counts": True,
            "synapse_points": True,
        },
    }


def save_catalog(catalog: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(catalog, handle, separators=(",", ":"))


def load_catalog(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)
