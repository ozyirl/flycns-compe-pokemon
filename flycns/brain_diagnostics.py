"""Compact, readable summary of the Fly CNS activity behind one battle decision."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from poke_env.battle import Battle
from poke_env.environment import SinglesEnv

from flycns.ppo_policy import PolicyDecision
from flycns.protocol_replay import order_to_choice
from flycns.stimulation import FlySpikeReport


POPULATION_LABELS = {
    "sensory": "sensory inputs",
    "kenyon": "mushroom body",
    "modulatory": "modulatory (dopamine, serotonin, octopamine)",
    "central": "central brain",
    "descending": "descending",
}
TOP_DESCENDING_LIMIT = 8
# Per-neuron bytes use 255 for "never fired"; spike counts cannot reach it in 32 steps.
NEVER_FIRED = 255


def graph_fingerprint(root_ids: Sequence[int]) -> str:
    """Identify a neuron ordering, so per-neuron arrays and layouts can be matched."""

    return hashlib.sha256(np.asarray(root_ids, dtype="<i8").tobytes()).hexdigest()[:16]


def encode_bytes(values: np.ndarray) -> str:
    """Base64 of ``values`` as unsigned bytes, the compact form the frontend decodes."""

    return base64.b64encode(np.asarray(values, dtype=np.uint8).tobytes()).decode("ascii")


def neuron_activity(report: FlySpikeReport) -> dict[str, Any] | None:
    """Each neuron's first spike step and spike count, in graph order."""

    if report.first_spike_steps is None:
        return None
    first = np.where(report.first_spike_steps >= 0, report.first_spike_steps, NEVER_FIRED)
    return {
        "graph": graph_fingerprint([neuron.root_id for neuron in report.graph.neurons]),
        "first_spike_step": encode_bytes(first),
        "spikes": encode_bytes(np.minimum(report.spike_counts, NEVER_FIRED - 1)),
    }


def choice_label(choice: str, request: dict[str, Any]) -> str:
    """Name what a choice such as ``move 2 terastallize`` or ``switch 3`` does."""

    kind, slot, *gimmicks = choice.split()
    if kind == "switch":
        return request["side"]["pokemon"][int(slot) - 1]["details"].split(",")[0]
    label = request["active"][0]["moves"][int(slot) - 1]["move"]
    return " + ".join([label, *(gimmick.capitalize() for gimmick in gimmicks)])


def summarize_decision(
    report: FlySpikeReport, decision: PolicyDecision, battle: Battle, request: dict[str, Any]
) -> dict[str, Any]:
    """Describe the spike simulation and the policy's legal choices for one decision."""

    populations = []
    for key, indices in report.graph.population_indices.items():
        counts = report.spike_counts[indices]
        population: dict[str, Any] = {
            "key": key,
            "label": POPULATION_LABELS.get(key, key),
            "neurons": int(indices.size),
            "active": int(np.count_nonzero(counts)),
            "spikes": int(counts.sum()),
            "first_spike_step": None,
        }
        if report.first_spike_steps is not None:
            fired = report.first_spike_steps[indices]
            fired = fired[fired >= 0]
            if fired.size:
                population["first_spike_step"] = int(fired.min())
        populations.append(population)

    logits = np.asarray(decision.policy_logits, dtype=np.float64)
    legal = np.flatnonzero(np.isfinite(logits))
    weights = np.exp(logits[legal] - logits[legal].max())
    actions = []
    for index, probability in zip(legal.tolist(), (weights / weights.sum()).tolist(), strict=True):
        try:
            order = SinglesEnv.action_to_order(np.int64(index), battle, strict=True)
            choice = order_to_choice(order, battle, request)
        except (ValueError, KeyError, IndexError):
            continue
        actions.append({
            "index": index,
            "choice": choice,
            "label": choice_label(choice, request),
            "probability": round(probability, 4),
            "selected": index == decision.selected_action_index,
        })
    actions.sort(key=lambda action: -action["probability"])

    return {
        "total_spikes": report.total_cns_spikes,
        "active_neurons": int(np.count_nonzero(report.spike_counts)),
        "neuron_count": int(report.spike_counts.size),
        "sensory_neurons_stimulated": report.sensory_neurons_stimulated,
        "steps": int(report.spikes_per_step.size) if report.spikes_per_step is not None else None,
        "spikes_per_step": (
            report.spikes_per_step.tolist() if report.spikes_per_step is not None else []
        ),
        "populations": populations,
        "top_descending": [
            {"root_id": neuron.root_id, "cell_type": neuron.cell_type, "spikes": neuron.spikes}
            for neuron in report.top_descending(TOP_DESCENDING_LIMIT)
        ],
        "actions": actions,
        "state_value": decision.state_value,
        "neuron_activity": neuron_activity(report),
    }
