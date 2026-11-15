"""Compact, readable summary of the Fly CNS activity behind one battle decision."""

from __future__ import annotations

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
    }
