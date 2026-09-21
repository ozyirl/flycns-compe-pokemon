"""Explicit transmitter-to-modeled-receptor assumptions.

The source connectome predicts presynaptic neurotransmitters. It does not contain
postsynaptic receptor-expression measurements. These channels are therefore model
assumptions, kept centralized and visible so they cannot be mistaken for source data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReceptorChannel:
    transmitter: str
    receptor: str
    effect: str
    gain: float
    source: str = "modeled"


RECEPTOR_CHANNELS: dict[str, ReceptorChannel] = {
    "ACH": ReceptorChannel("ACH", "nicotinic acetylcholine", "excitatory", 1.0),
    "GABA": ReceptorChannel("GABA", "GABA-A-like chloride", "inhibitory", -1.0),
    "GLUT": ReceptorChannel(
        "GLUT", "glutamate-gated chloride", "modeled inhibitory", -0.5
    ),
    "HIST": ReceptorChannel("HIST", "histamine-gated chloride", "inhibitory", -1.0),
    "DA": ReceptorChannel("DA", "dopamine", "modulatory", 0.25),
    "SER": ReceptorChannel("SER", "serotonin", "modulatory", 0.20),
    "OCT": ReceptorChannel("OCT", "octopamine", "modulatory", 0.20),
    "UNKNOWN": ReceptorChannel(
        "UNKNOWN", "unassigned", "disabled until annotated", 0.0
    ),
}


def receptor_for(transmitter: str) -> ReceptorChannel:
    return RECEPTOR_CHANNELS.get(transmitter or "UNKNOWN", RECEPTOR_CHANNELS["UNKNOWN"])
