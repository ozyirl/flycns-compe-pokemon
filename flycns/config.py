"""Configuration for selecting a tractable, inspectable fly sub-connectome."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FlyGraphConfig:
    """Population sizes and deterministic sampling settings.

    The first scaffold deliberately uses a few thousand annotated cells instead of
    pretending that a 166k-neuron brain can be dropped into PPO without design work.
    Every selected cell retains its real MaleCNS body ID and metadata.
    """

    sensory_neurons: int = 512
    kenyon_neurons: int = 1024
    modulatory_neurons: int = 256
    central_neurons: int = 2048
    descending_neurons: int = 512
    seed: int = 0
    propagation_steps: int = 3
    leak: float = 0.5

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
