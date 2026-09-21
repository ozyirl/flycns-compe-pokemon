"""PyTorch feature extractor backed by a frozen sparse fly connectome."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from flycns.graph import FlyGraph


class FlyConnectomeFeaturesExtractor(BaseFeaturesExtractor):
    """Rate-coded relaxation over real selected MaleCNS connections.

    The sparse connectome is fixed. Input and output adapters are trainable when this
    extractor is eventually used by PPO. No persistent hidden state is used, because
    PPO minibatches do not preserve battle sequence ordering.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        graph_path: str = "artifacts/fly_connectome.npz",
        features_dim: int = 128,
        propagation_steps: int | None = None,
        leak: float | None = None,
        telemetry: bool = False,
    ) -> None:
        state_space = observation_space.spaces["observation"]
        if not isinstance(state_space, spaces.Box):
            raise TypeError("The 'observation' entry must be a Box")
        super().__init__(observation_space, features_dim)

        graph = FlyGraph.load(Path(graph_path))
        populations = graph.population_indices
        if populations["sensory"].size == 0 or populations["descending"].size == 0:
            raise ValueError("Fly graph requires sensory inputs and descending outputs")
        input_dim = int(state_space.shape[0])
        self.input_adapter = nn.Linear(input_dim, int(populations["sensory"].size))
        self.output_adapter = nn.Linear(int(populations["descending"].size), features_dim)
        self.propagation_steps = propagation_steps or graph.config.propagation_steps
        self.leak = graph.config.leak if leak is None else leak
        self.telemetry_enabled = telemetry
        self.last_activity: torch.Tensor | None = None
        self.neuron_metadata = tuple(graph.neurons)

        indices = torch.as_tensor(
            [graph.edge_post.tolist(), graph.edge_pre.tolist()], dtype=torch.long
        )
        values = torch.as_tensor(graph.edge_weights, dtype=torch.float32)
        connectivity = torch.sparse_coo_tensor(
            indices,
            values,
            size=(len(graph.neurons), len(graph.neurons)),
            check_invariants=False,
        ).coalesce()
        self.register_buffer("connectivity", connectivity)
        self.register_buffer(
            "input_indices", torch.as_tensor(populations["sensory"], dtype=torch.long)
        )
        self.register_buffer(
            "output_indices",
            torch.as_tensor(populations["descending"], dtype=torch.long),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        stimulus = torch.tanh(self.input_adapter(observations["observation"]))
        batch_size = stimulus.shape[0]
        activity = stimulus.new_zeros((batch_size, self.connectivity.shape[0]))
        injected = activity.index_copy(1, self.input_indices, stimulus)
        activity = injected
        for _ in range(self.propagation_steps):
            recurrent = torch.sparse.mm(self.connectivity, activity.transpose(0, 1))
            recurrent = recurrent.transpose(0, 1)
            activity = torch.tanh(
                (1.0 - self.leak) * activity + self.leak * (recurrent + injected)
            )
        if self.telemetry_enabled:
            self.last_activity = activity.detach().cpu()
        descending = activity.index_select(1, self.output_indices)
        return torch.tanh(self.output_adapter(descending))

    def activity_snapshot(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return UI-ready metadata for the most active cells in the last forward pass."""

        if self.last_activity is None:
            return []
        mean_activity = self.last_activity.abs().mean(dim=0)
        count = min(limit, mean_activity.numel())
        values, indices = torch.topk(mean_activity, count)
        return [
            {
                "root_id": self.neuron_metadata[int(index)].root_id,
                "population": self.neuron_metadata[int(index)].population,
                "cell_type": self.neuron_metadata[int(index)].cell_type,
                "cell_class": self.neuron_metadata[int(index)].cell_class,
                "neurotransmitter": self.neuron_metadata[int(index)].neurotransmitter,
                "activity": float(value),
            }
            for value, index in zip(values, indices, strict=True)
        ]
