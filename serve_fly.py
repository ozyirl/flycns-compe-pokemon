"""Local, read-only HTTP API for the Fly CNS debug inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from encoding import OBSERVATION_DIM
from flycns.action_decoder import ACTION_COUNT, DescendingActionDecoder
from flycns.graph import FlyGraph
from flycns.stimulation import BattleSensoryAdapter, FlySpikeSimulator


SERVICE_VERSION = "0.1.0"
DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent / "artifacts/fly_connectome.npz"


class PredictRequest(BaseModel):
    observation: list[float] = Field(min_length=OBSERVATION_DIM, max_length=OBSERVATION_DIM)
    action_mask: list[int] = Field(min_length=ACTION_COUNT, max_length=ACTION_COUNT)


class TopDescendingNeuron(BaseModel):
    root_id: int
    cell_type: str
    spikes: int


class CNSDiagnostics(BaseModel):
    sensory_neurons_stimulated: int
    total_cns_spikes: int
    active_central_neurons: int
    active_descending_neurons: int
    top_descending_neurons: list[TopDescendingNeuron]


class PredictResponse(BaseModel):
    raw_scores: list[float]
    legal_masked_scores: list[float | None]
    hypothetical_selected_action_index: int
    diagnostics: CNSDiagnostics


class HealthResponse(BaseModel):
    status: str
    neuron_count: int
    version: str


class FlyInferenceService:
    """Own the immutable graph and deterministic debug inference components."""

    def __init__(self, graph_path: str | Path = DEFAULT_GRAPH_PATH) -> None:
        self.graph = FlyGraph.load(graph_path)
        self.adapter = BattleSensoryAdapter(self.graph)
        self.simulator = FlySpikeSimulator(self.graph)

        descending_indices = self.graph.population_indices["descending"]
        descending_root_ids = [
            self.graph.neurons[index].root_id for index in descending_indices
        ]
        if len(descending_root_ids) != 512:
            raise ValueError(
                "The HTTP debug service requires exactly 512 descending neurons; "
                f"graph contains {len(descending_root_ids)}"
            )
        self.decoder = DescendingActionDecoder(descending_root_ids)

    def predict(
        self, observation: list[float], action_mask: list[int]
    ) -> dict[str, Any]:
        observation_array = np.asarray(observation, dtype=np.float32)
        mask_array = np.asarray(action_mask)
        if observation_array.shape != (OBSERVATION_DIM,):
            raise ValueError(
                f"observation must contain exactly {OBSERVATION_DIM} values"
            )
        if not np.isfinite(observation_array).all():
            raise ValueError("observation values must be finite")
        if mask_array.shape != (ACTION_COUNT,):
            raise ValueError(f"action_mask must contain exactly {ACTION_COUNT} values")
        if not np.isin(mask_array, (0, 1)).all():
            raise ValueError("action_mask values must be 0 or 1")
        if not np.any(mask_array):
            raise ValueError("action_mask must contain at least one legal action")
        mask_array = mask_array.astype(np.int8, copy=False)

        stimulation = self.adapter.adapt(observation_array)
        report = self.simulator.run(stimulation)
        descending_spikes = report.spike_counts[report.descending_indices]
        raw_scores = self.decoder.score(descending_spikes)
        choice = self.decoder.hypothetical_choice(raw_scores, mask_array)
        legal_masked_scores: list[float | None] = [
            float(score) if is_legal else None
            for score, is_legal in zip(raw_scores, mask_array, strict=True)
        ]

        return {
            "raw_scores": raw_scores.astype(float).tolist(),
            "legal_masked_scores": legal_masked_scores,
            "hypothetical_selected_action_index": choice.action,
            "diagnostics": {
                "sensory_neurons_stimulated": report.sensory_neurons_stimulated,
                "total_cns_spikes": report.total_cns_spikes,
                "active_central_neurons": report.active_central_neurons,
                "active_descending_neurons": report.active_descending_neurons,
                "top_descending_neurons": [
                    {
                        "root_id": neuron.root_id,
                        "cell_type": neuron.cell_type,
                        "spikes": neuron.spikes,
                    }
                    for neuron in report.top_descending(10)
                ],
            },
        }


def create_app(graph_path: str | Path = DEFAULT_GRAPH_PATH) -> FastAPI:
    service = FlyInferenceService(graph_path)
    application = FastAPI(
        title="Fly CNS Debug Inference",
        version=SERVICE_VERSION,
        description="Read-only local access to the fixed Fly CNS debug pipeline.",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @application.get("/health", response_model=HealthResponse)
    def health() -> dict[str, int | str]:
        return {
            "status": "ok",
            "neuron_count": len(service.graph.neurons),
            "version": SERVICE_VERSION,
        }

    @application.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> dict[str, Any]:
        try:
            return service.predict(request.observation, request.action_mask)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve_fly:app", host="127.0.0.1", port=8000, reload=False)
