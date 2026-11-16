"""Read-only HTTP API for the Fly CNS debug pipeline and the trained battle policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from encoding import OBSERVATION_DIM
from flycns.action_decoder import ACTION_COUNT, DescendingActionDecoder
from flycns.brain_diagnostics import summarize_decision
from flycns.graph import FlyGraph
from flycns.player import decide_with_report, load_frozen_policy
from flycns.protocol_replay import order_to_choice, replay_battle
from flycns.stimulation import BattleSensoryAdapter, FlySpikeSimulator


SERVICE_VERSION = "0.4.0"
DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent / "artifacts/fly_connectome.npz"
# Copy of models/flycns_matchup_20261110_2049/matchup/seed_17_from1000/battle_1000.pt.
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent / "deploy/fly_matchup.pt"
MAX_PROTOCOL_LINES = 20_000


class ChooseRequest(BaseModel):
    format: str = Field(default="gen9randombattle", pattern=r"^gen9[a-z0-9]+$")
    protocol: list[str] = Field(min_length=1, max_length=MAX_PROTOCOL_LINES)
    # Clients whose rules forbid mega, z-move, dynamax and tera choices set this to false.
    gimmicks: bool = True


class PredictRequest(BaseModel):
    observation: list[float] = Field(min_length=OBSERVATION_DIM, max_length=OBSERVATION_DIM)
    action_mask: list[int] = Field(min_length=ACTION_COUNT, max_length=ACTION_COUNT)


class TopDescendingNeuron(BaseModel):
    root_id: int
    cell_type: str
    spikes: int


class BrainPopulation(BaseModel):
    key: str
    label: str
    neurons: int
    active: int
    spikes: int
    # Simulation step (of ``BrainDiagnostics.steps``) at which the population first fired.
    first_spike_step: int | None


class ActionProbability(BaseModel):
    index: int
    choice: str
    label: str
    probability: float
    selected: bool


class NeuronActivity(BaseModel):
    # Fingerprint of the neuron order; a brain layout built for the same graph matches it.
    graph: str
    # Base64 bytes, one per neuron in graph order: first spike step (255 if it never
    # fired) and spike count.
    first_spike_step: str
    spikes: str


class BrainDiagnostics(BaseModel):
    total_spikes: int
    active_neurons: int
    neuron_count: int
    sensory_neurons_stimulated: int
    steps: int | None
    spikes_per_step: list[int]
    populations: list[BrainPopulation]
    top_descending: list[TopDescendingNeuron]
    actions: list[ActionProbability]
    state_value: float
    neuron_activity: NeuronActivity | None


class ChooseResponse(BaseModel):
    choice: str
    action_index: int
    turn: int
    state_value: float
    diagnostics: BrainDiagnostics


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
    model: str


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


class FlyChoiceService:
    """Choose a Showdown action for the player whose protocol stream is replayed."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH) -> None:
        self.checkpoint_path = checkpoint_path
        self.policy = load_frozen_policy(checkpoint_path)

    def choose(
        self, protocol: list[str], battle_format: str, *, allow_gimmicks: bool = True
    ) -> dict[str, Any]:
        battle, request = replay_battle(protocol, battle_format)
        if request.get("wait") or request.get("teamPreview"):
            raise ValueError("the last request does not ask for a move or switch")
        order, decision, report = decide_with_report(
            self.policy, battle, allow_gimmicks=allow_gimmicks
        )
        return {
            "choice": order_to_choice(order, battle, request),
            "action_index": decision.selected_action_index,
            "turn": battle.turn,
            "state_value": decision.state_value,
            "diagnostics": summarize_decision(report, decision, battle, request),
        }


def create_app(
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> FastAPI:
    service = FlyInferenceService(graph_path)
    chooser = FlyChoiceService(checkpoint_path)
    application = FastAPI(
        title="Fly CNS Inference",
        version=SERVICE_VERSION,
        description="Read-only access to the Fly CNS debug pipeline and battle policy.",
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
            "model": chooser.checkpoint_path.name,
        }

    @application.post("/v1/choose", response_model=ChooseResponse)
    def choose(request: ChooseRequest) -> dict[str, Any]:
        try:
            return chooser.choose(
                request.protocol, request.format, allow_gimmicks=request.gimmicks
            )
        except (ValueError, KeyError, IndexError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail=f"Cannot choose: {error}") from error

    @application.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> dict[str, Any]:
        try:
            return service.predict(request.observation, request.action_mask)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return application


app = create_app()


if __name__ == "__main__":
    import os
    import socket

    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    if host == "::":
        # asyncio binds "::" as IPv6-only. Railway's private network uses IPv6 and
        # Docker's port forwarding uses IPv4, so listen on both.
        listener = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("::", port))
        uvicorn.Server(uvicorn.Config(app)).run(sockets=[listener])
    else:
        uvicorn.run("serve_fly:app", host=host, port=port, reload=False)
