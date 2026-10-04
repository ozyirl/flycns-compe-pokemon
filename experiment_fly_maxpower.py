"""Continue one frozen FlyCNS checkpoint for 100 MaxBasePower battles, then benchmark it."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration
from stable_baselines3 import PPO

from benchmark_fly_opponent import (
    BATTLES_PER_MATCHUP,
    MatchupResult,
    baseline_action,
    run_matchup,
    scripted_action,
)
from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from flycns.actor_critic import FlyCNSActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_update import update_once
from smoke_fly_rollout import run_one_battle


TRAINING_BATTLES = 100
DEFAULT_V1 = Path("models/flycns_ppo_medium_0lo7c7qq/seed_27_battle_100.pt")
DEFAULT_CANDIDATE = Path("models/flycns_maxpower_v2_candidate.pt")


@dataclass(frozen=True)
class TrainingResult:
    battles: int
    wins: int
    losses: int
    ties: int
    average_reward: float
    average_battle_length: float
    updates: int
    illegal_actions: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_heads(
    policy: FlyCNSPPOPolicy,
    environment_factory: Callable[[], Any],
    *,
    seed: int,
    battles: int = TRAINING_BATTLES,
) -> TrainingResult:
    """Reuse the existing rollout and default one-step PPO update unchanged."""

    if battles < 1:
        raise ValueError("battles must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    wins = losses = ties = illegal_actions = 0
    total_reward = total_turns = 0.0
    for index in range(battles):
        rollout = run_one_battle(environment_factory(), policy, seed=seed + index)
        if rollout.illegal_action_attempted:
            illegal_actions += 1
            raise AssertionError("Fly selected an illegal training action")
        update_once(policy, rollout)
        wins += int(rollout.outcome == "win")
        losses += int(rollout.outcome == "loss")
        ties += int(rollout.outcome == "tie")
        total_reward += rollout.total_reward
        total_turns += len(rollout.turns)
        if (index + 1) % 25 == 0 or index + 1 == battles:
            print(f"training vs max-power: {index + 1}/{battles} battles", flush=True)
    return TrainingResult(
        battles=battles,
        wins=wins,
        losses=losses,
        ties=ties,
        average_reward=total_reward / battles,
        average_battle_length=total_turns / battles,
        updates=battles,
        illegal_actions=illegal_actions,
    )


def benchmark_candidate(
    candidate_path: Path,
    baseline_path: Path,
    server: ServerConfiguration,
    *,
    seed: int,
) -> tuple[MatchupResult, ...]:
    """Use the exact same three matchups and metrics as benchmark_fly_opponent."""

    results: list[MatchupResult] = []
    for name in ("random", "max-power", "baseline"):
        environment = SingleAgentWrapper(
            ShowdownEnv(
                battle_format=DEFAULT_BATTLE_FORMAT,
                server_configuration=server,
            ),
            make_opponent(
                "fly", battle_format=DEFAULT_BATTLE_FORMAT, fly_checkpoint=candidate_path
            ),
        )
        try:
            if name == "baseline":
                model = PPO.load(baseline_path, env=environment)
                model.policy.eval()
                baseline_weights = {
                    key: value.detach().clone()
                    for key, value in model.policy.state_dict().items()
                }
                selector = baseline_action(model)
            else:
                selector = scripted_action(
                    make_opponent(name, battle_format=DEFAULT_BATTLE_FORMAT)
                )
            result = run_matchup(
                name, environment, selector, battles=BATTLES_PER_MATCHUP, seed=seed
            )
            if name == "baseline" and any(
                not torch.equal(value, baseline_weights[key])
                for key, value in model.policy.state_dict().items()
            ):
                raise AssertionError("PPO baseline weights changed during benchmark")
            results.append(result)
        finally:
            environment.close()
    return tuple(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-checkpoint", type=Path, default=DEFAULT_V1)
    parser.add_argument("--candidate-checkpoint", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--baseline-model", type=Path, default=Path("models/baseline"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--server-url",
        default=LocalhostServerConfiguration.websocket_url,
        help="WebSocket URL of the already-running local Showdown server",
    )
    args = parser.parse_args()
    v1 = args.v1_checkpoint.resolve()
    candidate = args.candidate_checkpoint.resolve()
    report_path = candidate.with_suffix(".metrics.json")
    if not v1.is_file():
        parser.error(f"v1 checkpoint not found: {v1}")
    if candidate == v1 or candidate.exists() or report_path.exists():
        parser.error("candidate and report paths must be new and distinct from v1")
    baseline_archive = args.baseline_model
    if not baseline_archive.is_file() and not baseline_archive.with_suffix(".zip").is_file():
        parser.error(f"PPO baseline not found: {baseline_archive}")

    original_hash = sha256(v1)
    server = ServerConfiguration(
        args.server_url, LocalhostServerConfiguration.authentication_url
    )
    policy = FlyCNSPPOPolicy(actor_critic=FlyCNSActorCritic.load_weights(v1))

    def make_training_environment() -> SingleAgentWrapper:
        return SingleAgentWrapper(
            ShowdownEnv(
                battle_format=DEFAULT_BATTLE_FORMAT,
                server_configuration=server,
            ),
            make_opponent("max-power", battle_format=DEFAULT_BATTLE_FORMAT),
        )

    training = train_heads(policy, make_training_environment, seed=args.seed)
    if sha256(v1) != original_hash:
        raise AssertionError("v1 checkpoint changed during training")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    policy.actor_critic.save_weights(candidate)
    reloaded = FlyCNSActorCritic.load_weights(candidate)
    if any(
        not torch.equal(saved, loaded)
        for saved, loaded in zip(
            policy.actor_critic.parameters(), reloaded.parameters(), strict=True
        )
    ):
        raise AssertionError("Candidate checkpoint did not reproduce trained weights")
    candidate_hash = sha256(candidate)
    print(f"candidate saved: {candidate}", flush=True)

    benchmark = benchmark_candidate(candidate, args.baseline_model, server, seed=args.seed)
    if sha256(v1) != original_hash or sha256(candidate) != candidate_hash:
        raise AssertionError("A checkpoint changed during read-only benchmarking")
    report = {
        "v1_checkpoint": str(v1),
        "v1_sha256": original_hash,
        "candidate_checkpoint": str(candidate),
        "candidate_sha256": candidate_hash,
        "training": asdict(training),
        "benchmark": [
            {
                "opponent": result.name,
                "battles": len(result.rewards),
                "wins": result.wins,
                "losses": result.losses,
                "ties": result.ties,
                "win_rate": result.win_rate,
                "average_reward": result.average_reward,
                "average_battle_length": result.average_battle_length,
                "fly_illegal_actions": result.fly_illegal_actions,
                "opposing_illegal_actions": result.opposing_illegal_actions,
            }
            for result in benchmark
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"v1 SHA-256 unchanged: {original_hash}")
    print(
        f"training: {training.battles} battles, {training.updates} PPO updates, "
        f"{training.wins}/{training.losses}/{training.ties} W/L/T, "
        f"average reward {training.average_reward:.3f}, "
        f"average turns {training.average_battle_length:.2f}"
    )
    print("\nFly candidate benchmark (Fly perspective; 100 battles each)")
    print("opponent      wins losses ties win_rate avg_reward avg_turns illegal(Fly/other)")
    for result in benchmark:
        print(
            f"{result.name:<13} {result.wins:>4} {result.losses:>6} "
            f"{result.ties:>4} {result.win_rate:>8.1%} "
            f"{result.average_reward:>10.3f} {result.average_battle_length:>9.2f} "
            f"{result.fly_illegal_actions}/{result.opposing_illegal_actions}"
        )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
