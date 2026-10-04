"""Save a frozen-v1 continuation every 10 battles and evaluate each against MaxBasePower."""

from __future__ import annotations

import argparse
import json
import random
import secrets
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration

from benchmark_fly_opponent import MatchupResult, run_matchup, scripted_action
from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from experiment_fly_maxpower import DEFAULT_V1, TRAINING_BATTLES, TrainingResult, sha256
from flycns.actor_critic import FlyCNSActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_update import update_once
from smoke_fly_rollout import run_one_battle


CHECKPOINT_INTERVAL = 10
EVALUATION_BATTLES = 50


@dataclass(frozen=True)
class CheckpointEvaluation:
    training_battles: int
    checkpoint: Path
    checkpoint_sha256: str
    wins: int
    losses: int
    ties: int
    win_rate: float
    average_reward: float
    average_battle_length: float
    fly_illegal_actions: int
    opposing_illegal_actions: int


def train_with_checkpoints(
    policy: FlyCNSPPOPolicy,
    environment_factory: Callable[[], Any],
    *,
    output_dir: Path,
    seed: int,
    battles: int = TRAINING_BATTLES,
) -> tuple[TrainingResult, tuple[Path, ...]]:
    """Run the existing rollout/update sequence, saving after every tenth update."""

    if battles != TRAINING_BATTLES:
        raise ValueError(f"This experiment requires exactly {TRAINING_BATTLES} battles")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    wins = losses = ties = illegal_actions = 0
    total_reward = total_turns = 0.0
    checkpoints: list[Path] = []
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
        completed = index + 1
        if completed % CHECKPOINT_INTERVAL == 0:
            checkpoint = output_dir / f"battle_{completed:03d}.pt"
            if checkpoint.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint}")
            policy.actor_critic.save_weights(checkpoint)
            checkpoints.append(checkpoint)
            print(f"training vs max-power: {completed}/{battles}; saved {checkpoint.name}", flush=True)

    training = TrainingResult(
        battles=battles,
        wins=wins,
        losses=losses,
        ties=ties,
        average_reward=total_reward / battles,
        average_battle_length=total_turns / battles,
        updates=battles,
        illegal_actions=illegal_actions,
    )
    return training, tuple(checkpoints)


def evaluate_checkpoint(
    checkpoint: Path,
    server: ServerConfiguration,
    *,
    seed: int,
) -> MatchupResult:
    """Run 50 read-only battles, scoring Fly against MaxBasePowerPlayer."""

    checkpoint_hash = sha256(checkpoint)
    nonce = secrets.token_hex(5)
    environment = SingleAgentWrapper(
        ShowdownEnv(
            battle_format=DEFAULT_BATTLE_FORMAT,
            server_configuration=server,
            account_configuration1=AccountConfiguration(f"fs{nonce}a", None),
            account_configuration2=AccountConfiguration(f"fs{nonce}b", None),
        ),
        make_opponent(
            "fly", battle_format=DEFAULT_BATTLE_FORMAT, fly_checkpoint=checkpoint
        ),
    )
    try:
        max_power = make_opponent("max-power", battle_format=DEFAULT_BATTLE_FORMAT)
        result = run_matchup(
            "max-power",
            environment,
            scripted_action(max_power),
            battles=EVALUATION_BATTLES,
            seed=seed,
        )
    finally:
        environment.close()
    if sha256(checkpoint) != checkpoint_hash:
        raise AssertionError(f"Checkpoint changed during read-only evaluation: {checkpoint}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-checkpoint", type=Path, default=DEFAULT_V1)
    parser.add_argument("--output-dir", type=Path, help="new directory for ten checkpoints and report")
    parser.add_argument(
        "--evaluate-existing",
        type=Path,
        help="evaluate ten checkpoints from an interrupted run without retraining",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--server-url",
        default=LocalhostServerConfiguration.websocket_url,
        help="WebSocket URL of the already-running local Showdown server",
    )
    args = parser.parse_args()
    v1 = args.v1_checkpoint.resolve()
    if not v1.is_file():
        parser.error(f"v1 checkpoint not found: {v1}")
    if args.output_dir is not None and args.evaluate_existing is not None:
        parser.error("--output-dir and --evaluate-existing cannot be combined")
    if args.output_dir is not None and args.output_dir.exists():
        parser.error(f"output directory must not already exist: {args.output_dir}")
    if args.evaluate_existing is not None and not args.evaluate_existing.is_dir():
        parser.error(f"checkpoint directory not found: {args.evaluate_existing}")

    v1_hash = sha256(v1)
    artifact_root = Path(__file__).resolve().parent / "models"
    if args.evaluate_existing is not None:
        output_dir = args.evaluate_existing.resolve()
    elif args.output_dir is None:
        artifact_root.mkdir(exist_ok=True)
        output_dir = Path(
            tempfile.mkdtemp(prefix="flycns_maxpower_sweep_", dir=artifact_root)
        )
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "results.json"
    if report_path.exists():
        parser.error(f"refusing to overwrite existing report: {report_path}")
    server = ServerConfiguration(
        args.server_url, LocalhostServerConfiguration.authentication_url
    )

    def make_training_environment() -> SingleAgentWrapper:
        return SingleAgentWrapper(
            ShowdownEnv(
                battle_format=DEFAULT_BATTLE_FORMAT,
                server_configuration=server,
            ),
            make_opponent("max-power", battle_format=DEFAULT_BATTLE_FORMAT),
        )

    if args.evaluate_existing is None:
        policy = FlyCNSPPOPolicy(actor_critic=FlyCNSActorCritic.load_weights(v1))
        training, checkpoints = train_with_checkpoints(
            policy, make_training_environment, output_dir=output_dir, seed=args.seed
        )
        del policy  # Evaluation loads separate frozen FlyCNSPlayer instances from disk.
    else:
        training = None
        checkpoints = tuple(
            output_dir / f"battle_{completed:03d}.pt"
            for completed in range(CHECKPOINT_INTERVAL, TRAINING_BATTLES + 1, CHECKPOINT_INTERVAL)
        )
        if any(not checkpoint.is_file() for checkpoint in checkpoints):
            parser.error("--evaluate-existing requires all ten checkpoint files")
        print(f"resuming read-only evaluation from {output_dir}; no training", flush=True)
    if sha256(v1) != v1_hash:
        raise AssertionError("v1 checkpoint changed during training")
    if len(checkpoints) != TRAINING_BATTLES // CHECKPOINT_INTERVAL:
        raise AssertionError("Expected exactly ten intermediate checkpoints")

    evaluations: list[CheckpointEvaluation] = []
    for completed, checkpoint in zip(
        range(CHECKPOINT_INTERVAL, TRAINING_BATTLES + 1, CHECKPOINT_INTERVAL),
        checkpoints,
        strict=True,
    ):
        print(f"evaluating battle {completed} checkpoint", flush=True)
        result = evaluate_checkpoint(checkpoint, server, seed=args.seed)
        evaluations.append(
            CheckpointEvaluation(
                training_battles=completed,
                checkpoint=checkpoint,
                checkpoint_sha256=sha256(checkpoint),
                wins=result.wins,
                losses=result.losses,
                ties=result.ties,
                win_rate=result.win_rate,
                average_reward=result.average_reward,
                average_battle_length=result.average_battle_length,
                fly_illegal_actions=result.fly_illegal_actions,
                opposing_illegal_actions=result.opposing_illegal_actions,
            )
        )
        if sha256(v1) != v1_hash:
            raise AssertionError("v1 checkpoint changed during evaluation")

    report_path.write_text(
        json.dumps(
            {
                "v1_checkpoint": str(v1),
                "v1_sha256": v1_hash,
                "training": asdict(training) if training is not None else None,
                "evaluation_battles_per_checkpoint": EVALUATION_BATTLES,
                "evaluations": [asdict(item) for item in evaluations],
                "selected_checkpoint": None,
            },
            default=str,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nv1 SHA-256 unchanged: {v1_hash}")
    print("Fly vs MaxBasePowerPlayer (50 read-only battles per checkpoint)")
    print("after  Fly W/L/T  Fly win rate  avg reward  avg turns  illegal(Fly/Max)")
    for item in evaluations:
        print(
            f"{item.training_battles:>5}  "
            f"{item.wins:>2}/{item.losses:>2}/{item.ties:<2}  "
            f"{item.win_rate:>11.1%}  "
            f"{item.average_reward:>10.3f}  "
            f"{item.average_battle_length:>9.2f}  "
            f"{item.fly_illegal_actions}/{item.opposing_illegal_actions}"
        )
    print(f"all ten checkpoints and report: {output_dir}")
    print("no checkpoint was selected or promoted")


if __name__ == "__main__":
    main()
