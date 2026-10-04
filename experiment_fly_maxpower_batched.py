"""Train frozen-v1 Fly heads in 8-battle PPO batches, then benchmark five checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration

from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from experiment_fly_maxpower import DEFAULT_V1, TRAINING_BATTLES, sha256
from experiment_fly_maxpower_sweep import EVALUATION_BATTLES, evaluate_checkpoint
from flycns.actor_critic import FlyCNSActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_update import advantages_and_returns, update_once
from smoke_fly_rollout import RolloutResult, run_one_battle


BATCH_BATTLES = 8
CHECKPOINT_INTERVAL = 20


@dataclass(frozen=True)
class BatchUpdate:
    through_battle: int
    battles: int
    transitions: int
    policy_entropy: float
    value_loss: float
    policy_loss: float
    advantage_mean: float
    advantage_std: float


@dataclass(frozen=True)
class CheckpointSnapshot:
    after_battle: int
    updated_through_battle: int
    pending_rollouts: int
    checkpoint: Path
    checkpoint_sha256: str
    updates_in_window: int
    updated_transitions_in_window: int
    policy_entropy: float
    value_loss: float
    advantage_mean: float
    advantage_std: float


@dataclass(frozen=True)
class CheckpointBenchmark:
    snapshot: CheckpointSnapshot
    wins: int
    losses: int
    ties: int
    win_rate: float
    average_reward: float
    average_turns: float
    fly_illegal_actions: int
    opposing_illegal_actions: int


def combine_rollouts(rollouts: tuple[RolloutResult, ...]) -> RolloutResult:
    """Concatenate complete episodes; their done flags preserve GAE boundaries."""

    if not rollouts:
        raise ValueError("Cannot update from an empty battle batch")
    for rollout in rollouts:
        if not rollout.turns or not rollout.turns[-1].done:
            raise ValueError("Each batched rollout must be complete")
        if rollout.illegal_action_attempted or not rollout.actor_critic_weights_unchanged:
            raise ValueError("Batch contains an illegal or mutated rollout")
    return RolloutResult(
        turns=tuple(turn for rollout in rollouts for turn in rollout.turns),
        total_reward=sum(rollout.total_reward for rollout in rollouts),
        outcome="batch",
        illegal_action_attempted=False,
        actor_critic_weights_unchanged=True,
    )


def train_batched(
    policy: FlyCNSPPOPolicy,
    environment_factory: Callable[[], Any],
    *,
    output_dir: Path,
    seed: int,
) -> tuple[tuple[BatchUpdate, ...], tuple[CheckpointSnapshot, ...]]:
    """Use unchanged one-step PPO loss on 12 batches of eight and a final four."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pending: list[RolloutResult] = []
    updates: list[BatchUpdate] = []
    checkpoints: list[CheckpointSnapshot] = []
    window_updates: list[BatchUpdate] = []
    window_advantages: list[np.ndarray] = []
    updated_through = 0

    for index in range(TRAINING_BATTLES):
        completed = index + 1
        rollout = run_one_battle(environment_factory(), policy, seed=seed + index)
        pending.append(rollout)
        if len(pending) == BATCH_BATTLES or completed == TRAINING_BATTLES:
            combined = combine_rollouts(tuple(pending))
            advantages, _ = advantages_and_returns(combined)
            result = update_once(policy, combined)
            if not result.actor_changed or not result.critic_changed or not result.fly_output_unchanged:
                raise AssertionError("Batched update violated heads-only PPO invariants")
            metric = BatchUpdate(
                through_battle=completed,
                battles=len(pending),
                transitions=len(combined.turns),
                policy_entropy=result.entropy,
                value_loss=result.value_loss,
                policy_loss=result.policy_loss,
                advantage_mean=float(np.mean(advantages)),
                advantage_std=float(np.std(advantages)),
            )
            updates.append(metric)
            window_updates.append(metric)
            window_advantages.append(advantages)
            updated_through = completed
            pending.clear()
            print(
                f"PPO update {len(updates)}/13 after battle {completed} "
                f"({metric.battles} battles, {metric.transitions} turns)",
                flush=True,
            )

        if completed % CHECKPOINT_INTERVAL == 0:
            checkpoint = output_dir / f"battle_{completed:03d}.pt"
            if checkpoint.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint}")
            policy.actor_critic.save_weights(checkpoint)
            if not window_updates:
                raise AssertionError("Checkpoint window had no PPO updates")
            transitions = sum(item.transitions for item in window_updates)
            combined_advantages = np.concatenate(window_advantages)
            snapshots_entropy = sum(
                item.policy_entropy * item.transitions for item in window_updates
            ) / transitions
            snapshots_value_loss = sum(
                item.value_loss * item.transitions for item in window_updates
            ) / transitions
            snapshot = CheckpointSnapshot(
                after_battle=completed,
                updated_through_battle=updated_through,
                pending_rollouts=len(pending),
                checkpoint=checkpoint,
                checkpoint_sha256=sha256(checkpoint),
                updates_in_window=len(window_updates),
                updated_transitions_in_window=transitions,
                policy_entropy=snapshots_entropy,
                value_loss=snapshots_value_loss,
                advantage_mean=float(np.mean(combined_advantages)),
                advantage_std=float(np.std(combined_advantages)),
            )
            checkpoints.append(snapshot)
            window_updates.clear()
            window_advantages.clear()
            print(
                f"saved {checkpoint.name}: updated through {updated_through}, "
                f"{snapshot.pending_rollouts} rollouts pending",
                flush=True,
            )

    if [item.battles for item in updates] != [BATCH_BATTLES] * 12 + [4]:
        raise AssertionError("Expected 12 eight-battle updates and one four-battle update")
    if [item.after_battle for item in checkpoints] != [20, 40, 60, 80, 100]:
        raise AssertionError("Expected checkpoints after battles 20, 40, 60, 80, 100")
    return tuple(updates), tuple(checkpoints)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-checkpoint", type=Path, default=DEFAULT_V1)
    parser.add_argument("--output-dir", type=Path, help="new directory for checkpoints and report")
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
    if args.output_dir is not None and args.output_dir.exists():
        parser.error(f"output directory must not already exist: {args.output_dir}")
    v1_hash = sha256(v1)
    artifact_root = Path(__file__).resolve().parent / "models"
    if args.output_dir is None:
        artifact_root.mkdir(exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="flycns_maxpower_batched_", dir=artifact_root))
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
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

    updates, snapshots = train_batched(
        policy, make_training_environment, output_dir=output_dir, seed=args.seed
    )
    del policy  # Benchmark loads fresh frozen FlyCNSPlayer instances from each checkpoint.
    if sha256(v1) != v1_hash:
        raise AssertionError("v1 checkpoint changed during batched training")

    benchmarks: list[CheckpointBenchmark] = []
    for snapshot in snapshots:
        print(f"benchmarking {snapshot.checkpoint.name}", flush=True)
        result = evaluate_checkpoint(snapshot.checkpoint, server, seed=args.seed)
        benchmarks.append(
            CheckpointBenchmark(
                snapshot=snapshot,
                wins=result.wins,
                losses=result.losses,
                ties=result.ties,
                win_rate=result.win_rate,
                average_reward=result.average_reward,
                average_turns=result.average_battle_length,
                fly_illegal_actions=result.fly_illegal_actions,
                opposing_illegal_actions=result.opposing_illegal_actions,
            )
        )
        if sha256(v1) != v1_hash or sha256(snapshot.checkpoint) != snapshot.checkpoint_sha256:
            raise AssertionError("A checkpoint changed during read-only benchmarking")

    report_path = output_dir / "results.json"
    report_path.write_text(
        json.dumps(
            {
                "v1_checkpoint": str(v1),
                "v1_sha256": v1_hash,
                "training_battles": TRAINING_BATTLES,
                "batch_battles": BATCH_BATTLES,
                "evaluation_battles_per_checkpoint": EVALUATION_BATTLES,
                "updates": [asdict(item) for item in updates],
                "benchmarks": [asdict(item) for item in benchmarks],
                "selected_checkpoint": None,
            },
            default=str,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\nFly v1 batched PPO vs MaxBasePowerPlayer (Fly perspective)")
    print("after updated pending Fly W/L win% avg reward avg turns entropy value loss adv mean adv std")
    for item in benchmarks:
        snapshot = item.snapshot
        print(
            f"{snapshot.after_battle:>5} {snapshot.updated_through_battle:>7} "
            f"{snapshot.pending_rollouts:>7} {item.wins:>2}/{item.losses:<2} "
            f"{item.win_rate:>5.1%} {item.average_reward:>10.3f} "
            f"{item.average_turns:>9.2f} {snapshot.policy_entropy:>7.3f} "
            f"{snapshot.value_loss:>10.3f} {snapshot.advantage_mean:>8.3f} "
            f"{snapshot.advantage_std:>7.3f}"
        )
    print(f"v1 SHA-256 unchanged: {v1_hash}")
    print(f"checkpoints and report: {output_dir}")
    print("no checkpoint was selected or promoted")


if __name__ == "__main__":
    main()
