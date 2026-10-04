"""Run eight PPO epochs per 8-battle Fly rollout batch from the frozen v1 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
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
from experiment_fly_maxpower_batched import BATCH_BATTLES, CHECKPOINT_INTERVAL, combine_rollouts
from experiment_fly_maxpower_sweep import EVALUATION_BATTLES, evaluate_checkpoint
from flycns.actor_critic import FlyCNSActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_update import advantages_and_returns, update_once
from smoke_fly_rollout import RolloutResult, run_one_battle


EPOCHS_PER_BATCH = 8
# Match update_once's existing clipped policy loss; this is diagnostic-only.
PPO_CLIP_RANGE = 0.2
ORIGINAL_OPTIMIZER_STEPS = TRAINING_BATTLES
EXPECTED_OPTIMIZER_STEPS = math.ceil(TRAINING_BATTLES / BATCH_BATTLES) * EPOCHS_PER_BATCH


@dataclass(frozen=True)
class OptimizerStep:
    through_battle: int
    batch_battles: int
    epoch: int
    transitions: int
    policy_entropy: float
    value_loss: float
    policy_loss: float
    total_loss: float
    advantage_mean: float
    advantage_std: float
    clip_fraction: float
    approximate_kl: float


@dataclass(frozen=True)
class CheckpointSnapshot:
    after_battle: int
    updated_through_battle: int
    pending_rollouts: int
    optimizer_steps_total: int
    optimizer_steps_in_window: int
    checkpoint: Path
    checkpoint_sha256: str
    policy_entropy: float
    value_loss: float
    policy_loss: float
    advantage_mean: float
    advantage_std: float
    clip_fraction: float
    approximate_kl: float


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


def policy_shift_metrics(
    policy: FlyCNSPPOPolicy,
    rollout: RolloutResult,
) -> tuple[float, float]:
    """Measure post-step clipping and approximate KL against the rollout policy."""

    observations = np.stack([turn.observation for turn in rollout.turns])
    masks = np.stack([turn.action_mask for turn in rollout.turns])
    actions = np.asarray([turn.selected_action for turn in rollout.turns], dtype=np.int64)
    old_log_probs = np.asarray([turn.log_probability for turn in rollout.turns], dtype=np.float64)
    current = policy.evaluate_batch(observations, masks, actions)
    log_ratio = current.action_log_probabilities.astype(np.float64) - old_log_probs
    ratio = np.exp(log_ratio)
    if not np.isfinite(ratio).all():
        raise FloatingPointError("Non-finite PPO policy ratio during epoch diagnosis")
    clip_fraction = float(np.mean(np.abs(ratio - 1.0) > PPO_CLIP_RANGE))
    approximate_kl = float(np.mean((ratio - 1.0) - log_ratio))
    return clip_fraction, approximate_kl


def _transition_weighted_mean(steps: list[OptimizerStep], field: str) -> float:
    total_transitions = sum(step.transitions for step in steps)
    return sum(getattr(step, field) * step.transitions for step in steps) / total_transitions


def train_multi_epoch(
    policy: FlyCNSPPOPolicy,
    environment_factory: Callable[[], Any],
    *,
    output_dir: Path,
    seed: int,
) -> tuple[tuple[OptimizerStep, ...], tuple[CheckpointSnapshot, ...]]:
    """Collect 8 complete battles, then apply 8 unchanged full-batch PPO steps."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pending: list[RolloutResult] = []
    all_steps: list[OptimizerStep] = []
    checkpoints: list[CheckpointSnapshot] = []
    window_steps: list[OptimizerStep] = []
    window_advantages: list[np.ndarray] = []
    updated_through = 0

    for index in range(TRAINING_BATTLES):
        completed = index + 1
        pending.append(run_one_battle(environment_factory(), policy, seed=seed + index))
        if len(pending) == BATCH_BATTLES or completed == TRAINING_BATTLES:
            combined = combine_rollouts(tuple(pending))
            advantages, _ = advantages_and_returns(combined)
            advantage_mean = float(np.mean(advantages))
            advantage_std = float(np.std(advantages))
            for epoch in range(1, EPOCHS_PER_BATCH + 1):
                result = update_once(policy, combined)
                if not result.actor_changed or not result.critic_changed or not result.fly_output_unchanged:
                    raise AssertionError("PPO epoch violated heads-only/Fly CNS invariants")
                clip_fraction, approximate_kl = policy_shift_metrics(policy, combined)
                step = OptimizerStep(
                    through_battle=completed,
                    batch_battles=len(pending),
                    epoch=epoch,
                    transitions=len(combined.turns),
                    policy_entropy=result.entropy,
                    value_loss=result.value_loss,
                    policy_loss=result.policy_loss,
                    total_loss=result.total_loss,
                    advantage_mean=advantage_mean,
                    advantage_std=advantage_std,
                    clip_fraction=clip_fraction,
                    approximate_kl=approximate_kl,
                )
                all_steps.append(step)
                window_steps.append(step)
            window_advantages.append(advantages)
            updated_through = completed
            print(
                f"battles {completed - len(pending) + 1}-{completed}: "
                f"{EPOCHS_PER_BATCH} PPO steps, {len(combined.turns)} turns "
                f"({len(all_steps)}/{EXPECTED_OPTIMIZER_STEPS} steps total)",
                flush=True,
            )
            pending.clear()

        if completed % CHECKPOINT_INTERVAL == 0:
            if not window_steps:
                raise AssertionError("Checkpoint window had no PPO optimization steps")
            checkpoint = output_dir / f"battle_{completed:03d}.pt"
            if checkpoint.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint}")
            policy.actor_critic.save_weights(checkpoint)
            combined_advantages = np.concatenate(window_advantages)
            snapshot = CheckpointSnapshot(
                after_battle=completed,
                updated_through_battle=updated_through,
                pending_rollouts=len(pending),
                optimizer_steps_total=len(all_steps),
                optimizer_steps_in_window=len(window_steps),
                checkpoint=checkpoint,
                checkpoint_sha256=sha256(checkpoint),
                policy_entropy=_transition_weighted_mean(window_steps, "policy_entropy"),
                value_loss=_transition_weighted_mean(window_steps, "value_loss"),
                policy_loss=_transition_weighted_mean(window_steps, "policy_loss"),
                advantage_mean=float(np.mean(combined_advantages)),
                advantage_std=float(np.std(combined_advantages)),
                clip_fraction=_transition_weighted_mean(window_steps, "clip_fraction"),
                approximate_kl=_transition_weighted_mean(window_steps, "approximate_kl"),
            )
            checkpoints.append(snapshot)
            window_steps.clear()
            window_advantages.clear()
            print(
                f"saved {checkpoint.name}: {len(all_steps)} optimizer steps, "
                f"updated through battle {updated_through}, {len(pending)} rollouts pending",
                flush=True,
            )

    if len(all_steps) != EXPECTED_OPTIMIZER_STEPS:
        raise AssertionError(f"Expected {EXPECTED_OPTIMIZER_STEPS} optimizer steps")
    if [step.batch_battles for step in all_steps[::EPOCHS_PER_BATCH]] != [BATCH_BATTLES] * 12 + [4]:
        raise AssertionError("Expected 12 eight-battle batches and one four-battle batch")
    if [item.after_battle for item in checkpoints] != [20, 40, 60, 80, 100]:
        raise AssertionError("Expected checkpoints after battles 20, 40, 60, 80, 100")
    return tuple(all_steps), tuple(checkpoints)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-checkpoint", type=Path, default=DEFAULT_V1)
    parser.add_argument("--output-dir", type=Path, help="new directory for checkpoints and results")
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
        output_dir = Path(tempfile.mkdtemp(prefix="flycns_maxpower_epochs_", dir=artifact_root))
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

    steps, snapshots = train_multi_epoch(
        policy, make_training_environment, output_dir=output_dir, seed=args.seed
    )
    del policy  # Evaluations load frozen FlyCNSPlayer instances from disk.
    if sha256(v1) != v1_hash:
        raise AssertionError("v1 checkpoint changed during multi-epoch training")

    training_path = output_dir / "training.json"
    training_path.write_text(
        json.dumps(
            {
                "v1_checkpoint": str(v1),
                "v1_sha256": v1_hash,
                "training_battles": TRAINING_BATTLES,
                "batch_battles": BATCH_BATTLES,
                "epochs_per_batch": EPOCHS_PER_BATCH,
                "original_optimizer_steps_per_100_battles": ORIGINAL_OPTIMIZER_STEPS,
                "optimizer_steps": len(steps),
                "steps": [asdict(item) for item in steps],
                "snapshots": [asdict(item) for item in snapshots],
            },
            default=str,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

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
                "training_manifest": str(training_path),
                "v1_checkpoint": str(v1),
                "v1_sha256": v1_hash,
                "training_battles": TRAINING_BATTLES,
                "batch_battles": BATCH_BATTLES,
                "epochs_per_batch": EPOCHS_PER_BATCH,
                "original_optimizer_steps_per_100_battles": ORIGINAL_OPTIMIZER_STEPS,
                "optimizer_steps": len(steps),
                "evaluation_battles_per_checkpoint": EVALUATION_BATTLES,
                "metric_notes": (
                    "Training losses, entropy, clip fraction, and approximate KL are "
                    "transition-weighted over optimizer steps since the previous checkpoint. "
                    "Advantage mean/std use unique rollout transitions in that window. "
                    "Clip fraction and KL compare the post-step policy to rollout log probabilities."
                ),
                "benchmarks": [asdict(item) for item in benchmarks],
                "selected_checkpoint": None,
            },
            default=str,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\nFly v1 eight-epoch batched PPO vs MaxBasePowerPlayer (Fly perspective)")
    print(
        "after updated pending steps win% avg reward avg turns entropy value loss "
        "policy loss adv mean/std clip frac approx KL"
    )
    for item in benchmarks:
        snapshot = item.snapshot
        print(
            f"{snapshot.after_battle:>5} {snapshot.updated_through_battle:>7} "
            f"{snapshot.pending_rollouts:>7} {snapshot.optimizer_steps_total:>5} "
            f"{item.win_rate:>5.1%} {item.average_reward:>10.3f} "
            f"{item.average_turns:>9.2f} {snapshot.policy_entropy:>7.3f} "
            f"{snapshot.value_loss:>10.3f} {snapshot.policy_loss:>11.3f} "
            f"{snapshot.advantage_mean:>7.3f}/{snapshot.advantage_std:<7.3f} "
            f"{snapshot.clip_fraction:>8.3f} {snapshot.approximate_kl:>9.5f}"
        )
    print(
        f"optimizer steps: {len(steps)} "
        f"(original one-battle trainer: {ORIGINAL_OPTIMIZER_STEPS} per 100 battles)"
    )
    print(f"v1 SHA-256 unchanged: {v1_hash}")
    print(f"checkpoints and reports: {output_dir}")
    print("no checkpoint was selected or promoted")


if __name__ == "__main__":
    main()
