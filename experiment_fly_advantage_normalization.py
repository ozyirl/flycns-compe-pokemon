"""Compare raw and normalized-policy-advantage Fly PPO from unchanged v1."""

from __future__ import annotations

import argparse
import json
import secrets
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from poke_env.environment import SingleAgentWrapper
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration

from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from experiment_fly_maxpower import DEFAULT_V1, TRAINING_BATTLES, sha256
from experiment_fly_maxpower_batched import BATCH_BATTLES, CHECKPOINT_INTERVAL
from experiment_fly_maxpower_epochs import (
    EPOCHS_PER_BATCH,
    EXPECTED_OPTIMIZER_STEPS,
    CheckpointSnapshot,
    OptimizerStep,
    train_multi_epoch,
)
from experiment_fly_maxpower_sweep import EVALUATION_BATTLES, evaluate_checkpoint
from flycns.actor_critic import FlyCNSActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_update import ADVANTAGE_NORMALIZATION_EPSILON


DEFAULT_SEEDS = (7, 17, 27)
MODES = ("unnormalized", "normalized")


@dataclass(frozen=True)
class CheckpointResult:
    mode: str
    seed: int
    after_battle: int
    updated_through_battle: int
    pending_rollouts: int
    optimizer_steps: int
    checkpoint: Path
    checkpoint_sha256: str
    wins: int
    losses: int
    ties: int
    win_rate: float
    average_reward: float
    average_turns: float
    policy_entropy: float
    policy_loss: float
    value_loss: float
    raw_advantage_mean: float
    raw_advantage_std: float
    normalized_advantage_mean: float
    normalized_advantage_std: float
    clip_fraction: float
    approximate_kl: float
    fly_illegal_actions: int
    opposing_illegal_actions: int


def record_checkpoint(
    mode: str, seed: int, snapshot: CheckpointSnapshot, result: object
) -> CheckpointResult:
    return CheckpointResult(
        mode=mode,
        seed=seed,
        after_battle=snapshot.after_battle,
        updated_through_battle=snapshot.updated_through_battle,
        pending_rollouts=snapshot.pending_rollouts,
        optimizer_steps=snapshot.optimizer_steps_total,
        checkpoint=snapshot.checkpoint,
        checkpoint_sha256=snapshot.checkpoint_sha256,
        wins=result.wins,
        losses=result.losses,
        ties=result.ties,
        win_rate=result.win_rate,
        average_reward=result.average_reward,
        average_turns=result.average_battle_length,
        policy_entropy=snapshot.policy_entropy,
        policy_loss=snapshot.policy_loss,
        value_loss=snapshot.value_loss,
        raw_advantage_mean=snapshot.advantage_mean,
        raw_advantage_std=snapshot.advantage_std,
        normalized_advantage_mean=snapshot.normalized_advantage_mean,
        normalized_advantage_std=snapshot.normalized_advantage_std,
        clip_fraction=snapshot.clip_fraction,
        approximate_kl=snapshot.approximate_kl,
        fly_illegal_actions=result.fly_illegal_actions,
        opposing_illegal_actions=result.opposing_illegal_actions,
    )


def aggregate(results: list[CheckpointResult]) -> dict[str, object]:
    """Aggregate a same-mode, same-checkpoint result across fixed seeds."""
    if not results or len({row.mode for row in results}) != 1 or len({row.after_battle for row in results}) != 1:
        raise ValueError("Aggregate requires one mode and checkpoint across seeds")
    fields = (
        "average_reward", "average_turns", "policy_entropy", "policy_loss", "value_loss",
        "raw_advantage_mean", "raw_advantage_std", "normalized_advantage_mean",
        "normalized_advantage_std", "clip_fraction", "approximate_kl",
    )
    total_battles = EVALUATION_BATTLES * len(results)
    return {
        "mode": results[0].mode,
        "after_battle": results[0].after_battle,
        "seeds": [row.seed for row in results],
        "evaluation_battles": total_battles,
        "wins": sum(row.wins for row in results),
        "losses": sum(row.losses for row in results),
        "ties": sum(row.ties for row in results),
        "win_rate": sum(row.wins for row in results) / total_battles,
        **{name: float(np.mean([getattr(row, name) for row in results])) for name in fields},
        "metric_note": "Training statistics are the arithmetic mean of seed-level checkpoint windows",
    }


def training_environment(server: ServerConfiguration) -> SingleAgentWrapper:
    """Use a unique temporary account pair for each local Showdown battle."""
    nonce = secrets.token_hex(5)
    return SingleAgentWrapper(
        ShowdownEnv(
            battle_format=DEFAULT_BATTLE_FORMAT,
            server_configuration=server,
            account_configuration1=AccountConfiguration(f"fan{nonce}a", None),
            account_configuration2=AccountConfiguration(f"fan{nonce}b", None),
        ),
        make_opponent("max-power", battle_format=DEFAULT_BATTLE_FORMAT),
    )


def run_mode_seed(
    mode: str,
    seed: int,
    *,
    v1: Path,
    v1_hash: str,
    server: ServerConfiguration,
    output_dir: Path,
) -> tuple[tuple[OptimizerStep, ...], tuple[CheckpointResult, ...]]:
    """Start a fresh actor/critic from v1 and run one fixed-count experiment."""
    if mode not in MODES:
        raise ValueError(f"Unknown advantage mode: {mode}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    policy = FlyCNSPPOPolicy(actor_critic=FlyCNSActorCritic.load_weights(v1))

    steps, snapshots = train_multi_epoch(
        policy,
        lambda: training_environment(server),
        output_dir=output_dir,
        seed=seed,
        normalize_advantages=(mode == "normalized"),
    )
    del policy  # Checkpoint evaluation uses fresh inference-only FlyCNSPlayer instances.
    if len(steps) != EXPECTED_OPTIMIZER_STEPS or sha256(v1) != v1_hash:
        raise AssertionError("Optimizer-step count or unchanged-v1 invariant failed")
    (output_dir / "training.json").write_text(
        json.dumps({
            "mode": mode, "seed": seed, "v1_checkpoint": str(v1), "v1_sha256": v1_hash,
            "training_battles": TRAINING_BATTLES, "batch_battles": BATCH_BATTLES,
            "epochs_per_batch": EPOCHS_PER_BATCH, "optimizer_steps": len(steps),
            "steps": [asdict(row) for row in steps], "snapshots": [asdict(row) for row in snapshots],
        }, default=str, indent=2) + "\n",
        encoding="utf-8",
    )
    evaluations: list[CheckpointResult] = []
    for snapshot in snapshots:
        print(f"evaluating {mode} seed {seed} battle {snapshot.after_battle}", flush=True)
        result = evaluate_checkpoint(snapshot.checkpoint, server, seed=seed)
        if sha256(v1) != v1_hash or sha256(snapshot.checkpoint) != snapshot.checkpoint_sha256:
            raise AssertionError("v1 or candidate checkpoint changed during read-only evaluation")
        evaluations.append(record_checkpoint(mode, seed, snapshot, result))
    (output_dir / "results.json").write_text(
        json.dumps([asdict(row) for row in evaluations], default=str, indent=2) + "\n",
        encoding="utf-8",
    )
    return steps, tuple(evaluations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-checkpoint", type=Path, default=DEFAULT_V1)
    parser.add_argument("--output-dir", type=Path, help="new directory for all seed/mode checkpoints and reports")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--server-url", default=LocalhostServerConfiguration.websocket_url)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    v1 = args.v1_checkpoint.resolve()
    if not v1.is_file():
        parser.error(f"v1 checkpoint not found: {v1}")
    if args.output_dir is not None and args.output_dir.exists():
        parser.error(f"output directory must not already exist: {args.output_dir}")
    v1_hash = sha256(v1)
    artifact_root = Path(__file__).resolve().parent / "models"
    if args.output_dir is None:
        artifact_root.mkdir(exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="flycns_adv_norm_", dir=artifact_root))
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
    server = ServerConfiguration(args.server_url, LocalhostServerConfiguration.authentication_url)
    all_results: list[CheckpointResult] = []
    for seed in args.seeds:
        for mode in MODES:
            print(f"\n{mode} seed {seed}: 100 training battles from v1", flush=True)
            _, results = run_mode_seed(
                mode, seed, v1=v1, v1_hash=v1_hash, server=server,
                output_dir=output_dir / f"seed_{seed}_{mode}",
            )
            all_results.extend(results)
    if sha256(v1) != v1_hash:
        raise AssertionError("v1 changed during comparison")
    aggregate_rows = [
        aggregate([row for row in all_results if row.mode == mode and row.after_battle == battle])
        for mode in MODES
        for battle in range(CHECKPOINT_INTERVAL, TRAINING_BATTLES + 1, CHECKPOINT_INTERVAL)
    ]
    report = {
        "v1_checkpoint": str(v1), "v1_sha256": v1_hash,
        "seeds": args.seeds, "modes": MODES,
        "training_battles_per_seed_mode": TRAINING_BATTLES,
        "evaluation_battles_per_checkpoint": EVALUATION_BATTLES,
        "batch_battles": BATCH_BATTLES, "epochs_per_batch": EPOCHS_PER_BATCH,
        "optimizer_steps_per_seed_mode": EXPECTED_OPTIMIZER_STEPS,
        "advantage_normalization_epsilon": ADVANTAGE_NORMALIZATION_EPSILON,
        "comparison_note": (
            "Both modes use the same seed values and battle counts, but Showdown random teams "
            "are not guaranteed to be identical paired battles. No checkpoint is selected."
        ),
        "per_seed": [asdict(row) for row in all_results],
        "aggregate": aggregate_rows,
        "selected_checkpoint": None,
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, default=str, indent=2) + "\n", encoding="utf-8")
    print("\nFly v1 PPO advantage normalization vs MaxBasePowerPlayer")
    print("mode seed after wins/50 win% reward entropy policy_loss value_loss raw_adv(mean/std) norm_adv(mean/std) clip KL")
    for row in all_results:
        print(
            f"{row.mode:<12} {row.seed:>3} {row.after_battle:>3} "
            f"{row.wins:>2}/50 {row.win_rate:>5.1%} {row.average_reward:>+7.3f} "
            f"{row.policy_entropy:>6.3f} {row.policy_loss:>+9.3f} {row.value_loss:>9.3f} "
            f"{row.raw_advantage_mean:>+7.3f}/{row.raw_advantage_std:<6.3f} "
            f"{row.normalized_advantage_mean:>+6.3f}/{row.normalized_advantage_std:<5.3f} "
            f"{row.clip_fraction:>5.3f} {row.approximate_kl:>7.5f}"
        )
    print("aggregate across seeds:")
    for row in aggregate_rows:
        print(
            f"{row['mode']:<12} after {row['after_battle']:>3} "
            f"{row['wins']}/{row['evaluation_battles']} ({row['win_rate']:.1%}) "
            f"reward {row['average_reward']:+.3f} entropy {row['policy_entropy']:.3f} "
            f"policy/value {row['policy_loss']:+.3f}/{row['value_loss']:.3f} "
            f"raw adv {row['raw_advantage_mean']:+.3f}/{row['raw_advantage_std']:.3f} "
            f"normalized adv {row['normalized_advantage_mean']:+.3f}/{row['normalized_advantage_std']:.3f} "
            f"clip/KL {row['clip_fraction']:.3f}/{row['approximate_kl']:.5f}"
        )
    print(f"\nv1 unchanged; no checkpoint promoted. Reports: {output_dir}")


if __name__ == "__main__":
    main()
