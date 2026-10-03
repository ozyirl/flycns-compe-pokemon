"""Run a fixed three-seed, 30/100/30 Fly CNS PPO experiment against RandomPlayer."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration

from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from flycns.actor_critic import FlyCNSActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_training import FIXED_SEEDS, PhaseMetrics, aggregate_phase
from smoke_fly_ppo_update import PPOUpdateResult, update_once
from smoke_fly_rollout import RolloutResult, run_one_battle


EVALUATION_BATTLES = 30
TRAINING_BATTLES = 100
LOG_INTERVAL = 10
CHECKPOINT_INTERVAL = 25


@dataclass(frozen=True)
class CurvePoint:
    training_battles: int
    win_rate: float
    average_reward: float
    average_battle_length: float
    policy_entropy: float
    policy_loss: float
    value_loss: float


@dataclass(frozen=True)
class SeedExperiment:
    seed: int
    before: PhaseMetrics
    after: PhaseMetrics
    learning_curve: tuple[CurvePoint, ...]
    checkpoints: tuple[Path, ...]


@dataclass(frozen=True)
class MediumExperiment:
    per_seed: tuple[SeedExperiment, ...]
    before: PhaseMetrics
    after: PhaseMetrics
    aggregate_curve: tuple[CurvePoint, ...]


def summarize_rollouts(rollouts: list[RolloutResult]) -> PhaseMetrics:
    if not rollouts:
        raise ValueError("Cannot summarize zero battles")
    if any(rollout.illegal_action_attempted for rollout in rollouts):
        raise AssertionError("A rollout attempted an illegal action")
    count = len(rollouts)
    return PhaseMetrics(
        battles=count,
        wins=sum(rollout.outcome == "win" for rollout in rollouts),
        losses=sum(rollout.outcome == "loss" for rollout in rollouts),
        ties=sum(rollout.outcome == "tie" for rollout in rollouts),
        average_reward=sum(rollout.total_reward for rollout in rollouts) / count,
        average_battle_length=sum(len(rollout.turns) for rollout in rollouts) / count,
    )


def curve_point(
    training_battles: int,
    rollouts: list[RolloutResult],
    updates: list[PPOUpdateResult],
) -> CurvePoint:
    if len(rollouts) != LOG_INTERVAL or len(updates) != LOG_INTERVAL:
        raise ValueError(f"Expected {LOG_INTERVAL} rollouts and PPO updates per curve point")
    metrics = summarize_rollouts(rollouts)
    return CurvePoint(
        training_battles=training_battles,
        win_rate=metrics.wins / metrics.battles,
        average_reward=metrics.average_reward,
        average_battle_length=metrics.average_battle_length,
        policy_entropy=sum(update.entropy for update in updates) / len(updates),
        policy_loss=sum(update.policy_loss for update in updates) / len(updates),
        value_loss=sum(update.value_loss for update in updates) / len(updates),
    )


def run_seed(
    policy: FlyCNSPPOPolicy,
    environment_factory: Callable[[], Any],
    *,
    seed: int,
    checkpoint_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> SeedExperiment:
    """Run 30 deterministic, 100 sampled/updating, then 30 deterministic battles."""

    before = [
        run_one_battle(environment_factory(), policy, seed=seed + index, deterministic=True)
        for index in range(EVALUATION_BATTLES)
    ]
    if progress is not None:
        progress(f"seed {seed}: before evaluation complete ({EVALUATION_BATTLES} battles)")

    learning_curve: list[CurvePoint] = []
    checkpoints: list[Path] = []
    block_rollouts: list[RolloutResult] = []
    block_updates: list[PPOUpdateResult] = []
    for index in range(TRAINING_BATTLES):
        rollout = run_one_battle(
            environment_factory(), policy, seed=seed + EVALUATION_BATTLES + index
        )
        update = update_once(policy, rollout)
        block_rollouts.append(rollout)
        block_updates.append(update)
        completed = index + 1

        if completed % LOG_INTERVAL == 0:
            point = curve_point(completed, block_rollouts, block_updates)
            learning_curve.append(point)
            block_rollouts.clear()
            block_updates.clear()
            if progress is not None:
                progress(
                    f"seed {seed}: trained {completed}/{TRAINING_BATTLES}, "
                    f"last-{LOG_INTERVAL} win rate {point.win_rate:.0%}, "
                    f"reward {point.average_reward:.3f}"
                )
        if completed % CHECKPOINT_INTERVAL == 0:
            path = checkpoint_dir / f"seed_{seed}_battle_{completed:03d}.pt"
            policy.actor_critic.save_weights(path)
            checkpoints.append(path)

    after = [
        run_one_battle(
            environment_factory(),
            policy,
            seed=seed + EVALUATION_BATTLES + TRAINING_BATTLES + index,
            deterministic=True,
        )
        for index in range(EVALUATION_BATTLES)
    ]
    if progress is not None:
        progress(f"seed {seed}: after evaluation complete ({EVALUATION_BATTLES} battles)")
    return SeedExperiment(
        seed=seed,
        before=summarize_rollouts(before),
        after=summarize_rollouts(after),
        learning_curve=tuple(learning_curve),
        checkpoints=tuple(checkpoints),
    )


def aggregate_learning_curve(runs: tuple[SeedExperiment, ...]) -> tuple[CurvePoint, ...]:
    points: list[CurvePoint] = []
    for index in range(TRAINING_BATTLES // LOG_INTERVAL):
        block = [run.learning_curve[index] for run in runs]
        points.append(
            CurvePoint(
                training_battles=block[0].training_battles,
                win_rate=sum(point.win_rate for point in block) / len(block),
                average_reward=sum(point.average_reward for point in block) / len(block),
                average_battle_length=(
                    sum(point.average_battle_length for point in block) / len(block)
                ),
                policy_entropy=sum(point.policy_entropy for point in block) / len(block),
                policy_loss=sum(point.policy_loss for point in block) / len(block),
                value_loss=sum(point.value_loss for point in block) / len(block),
            )
        )
    return tuple(points)


def run_experiment(
    environment_factory: Callable[[], Any],
    *,
    output_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> MediumExperiment:
    """Initialize independent normalized heads for the three existing fixed seeds."""

    runs: list[SeedExperiment] = []
    for seed in FIXED_SEEDS:
        torch.manual_seed(seed)
        policy = FlyCNSPPOPolicy(actor_critic=FlyCNSActorCritic())
        runs.append(
            run_seed(
                policy,
                environment_factory,
                seed=seed,
                checkpoint_dir=output_dir,
                progress=progress,
            )
        )
    per_seed = tuple(runs)
    return MediumExperiment(
        per_seed=per_seed,
        before=aggregate_phase(tuple(run.before for run in per_seed)),
        after=aggregate_phase(tuple(run.after for run in per_seed)),
        aggregate_curve=aggregate_learning_curve(per_seed),
    )


def print_curve(label: str, curve: tuple[CurvePoint, ...]) -> None:
    print(f"\n{label} learning curve (each row summarizes the preceding 10 battles)")
    print("battles  win rate  avg reward  avg turns  entropy  policy loss  value loss")
    for point in curve:
        print(
            f"{point.training_battles:>7}  {point.win_rate:>7.1%}  "
            f"{point.average_reward:>10.3f}  {point.average_battle_length:>9.1f}  "
            f"{point.policy_entropy:>7.3f}  {point.policy_loss:>11.3f}  "
            f"{point.value_loss:>10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-url",
        default=LocalhostServerConfiguration.websocket_url,
        help="WebSocket URL of the already-running local Showdown server",
    )
    args = parser.parse_args()
    server = ServerConfiguration(
        args.server_url, LocalhostServerConfiguration.authentication_url
    )

    def make_environment() -> SingleAgentWrapper:
        return SingleAgentWrapper(
            ShowdownEnv(
                battle_format=DEFAULT_BATTLE_FORMAT,
                server_configuration=server,
            ),
            make_opponent("random", battle_format=DEFAULT_BATTLE_FORMAT),
        )

    artifact_root = Path(__file__).resolve().parent / "models"
    artifact_root.mkdir(exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="flycns_ppo_medium_", dir=artifact_root))
    result = run_experiment(
        make_environment,
        output_dir=output_dir,
        progress=lambda message: print(message, flush=True),
    )
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(asdict(result), default=str, indent=2), encoding="utf-8")

    print("\nseed  before W/L/T  after W/L/T  before reward  after reward  before turns  after turns")
    for run in result.per_seed:
        before, after = run.before, run.after
        print(
            f"{run.seed:>4}  {before.wins:>2}/{before.losses}/{before.ties:<6} "
            f"{after.wins:>2}/{after.losses}/{after.ties:<5} "
            f"{before.average_reward:>13.3f} {after.average_reward:>13.3f} "
            f"{before.average_battle_length:>13.1f} {after.average_battle_length:>12.1f}"
        )
    before, after = result.before, result.after
    print(
        f" all  {before.wins:>2}/{before.losses}/{before.ties:<6} "
        f"{after.wins:>2}/{after.losses}/{after.ties:<5} "
        f"{before.average_reward:>13.3f} {after.average_reward:>13.3f} "
        f"{before.average_battle_length:>13.1f} {after.average_battle_length:>12.1f}"
    )
    for run in result.per_seed:
        print_curve(f"seed {run.seed}", run.learning_curve)
    print_curve("aggregate", result.aggregate_curve)
    print(f"\nresults and 12 head-only checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
