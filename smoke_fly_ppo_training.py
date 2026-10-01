"""Repeat the 10/10/10 Fly CNS PPO smoke experiment across three fixed seeds."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration

from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from flycns.actor_critic import FlyCNSActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_update import PPOUpdateResult, update_once
from smoke_fly_rollout import RolloutResult, run_one_battle


BATTLES_PER_PHASE = 10
FIXED_SEEDS = (7, 17, 27)


@dataclass(frozen=True)
class PhaseMetrics:
    battles: int
    wins: int
    losses: int
    ties: int
    average_reward: float
    average_battle_length: float


@dataclass(frozen=True)
class TrainingSmokeResult:
    before: PhaseMetrics
    training: PhaseMetrics
    after: PhaseMetrics
    updates: tuple[PPOUpdateResult, ...]
    checkpoint_path: Path


@dataclass(frozen=True)
class SeedResult:
    seed: int
    run: TrainingSmokeResult


@dataclass(frozen=True)
class MultiSeedResult:
    per_seed: tuple[SeedResult, ...]
    before: PhaseMetrics
    after: PhaseMetrics
    mean_win_improvement: float
    mean_reward_improvement: float
    mean_battle_length_change: float


def summarize(rollouts: list[RolloutResult]) -> PhaseMetrics:
    if len(rollouts) != BATTLES_PER_PHASE:
        raise ValueError(f"Expected exactly {BATTLES_PER_PHASE} battles")
    if any(rollout.illegal_action_attempted for rollout in rollouts):
        raise AssertionError("A rollout attempted an illegal action")
    return PhaseMetrics(
        battles=len(rollouts),
        wins=sum(rollout.outcome == "win" for rollout in rollouts),
        losses=sum(rollout.outcome == "loss" for rollout in rollouts),
        ties=sum(rollout.outcome == "tie" for rollout in rollouts),
        average_reward=sum(rollout.total_reward for rollout in rollouts) / len(rollouts),
        average_battle_length=sum(len(rollout.turns) for rollout in rollouts) / len(rollouts),
    )


def run_smoke(
    policy: FlyCNSPPOPolicy,
    environment_factory: Callable[[], Any],
    *,
    checkpoint_path: str | Path,
    seed: int = 7,
) -> TrainingSmokeResult:
    """Run exactly 30 battles, with one existing PPO update per training battle."""

    before: list[RolloutResult] = []
    training: list[RolloutResult] = []
    after: list[RolloutResult] = []
    updates: list[PPOUpdateResult] = []

    for index in range(BATTLES_PER_PHASE):
        before.append(
            run_one_battle(
                environment_factory(), policy, seed=seed + index, deterministic=True
            )
        )
    for index in range(BATTLES_PER_PHASE):
        rollout = run_one_battle(
            environment_factory(),
            policy,
            seed=seed + BATTLES_PER_PHASE + index,
        )
        training.append(rollout)
        updates.append(update_once(policy, rollout))
    for index in range(BATTLES_PER_PHASE):
        after.append(
            run_one_battle(
                environment_factory(),
                policy,
                seed=seed + 2 * BATTLES_PER_PHASE + index,
                deterministic=True,
            )
        )

    path = Path(checkpoint_path)
    policy.actor_critic.save_weights(path)
    reloaded = FlyCNSActorCritic.load_weights(path)
    if not all(
        torch.equal(saved, loaded)
        for saved, loaded in zip(
            policy.actor_critic.parameters(), reloaded.parameters(), strict=True
        )
    ):
        raise AssertionError("Temporary checkpoint did not reproduce trained weights")
    return TrainingSmokeResult(
        before=summarize(before),
        training=summarize(training),
        after=summarize(after),
        updates=tuple(updates),
        checkpoint_path=path,
    )


def aggregate_phase(metrics: tuple[PhaseMetrics, ...]) -> PhaseMetrics:
    """Combine equally sized seed runs without changing per-battle weighting."""

    battles = sum(item.battles for item in metrics)
    return PhaseMetrics(
        battles=battles,
        wins=sum(item.wins for item in metrics),
        losses=sum(item.losses for item in metrics),
        ties=sum(item.ties for item in metrics),
        average_reward=sum(item.average_reward * item.battles for item in metrics) / battles,
        average_battle_length=(
            sum(item.average_battle_length * item.battles for item in metrics) / battles
        ),
    )


def run_three_seed_smoke(
    environment_factory: Callable[[], Any], *, checkpoint_dir: str | Path
) -> MultiSeedResult:
    """Run the same 30-battle protocol with fresh heads for each fixed seed."""

    runs: list[SeedResult] = []
    for seed in FIXED_SEEDS:
        torch.manual_seed(seed)
        # FlyCNSPPOPolicy's default initialization is always seed 7. Pass fresh
        # heads explicitly so each experiment starts from its own fixed seed.
        policy = FlyCNSPPOPolicy(actor_critic=FlyCNSActorCritic())
        run = run_smoke(
            policy,
            environment_factory,
            checkpoint_path=Path(checkpoint_dir) / f"actor_critic_seed_{seed}.pt",
            seed=seed,
        )
        runs.append(SeedResult(seed=seed, run=run))

    before = aggregate_phase(tuple(item.run.before for item in runs))
    after = aggregate_phase(tuple(item.run.after for item in runs))
    return MultiSeedResult(
        per_seed=tuple(runs),
        before=before,
        after=after,
        mean_win_improvement=(after.wins - before.wins) / len(runs),
        mean_reward_improvement=after.average_reward - before.average_reward,
        mean_battle_length_change=after.average_battle_length - before.average_battle_length,
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

    temporary_dir = Path(tempfile.mkdtemp(prefix="flycns-ppo-smoke-"))
    result = run_three_seed_smoke(make_environment, checkpoint_dir=temporary_dir)
    print("seed   before W/L/T  after W/L/T  before reward  after reward  before turns  after turns")
    for item in result.per_seed:
        before, after = item.run.before, item.run.after
        print(
            f"{item.seed:>4}   {before.wins:>2}/{before.losses}/{before.ties:<6} "
            f"{after.wins:>2}/{after.losses}/{after.ties:<5} "
            f"{before.average_reward:>13.3f} {after.average_reward:>13.3f} "
            f"{before.average_battle_length:>13.1f} {after.average_battle_length:>12.1f}"
        )
    before, after = result.before, result.after
    print(
        f"all    {before.wins:>2}/{before.losses}/{before.ties:<6} "
        f"{after.wins:>2}/{after.losses}/{after.ties:<5} "
        f"{before.average_reward:>13.3f} {after.average_reward:>13.3f} "
        f"{before.average_battle_length:>13.1f} {after.average_battle_length:>12.1f}"
    )
    print(
        "mean improvement/seed: "
        f"{result.mean_win_improvement:+.2f} wins/10 "
        f"({10 * result.mean_win_improvement:+.1f} win-rate points), "
        f"{result.mean_reward_improvement:+.3f} reward, "
        f"{result.mean_battle_length_change:+.1f} turns"
    )
    print(f"PPO updates: {sum(len(item.run.updates) for item in result.per_seed)} (heads only)")
    print(f"checkpoints: {temporary_dir}/actor_critic_seed_*.pt")


if __name__ == "__main__":
    main()
