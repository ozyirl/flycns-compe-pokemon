"""Inspect frozen-v1 PPO rollout signals over 50 MaxBasePower battles."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration
from torch.distributions import Categorical

from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from experiment_fly_maxpower import DEFAULT_V1, sha256
from flycns.action_decoder import ACTION_COUNT
from flycns.actor_critic import FlyCNSActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_update import advantages_and_returns
from smoke_fly_rollout import RolloutResult, run_one_battle


DIAGNOSTIC_BATTLES = 50


@dataclass(frozen=True)
class TurnDiagnostic:
    battle: int
    turn: int
    outcome: str
    selected_action: int
    reward: float
    critic_value: float
    computed_return: float
    ppo_advantage: float
    critic_prediction_error: float
    policy_entropy: float
    selected_action_probability: float


@dataclass(frozen=True)
class BattleDiagnostic:
    battle: int
    outcome: str
    total_reward: float
    turns: tuple[TurnDiagnostic, ...]


@dataclass(frozen=True)
class OutcomeSummary:
    battles: int
    turns: int
    average_total_reward: float
    average_absolute_advantage: float
    average_positive_advantage: float | None
    positive_advantage_turns: int
    average_negative_advantage: float | None
    negative_advantage_turns: int
    average_absolute_critic_prediction_error: float
    average_policy_entropy: float
    action_frequency: dict[str, dict[str, float | int]]


def inspect_rollout(
    policy: FlyCNSPPOPolicy, rollout: RolloutResult, *, battle_index: int
) -> BattleDiagnostic:
    """Apply the existing terminal-aware GAE, then inspect the frozen action law."""

    if rollout.illegal_action_attempted or not rollout.actor_critic_weights_unchanged:
        raise AssertionError("Expected one legal, read-only Fly rollout")
    advantages, returns = advantages_and_returns(rollout)
    rows: list[TurnDiagnostic] = []
    for turn_index, (turn, advantage, computed_return) in enumerate(
        zip(rollout.turns, advantages, returns, strict=True), start=1
    ):
        # The policy is unchanged. Re-evaluate logits only to recover entropy;
        # the sampled action and log probability remain those recorded in the rollout.
        decision = policy.predict(turn.observation, turn.action_mask, deterministic=True)
        with torch.inference_mode():
            distribution = Categorical(logits=torch.tensor(decision.policy_logits))
            selected_log_probability = distribution.log_prob(torch.tensor(turn.selected_action))
            entropy = distribution.entropy()
        if not np.isclose(float(selected_log_probability), turn.log_probability, atol=1e-5):
            raise AssertionError("Frozen policy log probability changed after rollout")
        selected_probability = float(np.exp(turn.log_probability))
        if not np.isfinite(selected_probability) or not 0.0 <= selected_probability <= 1.0:
            raise ValueError("Selected action probability is not finite and valid")
        rows.append(
            TurnDiagnostic(
                battle=battle_index,
                turn=turn_index,
                outcome=rollout.outcome,
                selected_action=turn.selected_action,
                reward=turn.reward,
                critic_value=turn.state_value,
                computed_return=float(computed_return),
                ppo_advantage=float(advantage),
                critic_prediction_error=float(turn.state_value - computed_return),
                policy_entropy=float(entropy),
                selected_action_probability=selected_probability,
            )
        )
    return BattleDiagnostic(
        battle=battle_index,
        outcome=rollout.outcome,
        total_reward=rollout.total_reward,
        turns=tuple(rows),
    )


def summarize_outcome(battles: tuple[BattleDiagnostic, ...]) -> OutcomeSummary:
    """Average battle rewards by battle and signal metrics by turn."""

    if not battles:
        raise ValueError("Cannot summarize zero battles")
    turns = tuple(turn for battle in battles for turn in battle.turns)
    if not turns:
        raise ValueError("Cannot summarize battles without turns")
    advantages = np.asarray([turn.ppo_advantage for turn in turns], dtype=np.float64)
    errors = np.asarray([turn.critic_prediction_error for turn in turns], dtype=np.float64)
    entropies = np.asarray([turn.policy_entropy for turn in turns], dtype=np.float64)
    positive = advantages[advantages > 0]
    negative = advantages[advantages < 0]
    frequencies = Counter(turn.selected_action for turn in turns)
    if any(not 0 <= action < ACTION_COUNT for action in frequencies):
        raise ValueError("Diagnostic contains an out-of-range action")
    return OutcomeSummary(
        battles=len(battles),
        turns=len(turns),
        average_total_reward=float(np.mean([battle.total_reward for battle in battles])),
        average_absolute_advantage=float(np.mean(np.abs(advantages))),
        average_positive_advantage=float(np.mean(positive)) if len(positive) else None,
        positive_advantage_turns=len(positive),
        average_negative_advantage=float(np.mean(negative)) if len(negative) else None,
        negative_advantage_turns=len(negative),
        average_absolute_critic_prediction_error=float(np.mean(np.abs(errors))),
        average_policy_entropy=float(np.mean(entropies)),
        action_frequency={
            str(action): {
                "count": frequencies[action],
                "fraction": frequencies[action] / len(turns),
            }
            for action in range(ACTION_COUNT)
        },
    )


def run_diagnostic(
    policy: FlyCNSPPOPolicy,
    environment_factory: Callable[[], Any],
    *,
    battles: int = DIAGNOSTIC_BATTLES,
    seed: int = 7,
) -> tuple[BattleDiagnostic, ...]:
    """Collect stochastic training-style rollouts without calling PPO update."""

    if battles < 1:
        raise ValueError("battles must be positive")
    original_weights = [parameter.detach().clone() for parameter in policy.actor_critic.parameters()]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    results: list[BattleDiagnostic] = []
    for index in range(battles):
        rollout = run_one_battle(environment_factory(), policy, seed=seed + index)
        results.append(inspect_rollout(policy, rollout, battle_index=index + 1))
        if (index + 1) % 10 == 0 or index + 1 == battles:
            print(f"diagnostic: {index + 1}/{battles} battles", flush=True)
    if any(
        not torch.equal(parameter, original)
        for parameter, original in zip(
            policy.actor_critic.parameters(), original_weights, strict=True
        )
    ):
        raise AssertionError("Fly actor-critic weights changed during read-only diagnosis")
    return tuple(results)


def _action_distribution(summary: OutcomeSummary) -> str:
    active = sorted(
        (
            (int(action), values)
            for action, values in summary.action_frequency.items()
            if values["count"]
        ),
        key=lambda item: (-item[1]["count"], item[0]),
    )
    return ", ".join(
        f"{action}:{values['count']} ({values['fraction']:.1%})"
        for action, values in active
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_V1)
    parser.add_argument("--output-dir", type=Path, help="new directory for per-turn data and summary")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--server-url",
        default=LocalhostServerConfiguration.websocket_url,
        help="WebSocket URL of the already-running local Showdown server",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Fly checkpoint not found: {checkpoint}")
    if args.output_dir is not None and args.output_dir.exists():
        parser.error(f"output directory must not already exist: {args.output_dir}")
    checkpoint_hash = sha256(checkpoint)
    server = ServerConfiguration(
        args.server_url, LocalhostServerConfiguration.authentication_url
    )
    actor_critic = FlyCNSActorCritic.load_weights(checkpoint)
    actor_critic.eval()
    actor_critic.requires_grad_(False)
    policy = FlyCNSPPOPolicy(actor_critic=actor_critic)

    def make_environment() -> SingleAgentWrapper:
        return SingleAgentWrapper(
            ShowdownEnv(
                battle_format=DEFAULT_BATTLE_FORMAT,
                server_configuration=server,
            ),
            make_opponent("max-power", battle_format=DEFAULT_BATTLE_FORMAT),
        )

    battles = run_diagnostic(policy, make_environment, seed=args.seed)
    if sha256(checkpoint) != checkpoint_hash:
        raise AssertionError("v1 checkpoint changed during read-only diagnosis")
    summaries: dict[str, OutcomeSummary | None] = {}
    for outcome in ("win", "loss", "tie"):
        selected = tuple(battle for battle in battles if battle.outcome == outcome)
        if selected or outcome != "tie":
            summaries[outcome] = summarize_outcome(selected) if selected else None
    artifact_root = Path(__file__).resolve().parent / "models"
    if args.output_dir is None:
        artifact_root.mkdir(exist_ok=True)
        output_dir = Path(
            tempfile.mkdtemp(prefix="flycns_maxpower_advantages_", dir=artifact_root)
        )
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
    turn_path = output_dir / "turns.jsonl"
    turn_path.write_text(
        "\n".join(json.dumps(asdict(turn)) for battle in battles for turn in battle.turns) + "\n",
        encoding="utf-8",
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "opponent": "MaxBasePowerPlayer",
                "battles": len(battles),
                "turns": sum(len(battle.turns) for battle in battles),
                "seed": args.seed,
                "rollout_action_selection": "stochastic",
                "ppo_advantages": "existing terminal-aware GAE defaults",
                "critic_prediction_error": "value minus computed return; aggregate reports mean absolute error",
                "outcomes": {
                    outcome: asdict(summary) if summary is not None else None
                    for outcome, summary in summaries.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\nFly v1 vs MaxBasePowerPlayer (stochastic, read-only)")
    print("outcome battles turns avg reward |adv| +adv(n) -adv(n) critic MAE entropy")
    for outcome, summary in summaries.items():
        if summary is None:
            print(f"{outcome:<7} 0")
            continue
        positive = (
            f"{summary.average_positive_advantage:.3f}({summary.positive_advantage_turns})"
            if summary.average_positive_advantage is not None else "n/a(0)"
        )
        negative = (
            f"{summary.average_negative_advantage:.3f}({summary.negative_advantage_turns})"
            if summary.average_negative_advantage is not None else "n/a(0)"
        )
        print(
            f"{outcome:<7} {summary.battles:>3} {summary.turns:>5} "
            f"{summary.average_total_reward:>10.3f} "
            f"{summary.average_absolute_advantage:>5.3f} "
            f"{positive:>12} {negative:>12} "
            f"{summary.average_absolute_critic_prediction_error:>10.3f} "
            f"{summary.average_policy_entropy:>7.3f}"
        )
        print(f"  action frequency (index: count, share): {_action_distribution(summary)}")
    print(f"per-turn records: {turn_path}")
    print(f"summary: {summary_path}")
    print("checkpoint and actor-critic weights unchanged; no PPO update was run")


if __name__ == "__main__":
    main()
