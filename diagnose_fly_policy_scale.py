"""Inspect untrained Fly CNS spike and policy scales over real Showdown turns."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration
from torch.distributions import Categorical

from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from flycns.action_decoder import ACTION_COUNT
from flycns.ppo_policy import FlyCNSPPOPolicy
from flycns.trainable_decoder import DESCENDING_ACTIVITY_SIZE
from smoke_fly_rollout import TurnRecord, run_one_battle


DEFAULT_BATTLES = 5
METRICS = (
    ("spike minimum", "spike_minimum"),
    ("spike mean", "spike_mean"),
    ("spike maximum", "spike_maximum"),
    ("spike L2 norm", "spike_l2_norm"),
    ("legal actions", "legal_actions"),
    ("legal logit minimum", "legal_logit_minimum"),
    ("legal logit maximum", "legal_logit_maximum"),
    ("policy entropy", "policy_entropy"),
    ("highest legal probability", "highest_legal_probability"),
)


@dataclass(frozen=True)
class TurnScale:
    spike_minimum: float
    spike_mean: float
    spike_maximum: float
    spike_l2_norm: float
    legal_actions: int
    legal_logit_minimum: float
    legal_logit_maximum: float
    policy_entropy: float
    highest_legal_probability: float


@dataclass(frozen=True)
class ScaleSummary:
    mean: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class PolicyScaleDiagnostic:
    battles: int
    turns: tuple[TurnScale, ...]
    summaries: dict[str, ScaleSummary]
    weights_unchanged: bool


def inspect_turn(policy: FlyCNSPPOPolicy, turn: TurnRecord) -> TurnScale:
    """Measure one recorded turn without building gradients or choosing an action."""

    activity = policy._descending_activity(turn.observation)
    if activity.shape != (DESCENDING_ACTIVITY_SIZE,) or not np.isfinite(activity).all():
        raise ValueError("Expected 512 finite descending spike counts")
    legal = policy._validated_masks(turn.action_mask, (ACTION_COUNT,))
    parameter = next(policy.actor_critic.parameters())
    features = torch.as_tensor(activity, device=parameter.device, dtype=parameter.dtype)
    legal_tensor = torch.as_tensor(legal, device=parameter.device, dtype=torch.bool)
    with torch.inference_mode():
        logits, _ = policy.actor_critic(features)
        legal_logits = logits[legal_tensor]
        distribution = Categorical(
            logits=logits.masked_fill(~legal_tensor, float("-inf"))
        )
        entropy = distribution.entropy()
        highest_probability = distribution.probs[legal_tensor].max()

    return TurnScale(
        spike_minimum=float(np.min(activity)),
        spike_mean=float(np.mean(activity)),
        spike_maximum=float(np.max(activity)),
        spike_l2_norm=float(np.linalg.norm(activity)),
        legal_actions=int(np.count_nonzero(legal)),
        legal_logit_minimum=float(legal_logits.min().item()),
        legal_logit_maximum=float(legal_logits.max().item()),
        policy_entropy=float(entropy.item()),
        highest_legal_probability=float(highest_probability.item()),
    )


def summarize_turns(turns: tuple[TurnScale, ...]) -> dict[str, ScaleSummary]:
    if not turns:
        raise ValueError("Cannot summarize zero battle turns")
    summaries: dict[str, ScaleSummary] = {}
    for _, field in METRICS:
        values = np.asarray([getattr(turn, field) for turn in turns], dtype=np.float64)
        summaries[field] = ScaleSummary(
            mean=float(values.mean()),
            minimum=float(values.min()),
            maximum=float(values.max()),
        )
    return summaries


def run_diagnostic(
    policy: FlyCNSPPOPolicy,
    environment_factory: Callable[[], Any],
    *,
    battles: int = DEFAULT_BATTLES,
    seed: int = 7,
) -> PolicyScaleDiagnostic:
    """Collect real turns against RandomPlayer with fixed untrained weights."""

    if battles <= 0:
        raise ValueError("battles must be positive")
    original_weights = [
        parameter.detach().clone() for parameter in policy.actor_critic.parameters()
    ]
    turn_scales: list[TurnScale] = []
    for index in range(battles):
        rollout = run_one_battle(environment_factory(), policy, seed=seed + index)
        turn_scales.extend(inspect_turn(policy, turn) for turn in rollout.turns)
    weights_unchanged = all(
        torch.equal(parameter, original)
        for parameter, original in zip(
            policy.actor_critic.parameters(), original_weights, strict=True
        )
    )
    if not weights_unchanged:
        raise AssertionError("Actor-critic weights changed during read-only diagnosis")
    turns = tuple(turn_scales)
    return PolicyScaleDiagnostic(
        battles=battles,
        turns=turns,
        summaries=summarize_turns(turns),
        weights_unchanged=weights_unchanged,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-url",
        default=LocalhostServerConfiguration.websocket_url,
        help="WebSocket URL of the already-running local Showdown server",
    )
    parser.add_argument("--battles", type=int, default=DEFAULT_BATTLES)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    policy = FlyCNSPPOPolicy()
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

    result = run_diagnostic(
        policy, make_environment, battles=args.battles, seed=args.seed
    )
    print(f"Untrained Fly CNS policy scale: {result.battles} battles, {len(result.turns)} turns")
    print("metric                         mean          min          max")
    for label, field in METRICS:
        summary = result.summaries[field]
        print(
            f"{label:<27} {summary.mean:>11.4f} "
            f"{summary.minimum:>12.4f} {summary.maximum:>12.4f}"
        )
    multi_legal = tuple(turn for turn in result.turns if turn.legal_actions > 1)
    print(f"turns with 2+ legal actions: {len(multi_legal)}")
    if multi_legal:
        multi_summary = summarize_turns(multi_legal)
        probability = multi_summary["highest_legal_probability"]
        entropy = multi_summary["policy_entropy"]
        high_confidence = sum(
            turn.highest_legal_probability >= 0.9 for turn in multi_legal
        ) / len(multi_legal)
        print(
            "2+ legal top probability: "
            f"mean={probability.mean:.4f} min={probability.minimum:.4f} "
            f"max={probability.maximum:.4f}"
        )
        print(
            "2+ legal entropy: "
            f"mean={entropy.mean:.4f} min={entropy.minimum:.4f} "
            f"max={entropy.maximum:.4f}"
        )
        print(f"2+ legal turns with top probability >= 0.9: {high_confidence:.1%}")
    print(f"actor-critic weights unchanged: {'yes' if result.weights_unchanged else 'no'}")


if __name__ == "__main__":
    main()
