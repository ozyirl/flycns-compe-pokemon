"""Trace frozen-v1 FlyCNS TD residuals and GAE without performing PPO updates."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration

from diagnose_fly_reward_credit import RecordingEnvironment, StepObservation, observed_events
from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from experiment_fly_maxpower import DEFAULT_V1, sha256
from flycns.actor_critic import FlyCNSActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_update import advantages_and_returns, update_once
from smoke_fly_rollout import RolloutResult, run_one_battle


MIN_BATTLES = 20
MAX_BATTLES = 50
_GAE_PARAMETERS = inspect.signature(advantages_and_returns).parameters
GAMMA = float(_GAE_PARAMETERS["gamma"].default)
GAE_LAMBDA = float(_GAE_PARAMETERS["gae_lambda"].default)
EVENT_NAMES = ("opponent_fainted", "own_fainted", "final_win", "final_loss")


@dataclass(frozen=True)
class AdvantageTurn:
    battle: int
    turn: int
    outcome: str
    selected_action: int
    action_label: str
    events: tuple[str, ...]
    reward: float
    value: float
    next_value: float
    done: bool
    td_delta: float
    future_gae_contribution: float
    advantage: float
    computed_return: float


@dataclass(frozen=True)
class AdvantageBattle:
    battle: int
    outcome: str
    total_reward: float
    turns: tuple[AdvantageTurn, ...]


def advantages_normalized_before_policy_loss() -> bool:
    """Report the current update_once implementation, failing if its path drifts."""
    source = inspect.getsource(update_once)
    expected = (
        "advantage_tensor = torch.as_tensor(advantages",
        "ratio * advantage_tensor",
        "ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantage_tensor",
    )
    if any(fragment not in source for fragment in expected):
        raise RuntimeError("PPO advantage path changed; inspect normalization before reporting")
    if any(fragment in source for fragment in (
        "advantage_tensor.mean()", "advantage_tensor.std()", "advantages.mean()", "advantages.std()"
    )):
        raise RuntimeError("PPO advantage normalization may have changed; inspect update_once")
    return False


def trace_rollout(
    rollout: RolloutResult,
    steps: tuple[StepObservation, ...],
    *,
    battle_index: int,
) -> AdvantageBattle:
    """Reconstruct the existing terminal-aware GAE one turn at a time."""
    if rollout.illegal_action_attempted or not rollout.actor_critic_weights_unchanged:
        raise AssertionError("Expected a legal, read-only rollout")
    if len(rollout.turns) != len(steps):
        raise ValueError("Rollout and event steps must align")
    advantages, returns = advantages_and_returns(rollout)
    rows: list[AdvantageTurn] = []
    for index, (turn, step) in enumerate(zip(rollout.turns, steps, strict=True)):
        if turn.selected_action != step.action or not np.isclose(turn.reward, step.reward):
            raise AssertionError("Event step does not match recorded rollout")
        next_value = (
            0.0 if turn.done else float(rollout.turns[index + 1].state_value)
        )
        delta = float(turn.reward + GAMMA * next_value * (not turn.done) - turn.state_value)
        future = (
            0.0 if turn.done else GAMMA * GAE_LAMBDA * float(advantages[index + 1])
        )
        advantage = float(advantages[index])
        computed_return = float(returns[index])
        if not np.isclose(delta + future, advantage, atol=2e-5, rtol=2e-6):
            raise AssertionError("TD residual and future term do not reconstruct GAE")
        if not np.isclose(advantage + turn.state_value, computed_return, atol=2e-5):
            raise AssertionError("Advantage and critic value do not reconstruct return")
        flags = observed_events(step)
        rows.append(AdvantageTurn(
            battle=battle_index,
            turn=index + 1,
            outcome=rollout.outcome,
            selected_action=turn.selected_action,
            action_label=step.action_label,
            events=tuple(name for name in EVENT_NAMES if flags[name]),
            reward=float(turn.reward),
            value=float(turn.state_value),
            next_value=next_value,
            done=bool(turn.done),
            td_delta=delta,
            future_gae_contribution=future,
            advantage=advantage,
            computed_return=computed_return,
        ))
    return AdvantageBattle(battle_index, rollout.outcome, rollout.total_reward, tuple(rows))


def select_representative_turns(battles: tuple[AdvantageBattle, ...]) -> dict[str, AdvantageTurn | None]:
    turns = [turn for battle in battles for turn in battle.turns]
    selected: dict[str, AdvantageTurn | None] = {}
    for event in EVENT_NAMES:
        matches = [turn for turn in turns if event in turn.events]
        # Prefer a knockout with positive immediate reward but negative GAE.
        illustrative = next(
            (turn for turn in matches if turn.reward > 0 and turn.advantage < 0), None
        ) if event == "opponent_fainted" else None
        selected[event] = illustrative or (matches[0] if matches else None)
    return selected


def _print_turn(row: AdvantageTurn) -> None:
    events = ",".join(row.events) or "-"
    print(
        f"{row.turn:>3} {row.selected_action:>3} {row.reward:>+8.3f} "
        f"{row.value:>+8.3f} {row.next_value:>+8.3f} "
        f"{'Y' if row.done else 'N':>4} {row.td_delta:>+9.3f} "
        f"{row.future_gae_contribution:>+9.3f} {row.advantage:>+9.3f} "
        f"{row.computed_return:>+9.3f}  {events}"
    )


def _print_trace(battle: AdvantageBattle) -> None:
    print(f"\nComplete {battle.outcome} battle {battle.battle}: {len(battle.turns)} turns, reward {battle.total_reward:+.3f}")
    print("turn act      r_t   V(s_t) V(s_t+1) done     delta future GAE       A_t    return  events")
    for turn in battle.turns:
        _print_turn(turn)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_V1)
    parser.add_argument("--server-url", default=LocalhostServerConfiguration.websocket_url)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-battles", type=int, default=MIN_BATTLES)
    parser.add_argument("--max-battles", type=int, default=MAX_BATTLES)
    parser.add_argument("--output-dir", type=Path, help="new directory for all turn rows and summary")
    args = parser.parse_args()
    if not 1 <= args.min_battles <= args.max_battles:
        parser.error("require 1 <= min-battles <= max-battles")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Fly checkpoint not found: {checkpoint}")
    if args.output_dir is not None and args.output_dir.exists():
        parser.error(f"output directory must not already exist: {args.output_dir}")
    normalized = advantages_normalized_before_policy_loss()
    checksum = sha256(checkpoint)
    actor_critic = FlyCNSActorCritic.load_weights(checkpoint)
    actor_critic.eval().requires_grad_(False)
    policy = FlyCNSPPOPolicy(actor_critic=actor_critic)
    weights = [parameter.detach().clone() for parameter in actor_critic.parameters()]
    server = ServerConfiguration(args.server_url, LocalhostServerConfiguration.authentication_url)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    battles: list[AdvantageBattle] = []
    for index in range(args.max_battles):
        environment = RecordingEnvironment(SingleAgentWrapper(
            ShowdownEnv(battle_format=DEFAULT_BATTLE_FORMAT, server_configuration=server),
            make_opponent("max-power", battle_format=DEFAULT_BATTLE_FORMAT),
        ))
        rollout = run_one_battle(environment, policy, seed=args.seed + index)
        battles.append(trace_rollout(rollout, tuple(environment.steps), battle_index=index + 1))
        if (index + 1) % 10 == 0:
            print(f"diagnostic: {index + 1}/{args.max_battles} battles", flush=True)
        if len(battles) >= args.min_battles and {"win", "loss"} <= {b.outcome for b in battles}:
            break
    if not {"win", "loss"} <= {battle.outcome for battle in battles}:
        raise RuntimeError("Could not observe both a win and a loss within max-battles")
    if sha256(checkpoint) != checksum or any(
        not torch.equal(parameter, original)
        for parameter, original in zip(actor_critic.parameters(), weights, strict=True)
    ):
        raise AssertionError("v1 checkpoint or actor-critic weights changed")

    frozen = tuple(battles)
    examples = select_representative_turns(frozen)
    outcomes = Counter(b.outcome for b in frozen)
    summary = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": checksum,
        "opponent": "MaxBasePowerPlayer", "seed": args.seed,
        "battles": len(frozen), "outcomes": dict(outcomes),
        "gamma": GAMMA, "gae_lambda": GAE_LAMBDA,
        "advantages_normalized_before_policy_loss": normalized,
        "normalization_note": "update_once uses the raw GAE tensor directly in the clipped policy loss",
        "representative_turns": {
            event: asdict(turn) if turn is not None else None
            for event, turn in examples.items()
        },
    }
    root = Path(__file__).resolve().parent / "models"
    if args.output_dir is None:
        root.mkdir(exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="flycns_advantage_trace_", dir=root))
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "turns.jsonl").write_text(
        "\n".join(json.dumps(asdict(turn)) for battle in frozen for turn in battle.turns) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nGAE gamma={GAMMA}, lambda={GAE_LAMBDA}; advantages normalized before PPO loss: {normalized}")
    print(f"{len(frozen)} read-only battles: {dict(outcomes)}")
    print("\nRepresentative event turns")
    print("turn act      r_t   V(s_t) V(s_t+1) done     delta future GAE       A_t    return  events")
    for event, turn in examples.items():
        print(f"{event}:")
        if turn is None:
            print("  not observed")
        else:
            _print_turn(turn)
    _print_trace(next(b for b in frozen if b.outcome == "win"))
    _print_trace(next(b for b in frozen if b.outcome == "loss"))
    print(f"\nAll turns: {output_dir / 'turns.jsonl'}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print("No PPO update or training performed; v1 unchanged")


if __name__ == "__main__":
    main()
