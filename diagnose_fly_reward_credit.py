"""Read-only, turn-level reward attribution for frozen FlyCNS vs MaxBasePower."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration

from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from experiment_fly_maxpower import DEFAULT_V1, sha256
from flycns.actor_critic import FlyCNSActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_update import advantages_and_returns
from smoke_fly_rollout import run_one_battle


BATTLES = 50
HP_WEIGHT = 0.5
FAINT_WEIGHT = 2.0
STATUS_WEIGHT = 0.1
VICTORY_WEIGHT = 10.0
TEAM_SIZE = 6
COMPONENTS = (
    "own_hp", "own_faint", "own_status", "own_unrevealed",
    "opponent_hp", "opponent_faint", "opponent_status", "opponent_unrevealed",
    "victory",
)
EVENTS = (
    "damage_dealt", "damage_received", "opponent_fainted", "own_fainted",
    "successful_switch", "status_inflicted", "status_received", "final_win", "final_loss",
)


@dataclass(frozen=True)
class MonState:
    hp: float
    fainted: bool
    status: str | None
    active: bool


@dataclass(frozen=True)
class BattleState:
    own: dict[str, MonState]
    opponent: dict[str, MonState]
    won: bool
    lost: bool
    active_own: str | None


@dataclass(frozen=True)
class StepObservation:
    before: BattleState
    after: BattleState
    action: int
    action_label: str
    reward: float


def snapshot(battle: Any) -> BattleState:
    def team_state(team: Any) -> dict[str, MonState]:
        return {
            str(key): MonState(
                hp=float(mon.current_hp_fraction),
                fainted=bool(mon.fainted),
                status=str(mon.status) if mon.status is not None else None,
                active=bool(mon.active),
            )
            for key, mon in team.items()
        }

    own = team_state(battle.team)
    return BattleState(
        own=own,
        opponent=team_state(battle.opponent_team),
        won=bool(battle.won),
        lost=bool(battle.lost),
        active_own=next((key for key, mon in own.items() if mon.active), None),
    )


def reward_components(state: BattleState) -> dict[str, float]:
    """Exactly mirror ShowdownEnv's current reward-helper state value."""
    return {
        "own_hp": HP_WEIGHT * sum(mon.hp for mon in state.own.values()),
        "own_faint": -FAINT_WEIGHT * sum(mon.fainted for mon in state.own.values()),
        "own_status": -STATUS_WEIGHT * sum(
            mon.status is not None and not mon.fainted for mon in state.own.values()
        ),
        "own_unrevealed": HP_WEIGHT * (TEAM_SIZE - len(state.own)),
        "opponent_hp": -HP_WEIGHT * sum(mon.hp for mon in state.opponent.values()),
        "opponent_faint": FAINT_WEIGHT * sum(mon.fainted for mon in state.opponent.values()),
        "opponent_status": STATUS_WEIGHT * sum(
            mon.status is not None and not mon.fainted for mon in state.opponent.values()
        ),
        "opponent_unrevealed": -HP_WEIGHT * (TEAM_SIZE - len(state.opponent)),
        "victory": VICTORY_WEIGHT * (int(state.won) - int(state.lost)),
    }


def observed_events(step: StepObservation) -> dict[str, bool]:
    before, after = step.before, step.after

    def any_delta(old: dict[str, MonState], new: dict[str, MonState], predicate: Any) -> bool:
        return any(predicate(old[key], new[key]) for key in old.keys() & new.keys())

    switched = (
        0 <= step.action < 6
        and step.action_label != "wait/default"
        and before.active_own != after.active_own
        and after.active_own is not None
    )
    return {
        "damage_dealt": any_delta(before.opponent, after.opponent, lambda a, b: b.hp < a.hp - 1e-6),
        "damage_received": any_delta(before.own, after.own, lambda a, b: b.hp < a.hp - 1e-6),
        "opponent_fainted": any_delta(before.opponent, after.opponent, lambda a, b: not a.fainted and b.fainted),
        "own_fainted": any_delta(before.own, after.own, lambda a, b: not a.fainted and b.fainted),
        "successful_switch": switched,
        "status_inflicted": any_delta(
            before.opponent, after.opponent,
            lambda a, b: a.status is None and b.status is not None and not b.fainted,
        ),
        "status_received": any_delta(
            before.own, after.own,
            lambda a, b: a.status is None and b.status is not None and not b.fainted,
        ),
        "final_win": not before.won and after.won,
        "final_loss": not before.lost and after.lost,
    }


def direct_event_reward(step: StepObservation) -> dict[str, float]:
    """Reward-helper terms tied to observed transitions, excluding unseen changes."""
    before, after = step.before, step.after

    def shared(old: dict[str, MonState], new: dict[str, MonState]) -> list[tuple[MonState, MonState]]:
        return [(old[key], new[key]) for key in old.keys() & new.keys()]

    own = shared(before.own, after.own)
    opponent = shared(before.opponent, after.opponent)
    return {
        "damage_dealt": HP_WEIGHT * sum(max(a.hp - b.hp, 0.0) for a, b in opponent),
        "damage_received": -HP_WEIGHT * sum(max(a.hp - b.hp, 0.0) for a, b in own),
        "opponent_fainted": FAINT_WEIGHT * sum(not a.fainted and b.fainted for a, b in opponent),
        "own_fainted": -FAINT_WEIGHT * sum(not a.fainted and b.fainted for a, b in own),
        "successful_switch": 0.0,
        "status_inflicted": STATUS_WEIGHT * sum(
            a.status is None and b.status is not None and not b.fainted for a, b in opponent
        ),
        "status_received": -STATUS_WEIGHT * sum(
            a.status is None and b.status is not None and not b.fainted for a, b in own
        ),
        "final_win": VICTORY_WEIGHT * int(not before.won and after.won),
        "final_loss": -VICTORY_WEIGHT * int(not before.lost and after.lost),
    }


class RecordingEnvironment:
    """Transparent step observer; existing rollout/policy/environment stay unchanged."""

    def __init__(self, environment: Any) -> None:
        self._environment = environment
        self.env = environment.env
        self.steps: list[StepObservation] = []

    def reset(self, **kwargs: Any) -> Any:
        return self._environment.reset(**kwargs)

    def step(self, action: Any) -> Any:
        battle = self.env.battle1
        before = snapshot(battle)
        action_index = int(action)
        # The mask uses index 0 as a placeholder while our side is waiting;
        # SinglesEnv.step deliberately does not convert an order in that case.
        action_label = (
            "wait/default"
            if not self.env.agent1_to_move
            else str(self.env.action_to_order(np.int64(action_index), battle))
        )
        result = self._environment.step(action)
        self.steps.append(
            StepObservation(
                before=before,
                after=snapshot(self.env.battle1),
                action=action_index,
                action_label=action_label,
                reward=float(result[1]),
            )
        )
        return result

    def close(self) -> None:
        self._environment.close()


def inspect_battle(policy: FlyCNSPPOPolicy, environment: RecordingEnvironment, *, seed: int, battle_index: int) -> list[dict[str, Any]]:
    rollout = run_one_battle(environment, policy, seed=seed)
    if rollout.illegal_action_attempted or not rollout.actor_critic_weights_unchanged:
        raise AssertionError("Expected legal, read-only Fly rollout")
    if len(rollout.turns) != len(environment.steps):
        raise AssertionError("Turn/step count mismatch")
    advantages, _ = advantages_and_returns(rollout)
    rows: list[dict[str, Any]] = []
    previous = {name: 0.0 for name in COMPONENTS}  # reward helper starts at zero on each battle
    for index, (turn, step, advantage) in enumerate(
        zip(rollout.turns, environment.steps, advantages, strict=True), start=1
    ):
        if turn.selected_action != step.action or not np.isclose(turn.reward, step.reward):
            raise AssertionError("Recorded step differs from PPO rollout")
        current = reward_components(step.after)
        contributions = {name: current[name] - previous[name] for name in COMPONENTS}
        if not np.isclose(sum(contributions.values()), turn.reward, atol=1e-5):
            raise AssertionError("Reward attribution does not reconstruct environment reward")
        events = observed_events(step)
        direct = direct_event_reward(step)
        rows.append({
            "battle": battle_index, "turn": index, "outcome": rollout.outcome,
            "selected_action": turn.selected_action, "action_label": step.action_label,
            "reward": turn.reward, "ppo_advantage": float(advantage),
            "events": events, "reward_components": contributions,
            "direct_event_reward_terms": direct,
            "net_opponent_hp_change": sum(
                step.after.opponent[key].hp - step.before.opponent[key].hp
                for key in step.before.opponent.keys() & step.after.opponent.keys()
            ),
            "net_own_hp_change": sum(
                step.after.own[key].hp - step.before.own[key].hp
                for key in step.before.own.keys() & step.after.own.keys()
            ),
        })
        previous = current
    if not np.isclose(sum(row["reward"] for row in rows), rollout.total_reward):
        raise AssertionError("Turn rewards do not sum to battle reward")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No diagnostic turns")
    components = {name: float(sum(row["reward_components"][name] for row in rows)) for name in COMPONENTS}
    events: dict[str, Any] = {}
    for name in EVENTS:
        indices = [i for i, row in enumerate(rows) if row["events"][name]]
        event_rows = [rows[i] for i in indices]
        def nearby(offset: int) -> float | None:
            values = [rows[i + offset]["ppo_advantage"] for i in indices
                      if 0 <= i + offset < len(rows) and rows[i + offset]["battle"] == rows[i]["battle"]]
            return float(np.mean(values)) if values else None
        events[name] = {
            "turns": len(indices),
            "direct_reward_term_sum": float(sum(row["direct_event_reward_terms"][name] for row in event_rows)),
            "reward_sum_on_event_turns_overlapping": float(sum(row["reward"] for row in event_rows)),
            "mean_reward_on_event_turns": float(np.mean([row["reward"] for row in event_rows])) if event_rows else None,
            "mean_advantage_without_event": float(np.mean([
                row["ppo_advantage"] for row in rows if not row["events"][name]
            ])) if len(indices) < len(rows) else None,
            "mean_advantage_previous_turn": nearby(-1),
            "mean_advantage_event_turn": nearby(0),
            "mean_advantage_next_turn": nearby(1),
        }
    outcomes = Counter(row["outcome"] for row in rows if row["turn"] == 1)
    return {
        "battles": sum(outcomes.values()), "turns": len(rows), "outcomes": dict(outcomes),
        "total_reward": float(sum(row["reward"] for row in rows)),
        "reward_components_exact": components,
        "events_observed": events,
        "notes": [
            "Exact reward components are changes in the existing reward-helper state terms, not causal action credit.",
            "Event-turn reward sums overlap when events co-occur and must not be added together.",
            "Direct event reward terms apply the existing helper's weights to visible net transitions; they exclude healing, status removal, roster revelation, and initial-state effects.",
            "HP/status events use visible before/after snapshots; within-step changes that reverse are not observable.",
            "Switch means a selected switch action with a changed active Pokémon; switching has no direct reward term.",
            "Advantages are existing terminal-aware GAE, computed separately per battle from frozen critic values.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_V1)
    parser.add_argument("--output-dir", type=Path, help="new directory for turn rows and summary")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--server-url", default=LocalhostServerConfiguration.websocket_url)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Fly checkpoint not found: {checkpoint}")
    if args.output_dir is not None and args.output_dir.exists():
        parser.error(f"output directory must not already exist: {args.output_dir}")
    original_hash = sha256(checkpoint)
    actor_critic = FlyCNSActorCritic.load_weights(checkpoint)
    actor_critic.eval().requires_grad_(False)
    policy = FlyCNSPPOPolicy(actor_critic=actor_critic)
    original_weights = [p.detach().clone() for p in actor_critic.parameters()]
    server = ServerConfiguration(args.server_url, LocalhostServerConfiguration.authentication_url)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rows: list[dict[str, Any]] = []
    for battle_index in range(1, BATTLES + 1):
        environment = RecordingEnvironment(SingleAgentWrapper(
            ShowdownEnv(battle_format=DEFAULT_BATTLE_FORMAT, server_configuration=server),
            make_opponent("max-power", battle_format=DEFAULT_BATTLE_FORMAT),
        ))
        rows.extend(inspect_battle(policy, environment, seed=args.seed + battle_index - 1, battle_index=battle_index))
        if battle_index % 10 == 0:
            print(f"diagnostic: {battle_index}/{BATTLES} battles", flush=True)
    if sha256(checkpoint) != original_hash or any(
        not torch.equal(p, old) for p, old in zip(actor_critic.parameters(), original_weights, strict=True)
    ):
        raise AssertionError("v1 checkpoint or actor-critic weights changed")
    report = summarize(rows)
    report.update({"checkpoint": str(checkpoint), "checkpoint_sha256": original_hash,
                   "opponent": "MaxBasePowerPlayer", "seed": args.seed, "rollout_action_selection": "stochastic"})
    if args.output_dir is None:
        root = Path(__file__).resolve().parent / "models"
        root.mkdir(exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="flycns_reward_credit_", dir=root))
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "turns.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{report['battles']} battles, {report['turns']} turns, outcomes {report['outcomes']}")
    print(f"Total reward: {report['total_reward']:.3f}")
    print("Exact reward terms:")
    for name, amount in report["reward_components_exact"].items():
        print(f"  {name:<22} {amount:+.3f}")
    print("Observed events: turns, direct reward term, overlapping turn reward, advantage previous/event/next")
    for name, item in report["events_observed"].items():
        fmt = lambda value: "n/a" if value is None else f"{value:+.3f}"
        print(f"  {name:<22} {item['turns']:>4} {item['direct_reward_term_sum']:>+9.3f} "
              f"{item['reward_sum_on_event_turns_overlapping']:>+9.3f} "
              f"{fmt(item['mean_advantage_previous_turn'])}/"
              f"{fmt(item['mean_advantage_event_turn'])}/"
              f"{fmt(item['mean_advantage_next_turn'])}")
    print(f"Turn records: {output_dir / 'turns.jsonl'}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
