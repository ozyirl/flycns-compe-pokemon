"""Benchmark one frozen FlyCNSPlayer checkpoint against three existing opponents."""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.player import Player
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration
from stable_baselines3 import PPO

from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from flycns.player import FlyCNSPlayer


BATTLES_PER_MATCHUP = 100
MatchupAction = Callable[[SingleAgentWrapper, dict[str, Any]], np.int64]


@dataclass(frozen=True)
class MatchupResult:
    name: str
    wins: int
    losses: int
    ties: int
    rewards: tuple[float, ...]
    turns: tuple[int, ...]
    fly_illegal_actions: int
    opposing_illegal_actions: int

    @property
    def win_rate(self) -> float:
        return self.wins / len(self.rewards)

    @property
    def average_reward(self) -> float:
        return float(np.mean(self.rewards))

    @property
    def average_battle_length(self) -> float:
        return float(np.mean(self.turns))


def scripted_action(player: Player) -> MatchupAction:
    """Convert an existing scripted Player order through the shared action space."""

    def choose(environment: SingleAgentWrapper, _observation: dict[str, Any]) -> np.int64:
        battle = environment.env.battle1
        if battle is None:
            raise RuntimeError("No active battle for scripted opponent")
        order = player.choose_move(battle)
        return environment.env.order_to_action(order, battle, strict=True)

    return choose


def baseline_action(model: PPO) -> MatchupAction:
    """Use the saved baseline policy for deterministic inference only."""

    def choose(_environment: SingleAgentWrapper, observation: dict[str, Any]) -> np.int64:
        action, _ = model.predict(observation, deterministic=True)
        return np.int64(np.asarray(action).item())

    return choose


def run_matchup(
    name: str,
    environment: SingleAgentWrapper,
    choose_action: MatchupAction,
    *,
    battles: int = BATTLES_PER_MATCHUP,
    seed: int = 7,
) -> MatchupResult:
    """Run complete battles, scoring every metric from FlyCNSPlayer's side."""

    if battles < 1:
        raise ValueError("battles must be positive")
    fly = environment.opponent
    if not isinstance(fly, FlyCNSPlayer):
        raise TypeError("Benchmark environment must use FlyCNSPlayer")
    original_weights = [parameter.detach().clone() for parameter in fly.policy.actor_critic.parameters()]
    initial_fly_illegal = fly.illegal_action_count
    wins = losses = ties = opposing_illegal = 0
    rewards: list[float] = []
    turns: list[int] = []
    random.seed(seed)
    np.random.seed(seed)

    for episode in range(battles):
        observation, _ = environment.reset(seed=seed + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            battle = environment.env.battle1
            if battle is None:
                raise RuntimeError("Environment has no trainer-side battle")
            if battle.wait:
                # The same default order used by poke-env when a side must wait.
                action = np.int64(-2)
            else:
                action = np.int64(choose_action(environment, observation))
                mask = np.asarray(observation["action_mask"])
                if not 0 <= action < mask.size or mask[action] != 1:
                    opposing_illegal += 1
                    raise ValueError(f"{name} selected illegal action {action}")
            observation, _, terminated, truncated, _ = environment.step(action)

        fly_battle = environment.env.battle2
        if fly_battle is None:
            raise RuntimeError("Environment finished without Fly's battle result")
        wins += int(fly_battle.won)
        losses += int(fly_battle.lost)
        ties += int(not fly_battle.won and not fly_battle.lost)
        # PokeEnv.step already called ShowdownEnv.calc_reward on both battles.
        # The reward buffer contains the exact cumulative reward for Fly's side.
        rewards.append(float(environment.env._reward_buffer[fly_battle]))
        turns.append(int(fly_battle.turn))
        if (episode + 1) % 25 == 0 or episode + 1 == battles:
            print(f"{name}: {episode + 1}/{battles} battles", flush=True)

    if any(
        not torch.equal(parameter, original)
        for parameter, original in zip(
            fly.policy.actor_critic.parameters(), original_weights, strict=True
        )
    ):
        raise AssertionError("FlyCNSPlayer weights changed during benchmark")
    return MatchupResult(
        name=name,
        wins=wins,
        losses=losses,
        ties=ties,
        rewards=tuple(rewards),
        turns=tuple(turns),
        fly_illegal_actions=fly.illegal_action_count - initial_fly_illegal,
        opposing_illegal_actions=opposing_illegal,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fly-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, default=Path("models/baseline"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--server-url",
        default=LocalhostServerConfiguration.websocket_url,
        help="WebSocket URL of the already-running local Showdown server",
    )
    args = parser.parse_args()
    server = ServerConfiguration(
        args.server_url, LocalhostServerConfiguration.authentication_url
    )
    results: list[MatchupResult] = []
    for name in ("random", "max-power", "baseline"):
        environment = SingleAgentWrapper(
            ShowdownEnv(
                battle_format=DEFAULT_BATTLE_FORMAT,
                server_configuration=server,
            ),
            make_opponent(
                "fly", battle_format=DEFAULT_BATTLE_FORMAT, fly_checkpoint=args.fly_checkpoint
            ),
        )
        try:
            if name == "baseline":
                model = PPO.load(args.baseline_model, env=environment)
                model.policy.eval()
                baseline_weights = {
                    key: value.detach().clone()
                    for key, value in model.policy.state_dict().items()
                }
                action_selector = baseline_action(model)
            else:
                player = make_opponent(name, battle_format=DEFAULT_BATTLE_FORMAT)
                action_selector = scripted_action(player)
            result = run_matchup(name, environment, action_selector, seed=args.seed)
            if name == "baseline" and any(
                not torch.equal(value, baseline_weights[key])
                for key, value in model.policy.state_dict().items()
            ):
                raise AssertionError("PPO baseline weights changed during benchmark")
            results.append(result)
        finally:
            environment.close()

    print("\nFlyCNSPlayer benchmark (Fly perspective; 100 battles each)")
    print(f"Fly checkpoint: {args.fly_checkpoint}")
    print(f"PPO baseline: {args.baseline_model}")
    print("opponent      wins losses ties win_rate avg_reward avg_turns illegal(Fly/other)")
    for result in results:
        print(
            f"{result.name:<13} {result.wins:>4} {result.losses:>6} "
            f"{result.ties:>4} {result.win_rate:>8.1%} "
            f"{result.average_reward:>10.3f} {result.average_battle_length:>9.2f} "
            f"{result.fly_illegal_actions}/{result.opposing_illegal_actions}"
        )


if __name__ == "__main__":
    main()
