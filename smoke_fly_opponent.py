"""Evaluate the saved PPO trainer agent in 10 battles against FlyCNSPlayer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration
from stable_baselines3 import PPO

from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from flycns.player import FlyCNSPlayer


SMOKE_BATTLES = 10


@dataclass(frozen=True)
class OpponentSmokeResult:
    wins: int
    losses: int
    ties: int
    rewards: tuple[float, ...]
    trainer_illegal_actions: int
    fly_illegal_actions: int
    fly_weights_unchanged: bool


def run_ten_battles(model: Any, environment: SingleAgentWrapper, *, seed: int = 7) -> OpponentSmokeResult:
    """Use the saved PPO agent for inference only; never call learn or update."""

    opponent = environment.opponent
    if not isinstance(opponent, FlyCNSPlayer):
        raise TypeError("Smoke environment must use FlyCNSPlayer")
    original_weights = [
        parameter.detach().clone() for parameter in opponent.policy.actor_critic.parameters()
    ]
    wins = losses = ties = trainer_illegal_actions = 0
    rewards: list[float] = []
    for episode in range(SMOKE_BATTLES):
        observation, _ = environment.reset(seed=seed + episode)
        terminated = truncated = False
        episode_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            selected = int(np.asarray(action).item())
            mask = np.asarray(observation["action_mask"])
            if not 0 <= selected < mask.size or mask[selected] != 1:
                trainer_illegal_actions += 1
                raise AssertionError(f"Trainer agent selected illegal action {selected}")
            observation, reward, terminated, truncated, _ = environment.step(
                np.int64(selected)
            )
            episode_reward += float(reward)
        battle = environment.env.battle1
        if battle is None:
            raise RuntimeError("Environment finished without a battle result")
        wins += int(battle.won)
        losses += int(battle.lost)
        ties += int(not battle.won and not battle.lost)
        rewards.append(episode_reward)

    weights_unchanged = all(
        torch.equal(parameter, original)
        for parameter, original in zip(
            opponent.policy.actor_critic.parameters(), original_weights, strict=True
        )
    )
    if not weights_unchanged:
        raise AssertionError("Fly CNS opponent weights changed during inference")
    return OpponentSmokeResult(
        wins=wins,
        losses=losses,
        ties=ties,
        rewards=tuple(rewards),
        trainer_illegal_actions=trainer_illegal_actions,
        fly_illegal_actions=opponent.illegal_action_count,
        fly_weights_unchanged=weights_unchanged,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("models/baseline"), help="saved trainer PPO model"
    )
    parser.add_argument(
        "--fly-checkpoint", type=Path, required=True, help="trained FlyCNSActorCritic checkpoint"
    )
    parser.add_argument(
        "--server-url",
        default=LocalhostServerConfiguration.websocket_url,
        help="WebSocket URL of the already-running local Showdown server",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    server = ServerConfiguration(
        args.server_url, LocalhostServerConfiguration.authentication_url
    )
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
        model = PPO.load(args.model, env=environment)
        model.policy.eval()
        result = run_ten_battles(model, environment, seed=args.seed)
    finally:
        environment.close()

    print(f"trainer wins/losses/ties: {result.wins}/{result.losses}/{result.ties}")
    print(f"rewards per battle: {[round(reward, 3) for reward in result.rewards]}")
    print(f"total reward: {sum(result.rewards):.3f}")
    print(f"average reward: {sum(result.rewards) / SMOKE_BATTLES:.3f}")
    print(
        "illegal actions attempted: "
        f"trainer={result.trainer_illegal_actions}, Fly={result.fly_illegal_actions}"
    )
    print(f"Fly weights unchanged: {'yes' if result.fly_weights_unchanged else 'no'}")


if __name__ == "__main__":
    main()
