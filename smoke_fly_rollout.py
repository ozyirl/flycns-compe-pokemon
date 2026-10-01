"""Play one local Showdown battle with the untrained Fly CNS policy, without learning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration

from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from flycns.ppo_policy import FlyCNSPPOPolicy


@dataclass(frozen=True)
class TurnRecord:
    observation: np.ndarray
    action_mask: np.ndarray
    selected_action: int
    log_probability: float
    state_value: float
    reward: float
    done: bool


@dataclass(frozen=True)
class RolloutResult:
    turns: tuple[TurnRecord, ...]
    total_reward: float
    outcome: str
    illegal_action_attempted: bool
    actor_critic_weights_unchanged: bool


def run_one_battle(
    environment: Any, policy: FlyCNSPPOPolicy, *, seed: int = 7
) -> RolloutResult:
    """Collect one complete episode from the existing Gymnasium environment."""

    original_weights = [
        parameter.detach().clone() for parameter in policy.actor_critic.parameters()
    ]
    records: list[TurnRecord] = []
    total_reward = 0.0
    illegal_action_attempted = False
    try:
        state, _ = environment.reset(seed=seed)
        done = False
        while not done:
            observation = np.asarray(state["observation"], dtype=np.float32).copy()
            action_mask = np.asarray(state["action_mask"], dtype=np.int8).copy()
            decision = policy.sample_action(observation, action_mask)
            action = decision.selected_action_index
            if not 0 <= action < action_mask.size or action_mask[action] != 1:
                illegal_action_attempted = True
                raise AssertionError(f"Fly CNS sampled illegal action {action}")

            state, reward, terminated, truncated, _ = environment.step(np.int64(action))
            done = bool(terminated or truncated)
            reward = float(reward)
            total_reward += reward
            records.append(
                TurnRecord(
                    observation=observation,
                    action_mask=action_mask,
                    selected_action=action,
                    log_probability=decision.action_log_probability,
                    state_value=decision.state_value,
                    reward=reward,
                    done=done,
                )
            )

        battle = environment.env.battle1
        if battle is None:
            raise RuntimeError("Environment completed without a battle result")
        outcome = "win" if battle.won else "loss" if battle.lost else "tie"
        weights_unchanged = all(
            torch.equal(parameter, original)
            for parameter, original in zip(
                policy.actor_critic.parameters(), original_weights, strict=True
            )
        )
        if not weights_unchanged:
            raise AssertionError("Actor-critic weights changed during the rollout")
        return RolloutResult(
            turns=tuple(records),
            total_reward=total_reward,
            outcome=outcome,
            illegal_action_attempted=illegal_action_attempted,
            actor_critic_weights_unchanged=weights_unchanged,
        )
    finally:
        environment.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-url",
        default=LocalhostServerConfiguration.websocket_url,
        help="WebSocket URL of the already-running local Showdown server",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = FlyCNSPPOPolicy()
    server = ServerConfiguration(
        args.server_url, LocalhostServerConfiguration.authentication_url
    )
    environment = SingleAgentWrapper(
        ShowdownEnv(
            battle_format=DEFAULT_BATTLE_FORMAT,
            server_configuration=server,
        ),
        make_opponent("random", battle_format=DEFAULT_BATTLE_FORMAT),
    )
    torch.manual_seed(args.seed)
    result = run_one_battle(environment, policy, seed=args.seed)

    print(f"turns: {len(result.turns)}")
    print(f"total reward: {result.total_reward:.3f}")
    print(f"win/loss: {result.outcome}")
    print(f"illegal action attempted: {'yes' if result.illegal_action_attempted else 'no'}")
    print(
        "actor-critic weights unchanged: "
        f"{'yes' if result.actor_critic_weights_unchanged else 'no'}"
    )


if __name__ == "__main__":
    main()
