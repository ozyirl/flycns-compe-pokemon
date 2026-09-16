"""Evaluate a saved policy against a scripted opponent."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from env import DEFAULT_BATTLE_FORMAT, make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=Path("models/baseline"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponent", choices=("random", "max-power"), default="random")
    parser.add_argument("--battle-format", default=DEFAULT_BATTLE_FORMAT)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_env(opponent=args.opponent, battle_format=args.battle_format)
    model = PPO.load(args.model, env=env)
    wins = 0
    total_reward = 0.0
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            terminated = truncated = False
            episode_reward = 0.0
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
            total_reward += episode_reward
            wins += int(env.env.battle1 is not None and env.env.battle1.won)
    finally:
        env.close()

    print(f"wins: {wins}/{args.episodes} ({wins / args.episodes:.1%})")
    print(f"mean episode reward: {total_reward / args.episodes:.3f}")


if __name__ == "__main__":
    main()

