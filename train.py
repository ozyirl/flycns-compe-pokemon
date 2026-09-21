"""Train a baseline or fly-connectome PPO policy against a scripted opponent."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from env import DEFAULT_BATTLE_FORMAT, make_env
from policies import (
    BaselineFeaturesExtractor,
    FlyConnectomeFeaturesExtractor,
    MaskedActorCriticPolicy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("models/baseline"))
    parser.add_argument("--opponent", choices=("random", "max-power"), default="random")
    parser.add_argument("--battle-format", default=DEFAULT_BATTLE_FORMAT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--architecture", choices=("baseline", "fly"), default="baseline"
    )
    parser.add_argument(
        "--connectome", type=Path, default=Path("artifacts/fly_connectome.npz")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = Monitor(make_env(opponent=args.opponent, battle_format=args.battle_format))
    try:
        if args.architecture == "fly":
            extractor_class = FlyConnectomeFeaturesExtractor
            extractor_kwargs = {
                "features_dim": 128,
                "graph_path": str(args.connectome),
            }
        else:
            extractor_class = BaselineFeaturesExtractor
            extractor_kwargs = {"features_dim": 128}

        model = PPO(
            MaskedActorCriticPolicy,
            env,
            policy_kwargs={
                "features_extractor_class": extractor_class,
                "features_extractor_kwargs": extractor_kwargs,
                "net_arch": {"pi": [64], "vf": [64]},
            },
            n_steps=1024,
            batch_size=64,
            learning_rate=3e-4,
            verbose=1,
            seed=args.seed,
        )
        model.learn(total_timesteps=args.timesteps)
        model.save(args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
