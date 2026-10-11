"""Offline linear probe of frozen FlyCNS features using MaxBasePower decisions.

MaxBasePowerPlayer plays complete singles battles against RandomPlayer. This
script records its actual legal orders in the existing 26-action encoding, then
trains only a fresh linear decoder on frozen descending spike counts.
"""

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
from poke_env.player import MaxBasePowerPlayer
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration
from torch.nn import functional as F

from encoding import OBSERVATION_DIM
from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from flycns.action_decoder import ACTION_COUNT
from flycns.graph import FlyGraph
from flycns.ppo_policy import DEFAULT_GRAPH_PATH
from flycns.stimulation import BattleSensoryAdapter, FlySpikeSimulator
from flycns.trainable_decoder import DESCENDING_ACTIVITY_SIZE, TrainableActionDecoder


@dataclass(frozen=True)
class Examples:
    observations: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    battle_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)


def collect_examples(
    environment: Any,
    teacher: MaxBasePowerPlayer,
    *,
    battles: int,
    seed: int,
) -> Examples:
    """Record the teacher's chosen order before executing each non-wait turn."""

    if battles < 2:
        raise ValueError("At least two battles are needed for a battle-level split")
    random.seed(seed)  # MaxBasePower uses random only when forced to switch.
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    battle_ids: list[int] = []

    for battle_id in range(battles):
        state, _ = environment.reset(seed=seed + battle_id)
        done = False
        while not done:
            battle = environment.env.battle1
            if battle is None:
                raise RuntimeError("Environment has no active teacher-side battle")
            if battle.wait:
                action = np.int64(-2)
            else:
                observation = np.asarray(state["observation"], dtype=np.float32).copy()
                mask = np.asarray(state["action_mask"], dtype=np.int8).copy()
                if observation.shape != (OBSERVATION_DIM,) or mask.shape != (ACTION_COUNT,):
                    raise ValueError("Environment returned an incompatible observation or mask")
                order = teacher.choose_move(battle)
                action = environment.env.order_to_action(order, battle, strict=True)
                index = int(action)
                if not 0 <= index < ACTION_COUNT or mask[index] != 1:
                    raise ValueError(f"MaxBasePower selected non-representable/illegal action {index}")
                observations.append(observation)
                masks.append(mask)
                actions.append(index)
                battle_ids.append(battle_id)
            state, _, terminated, truncated, _ = environment.step(action)
            done = bool(terminated or truncated)
        print(f"collected battle {battle_id + 1}/{battles}: {len(actions)} decisions", flush=True)

    return Examples(
        observations=np.stack(observations),
        action_masks=np.stack(masks),
        actions=np.asarray(actions, dtype=np.int64),
        battle_ids=np.asarray(battle_ids, dtype=np.int32),
    )


def frozen_features(examples: Examples, graph_path: Path) -> np.ndarray:
    """Use the existing, parameter-free sensory adapter and spike simulator."""

    graph = FlyGraph.load(graph_path)
    adapter = BattleSensoryAdapter(graph)
    simulator = FlySpikeSimulator(graph)
    descending: list[np.ndarray] = []
    for observation in examples.observations:
        report = simulator.run(adapter.adapt(observation))
        descending.append(report.spike_counts[report.descending_indices])
    features = np.stack(descending).astype(np.float32)
    if features.shape != (len(examples), DESCENDING_ACTIVITY_SIZE):
        raise ValueError(f"Unexpected descending feature shape: {features.shape}")
    return features


def split_battles(battle_ids: np.ndarray, *, seed: int, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    unique = np.unique(battle_ids)
    if len(unique) < 2:
        raise ValueError("Need examples from at least two battles")
    shuffled = np.random.default_rng(seed).permutation(unique)
    test_count = min(len(unique) - 1, max(1, round(len(unique) * test_fraction)))
    test = np.flatnonzero(np.isin(battle_ids, shuffled[:test_count]))
    train = np.flatnonzero(np.isin(battle_ids, shuffled[test_count:]))
    if not len(train) or not len(test):
        raise ValueError("Both train and test splits must contain decisions")
    return train, test


def masked_logits(decoder: TrainableActionDecoder, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return decoder(features).masked_fill(~masks, float("-inf"))


def accuracy(
    decoder: TrainableActionDecoder,
    features: torch.Tensor,
    masks: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[float, float]:
    decoder.eval()
    with torch.inference_mode():
        logits = masked_logits(decoder, features, masks)
        top = logits.topk(3, dim=-1).indices
        top1 = (top[:, 0] == actions).float().mean().item()
        top3 = (top == actions[:, None]).any(dim=-1).float().mean().item()
    return float(top1), float(top3)


def train_probe(
    features: np.ndarray,
    examples: Examples,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> dict[str, float]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if not np.isfinite(features).all():
        raise ValueError("Descending features must be finite")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        decoder = TrainableActionDecoder()
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        x = torch.from_numpy(features)
        masks = torch.from_numpy(examples.action_masks.astype(bool))
        labels = torch.from_numpy(examples.actions)
        if not torch.all(masks[torch.arange(len(labels)), labels]):
            raise ValueError("Teacher labels must be legal under their action masks")
        train_rows = torch.from_numpy(train_indices)
        test_rows = torch.from_numpy(test_indices)
        generator = torch.Generator().manual_seed(seed)

        for _ in range(epochs):
            decoder.train()
            order = train_rows[torch.randperm(len(train_rows), generator=generator)]
            for rows in order.split(batch_size):
                optimizer.zero_grad()
                loss = F.cross_entropy(masked_logits(decoder, x[rows], masks[rows]), labels[rows])
                loss.backward()
                optimizer.step()

        train_accuracy, train_top3 = accuracy(decoder, x[train_rows], masks[train_rows], labels[train_rows])
        test_accuracy, test_top3 = accuracy(decoder, x[test_rows], masks[test_rows], labels[test_rows])
    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_top3_accuracy": train_top3,
        "test_top3_accuracy": test_top3,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battles", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--output-dir", type=Path, help="New directory for collected examples and metrics")
    parser.add_argument(
        "--server-url",
        default=LocalhostServerConfiguration.websocket_url,
        help="WebSocket URL of the local Pokémon Showdown server",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        parent = Path("models")
        parent.mkdir(exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="fly_supervised_probe_", dir=parent))
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=False)

    server = ServerConfiguration(args.server_url, LocalhostServerConfiguration.authentication_url)
    environment = SingleAgentWrapper(
        ShowdownEnv(battle_format=DEFAULT_BATTLE_FORMAT, server_configuration=server),
        make_opponent("random", battle_format=DEFAULT_BATTLE_FORMAT),
    )
    teacher = MaxBasePowerPlayer(battle_format=DEFAULT_BATTLE_FORMAT, start_listening=False)
    try:
        examples = collect_examples(environment, teacher, battles=args.battles, seed=args.seed)
    finally:
        environment.close()
    np.savez_compressed(
        output_dir / "decisions.npz",
        observations=examples.observations,
        action_masks=examples.action_masks,
        actions=examples.actions,
        battle_ids=examples.battle_ids,
    )
    features = frozen_features(examples, args.graph)
    np.save(output_dir / "descending_features.npy", features)
    train_indices, test_indices = split_battles(
        examples.battle_ids, seed=args.seed, test_fraction=args.test_fraction
    )
    np.savez_compressed(output_dir / "split_indices.npz", train=train_indices, test=test_indices)
    metrics = train_probe(
        features, examples, train_indices, test_indices,
        seed=args.seed, epochs=args.epochs, learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    counts = Counter(int(action) for action in examples.actions)
    report = {
        "battles": args.battles,
        "examples": len(examples),
        "train_examples": len(train_indices),
        "test_examples": len(test_indices),
        "seed": args.seed,
        "epochs": args.epochs,
        "action_frequency": {str(index): counts[index] for index in range(ACTION_COUNT)},
        **metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nexamples: {len(examples)} ({len(train_indices)} train, {len(test_indices)} test)")
    print(f"train accuracy: {metrics['train_accuracy']:.1%} | top-3: {metrics['train_top3_accuracy']:.1%}")
    print(f"test accuracy:  {metrics['test_accuracy']:.1%} | top-3: {metrics['test_top3_accuracy']:.1%}")
    print("action frequency (index: count): " + ", ".join(f"{i}: {counts[i]}" for i in range(ACTION_COUNT)))
    print(f"saved examples, frozen features, split, and metrics: {output_dir}")


if __name__ == "__main__":
    main()
