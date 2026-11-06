"""Compare Fly-only and hybrid heads under identical PPO, both warm-started from v1.

The hybrid adds zero-initialized scorers that read each switch/move slot's raw
features beside the fixed Fly CNS activity, so both arms start from exactly the
v1 policy. Unlike ``update_once``, one Adam optimizer persists across steps.
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration
from torch.distributions import Categorical

from benchmark_fly_opponent import run_matchup, scripted_action
from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from experiment_fly_maxpower import DEFAULT_V1, sha256
from experiment_fly_maxpower_batched import BATCH_BATTLES, combine_rollouts
from flycns.actor_critic import FlyCNSActorCritic
from flycns.hybrid_actor_critic import FlyCNSHybridActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_update import advantages_and_returns, normalized_policy_advantages
from smoke_fly_rollout import RolloutResult, run_one_battle


ARMS = ("fly", "hybrid")
EPOCHS_PER_BATCH = 8
CLIP_RANGE = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
MAX_GRAD_NORM = 0.5


@dataclass(frozen=True)
class BatchStats:
    through_battle: int
    transitions: int
    wins: int
    policy_loss: float
    value_loss: float
    entropy: float
    clip_fraction: float
    approximate_kl: float


@dataclass(frozen=True)
class CheckpointEvaluation:
    after_battle: int
    checkpoint: str
    checkpoint_sha256: str
    opponent: str
    battles: int
    wins: int
    losses: int
    ties: int
    win_rate: float
    average_reward: float
    average_turns: float
    fly_illegal_actions: int


def make_model(arm: str, v1: Path) -> FlyCNSActorCritic | FlyCNSHybridActorCritic:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
    source = FlyCNSActorCritic.load_weights(v1)
    return source if arm == "fly" else FlyCNSHybridActorCritic.from_fly(source)


def make_optimizer(
    model: FlyCNSActorCritic | FlyCNSHybridActorCritic,
    *,
    learning_rate: float,
    raw_learning_rate: float,
) -> torch.optim.Adam:
    """One persistent Adam; raw-feature parameters get their own learning rate."""

    if isinstance(model, FlyCNSHybridActorCritic):
        return torch.optim.Adam([
            {"params": model.fly_parameters(), "lr": learning_rate},
            {"params": model.raw_parameters(), "lr": raw_learning_rate},
        ])
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def ppo_epochs(
    policy: FlyCNSPPOPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutResult,
    *,
    epochs: int = EPOCHS_PER_BATCH,
) -> dict[str, float]:
    """Full-batch clipped PPO with normalized advantages; Fly features computed once."""

    advantages, returns = advantages_and_returns(rollout)
    observations = np.stack([turn.observation for turn in rollout.turns])
    masks = np.stack([turn.action_mask for turn in rollout.turns]).astype(bool)
    actions = np.asarray([turn.selected_action for turn in rollout.turns], dtype=np.int64)
    if not np.all(masks[np.arange(len(actions)), actions]):
        raise ValueError("Rollout contains an illegal selected action")
    descending = np.stack([policy._descending_activity(row) for row in observations])

    model = policy.actor_critic
    activity = torch.from_numpy(descending)
    raw_observations = torch.from_numpy(observations.astype(np.float32))
    legal = torch.from_numpy(masks)
    selected = torch.from_numpy(actions)
    old_log_probs = torch.tensor([turn.log_probability for turn in rollout.turns], dtype=torch.float32)
    advantage = torch.from_numpy(normalized_policy_advantages(advantages))
    target = torch.from_numpy(returns)

    model.train()
    for _ in range(epochs):
        logits, values = model(activity, raw_observations)
        distribution = Categorical(logits=logits.masked_fill(~legal, float("-inf")))
        log_ratio = distribution.log_prob(selected) - old_log_probs
        ratio = log_ratio.exp()
        policy_loss = -torch.minimum(
            ratio * advantage, ratio.clamp(1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * advantage
        ).mean()
        value_loss = 0.5 * torch.square(values.squeeze(-1) - target).mean()
        entropy = distribution.entropy().mean()
        loss = policy_loss + VALUE_COEFFICIENT * value_loss - ENTROPY_COEFFICIENT * entropy
        if not torch.isfinite(loss):
            raise FloatingPointError("PPO produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
    model.eval()

    with torch.no_grad():
        logits, _ = model(activity, raw_observations)
        log_ratio = Categorical(logits=logits.masked_fill(~legal, float("-inf"))).log_prob(selected) - old_log_probs
        ratio = log_ratio.exp()
    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "clip_fraction": float(((ratio - 1.0).abs() > CLIP_RANGE).float().mean().item()),
        "approximate_kl": float(((ratio - 1.0) - log_ratio).mean().item()),
    }


def training_environment(server: ServerConfiguration, opponent: str) -> SingleAgentWrapper:
    nonce = secrets.token_hex(5)
    return SingleAgentWrapper(
        ShowdownEnv(
            battle_format=DEFAULT_BATTLE_FORMAT,
            server_configuration=server,
            account_configuration1=AccountConfiguration(f"fh{nonce}a", None),
            account_configuration2=AccountConfiguration(f"fh{nonce}b", None),
        ),
        make_opponent(opponent, battle_format=DEFAULT_BATTLE_FORMAT),
    )


def evaluate(
    checkpoint: Path, server: ServerConfiguration, *, opponent: str, battles: int, seed: int, after_battle: int
) -> CheckpointEvaluation:
    """Read-only battles with the checkpoint loaded as a fresh FlyCNSPlayer."""

    checkpoint_hash = sha256(checkpoint)
    nonce = secrets.token_hex(5)
    environment = SingleAgentWrapper(
        ShowdownEnv(
            battle_format=DEFAULT_BATTLE_FORMAT,
            server_configuration=server,
            account_configuration1=AccountConfiguration(f"fe{nonce}a", None),
            account_configuration2=AccountConfiguration(f"fe{nonce}b", None),
        ),
        make_opponent("fly", battle_format=DEFAULT_BATTLE_FORMAT, fly_checkpoint=checkpoint),
    )
    try:
        scripted = make_opponent(opponent, battle_format=DEFAULT_BATTLE_FORMAT)
        result = run_matchup(opponent, environment, scripted_action(scripted), battles=battles, seed=seed)
    finally:
        environment.close()
    if sha256(checkpoint) != checkpoint_hash:
        raise AssertionError(f"Checkpoint changed during read-only evaluation: {checkpoint}")
    return CheckpointEvaluation(
        after_battle=after_battle,
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_hash,
        opponent=opponent,
        battles=battles,
        wins=result.wins,
        losses=result.losses,
        ties=result.ties,
        win_rate=result.win_rate,
        average_reward=result.average_reward,
        average_turns=result.average_battle_length,
        fly_illegal_actions=result.fly_illegal_actions,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    v1_hash = sha256(args.v1_checkpoint)
    server = ServerConfiguration(args.server_url, LocalhostServerConfiguration.authentication_url)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = make_model(args.arm, args.v1_checkpoint)
    model.eval()
    policy = FlyCNSPPOPolicy(actor_critic=model)
    optimizer = make_optimizer(
        model, learning_rate=args.learning_rate, raw_learning_rate=args.raw_learning_rate
    )

    pending: list[RolloutResult] = []
    batches: list[BatchStats] = []
    checkpoints: list[tuple[int, Path]] = []
    for index in range(args.training_battles):
        completed = index + 1
        pending.append(run_one_battle(
            training_environment(server, args.training_opponent), policy, seed=args.seed + index
        ))
        if len(pending) == BATCH_BATTLES or completed == args.training_battles:
            stats = ppo_epochs(policy, optimizer, combine_rollouts(tuple(pending)))
            wins = sum(rollout.outcome == "win" for rollout in pending)
            batches.append(BatchStats(
                through_battle=completed, transitions=sum(len(r.turns) for r in pending), wins=wins, **stats
            ))
            print(
                f"[{args.arm} seed {args.seed}] battles {completed - len(pending) + 1}-{completed}: "
                f"won {wins}/{len(pending)} | kl {stats['approximate_kl']:.4f} "
                f"clip {stats['clip_fraction']:.2f} entropy {stats['entropy']:.3f}",
                flush=True,
            )
            pending.clear()
        if completed % args.checkpoint_interval == 0 or completed == args.training_battles:
            checkpoint = output_dir / f"battle_{completed:04d}.pt"
            if checkpoint.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint}")
            model.save_weights(checkpoint)
            checkpoints.append((completed, checkpoint))

    evaluations: list[CheckpointEvaluation] = []
    for after_battle, checkpoint in checkpoints:
        for opponent in args.eval_opponents:
            row = evaluate(
                checkpoint, server, opponent=opponent, battles=args.eval_battles,
                seed=args.eval_seed, after_battle=after_battle,
            )
            evaluations.append(row)
            print(
                f"[{args.arm} seed {args.seed}] battle {after_battle} vs {opponent}: "
                f"{row.wins}/{row.battles} ({row.win_rate:.0%})",
                flush=True,
            )
    if sha256(args.v1_checkpoint) != v1_hash:
        raise AssertionError("v1 checkpoint changed during the experiment")

    report: dict[str, object] = {
        "arm": args.arm,
        "seed": args.seed,
        "v1_checkpoint": str(args.v1_checkpoint),
        "v1_sha256": v1_hash,
        "training_opponent": args.training_opponent,
        "training_battles": args.training_battles,
        "batch_battles": BATCH_BATTLES,
        "epochs_per_batch": EPOCHS_PER_BATCH,
        "learning_rate": args.learning_rate,
        "raw_learning_rate": args.raw_learning_rate if args.arm == "hybrid" else None,
        "evaluation_seed": args.eval_seed,
        "batches": [asdict(row) for row in batches],
        "evaluations": [asdict(row) for row in evaluations],
    }
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def summarize(directories: list[Path]) -> None:
    """Print win rates per arm and checkpoint, pooled over seeds."""

    pooled: dict[tuple[str, int, str], list[int]] = {}
    for directory in directories:
        report = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        for row in report["evaluations"]:
            key = (report["arm"], row["after_battle"], row["opponent"])
            wins, battles = pooled.get(key, [0, 0])
            pooled[key] = [wins + row["wins"], battles + row["battles"]]
    print(f"{'arm':<7} {'after':>6} {'opponent':<10} {'wins':>9} {'win rate':>9}")
    for (arm, after, opponent), (wins, battles) in sorted(pooled.items()):
        print(f"{arm:<7} {after:>6} {opponent:<10} {wins:>4}/{battles:<4} {wins / battles:>9.1%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--v1-checkpoint", type=Path, default=DEFAULT_V1)
    parser.add_argument("--training-opponent", default="max-power")
    parser.add_argument("--training-battles", type=int, default=300)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--raw-learning-rate", type=float, default=1e-2)
    parser.add_argument("--eval-battles", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--eval-opponents", nargs="+", default=["max-power"])
    parser.add_argument("--output-dir", type=Path, help="new directory for checkpoints and results.json")
    parser.add_argument("--summarize", type=Path, nargs="+", help="print pooled results of finished runs")
    parser.add_argument(
        "--server-url",
        default=LocalhostServerConfiguration.websocket_url,
        help="WebSocket URL of the local Pokémon Showdown server",
    )
    args = parser.parse_args()
    if args.summarize is None and (args.arm is None or args.output_dir is None):
        parser.error("--arm and --output-dir are required unless --summarize is given")
    return args


def main() -> None:
    args = parse_args()
    if args.summarize is not None:
        summarize(args.summarize)
        return
    run(args)


if __name__ == "__main__":
    main()
