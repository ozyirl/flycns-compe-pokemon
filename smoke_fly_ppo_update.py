"""Collect one Fly CNS battle rollout, then make exactly one heads-only PPO update."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration
from torch.distributions import Categorical

from env import DEFAULT_BATTLE_FORMAT, ShowdownEnv, make_opponent
from flycns.action_decoder import ACTION_COUNT
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_rollout import RolloutResult, run_one_battle


@dataclass(frozen=True)
class PPOUpdateResult:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    actor_changed: bool
    critic_changed: bool
    fly_output_unchanged: bool


ADVANTAGE_NORMALIZATION_EPSILON = 1e-8


def normalized_policy_advantages(advantages: np.ndarray) -> np.ndarray:
    """Normalize only the policy-loss copy of a collected rollout batch."""

    raw = np.asarray(advantages, dtype=np.float32)
    if raw.ndim != 1 or not raw.size or not np.isfinite(raw).all():
        raise ValueError("Advantages must be a nonempty finite vector")
    mean = float(np.mean(raw, dtype=np.float64))
    std = float(np.std(raw, dtype=np.float64))
    return ((raw.astype(np.float64) - mean) / (std + ADVANTAGE_NORMALIZATION_EPSILON)).astype(np.float32)


def advantages_and_returns(
    rollout: RolloutResult, *, gamma: float = 0.99, gae_lambda: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Compute terminal-aware GAE from the values saved during the rollout."""

    if not rollout.turns or not rollout.turns[-1].done:
        raise ValueError("PPO update requires one completed rollout")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")

    rewards = np.asarray([turn.reward for turn in rollout.turns], dtype=np.float64)
    values = np.asarray([turn.state_value for turn in rollout.turns], dtype=np.float64)
    if not np.isfinite(rewards).all() or not np.isfinite(values).all():
        raise ValueError("Rollout rewards and values must be finite")

    advantages = np.zeros(len(rollout.turns), dtype=np.float64)
    next_advantage = 0.0
    for index in reversed(range(len(rollout.turns))):
        not_done = 0.0 if rollout.turns[index].done else 1.0
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        delta = rewards[index] + gamma * next_value * not_done - values[index]
        next_advantage = delta + gamma * gae_lambda * not_done * next_advantage
        advantages[index] = next_advantage
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


def update_once(
    policy: FlyCNSPPOPolicy,
    rollout: RolloutResult,
    *,
    learning_rate: float = 3e-4,
    clip_range: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    normalize_advantages: bool = False,
) -> PPOUpdateResult:
    """Make one PPO optimizer step on the actor and critic heads only."""

    if not rollout.actor_critic_weights_unchanged:
        raise ValueError("Rollout weights changed before the PPO update")
    if learning_rate <= 0 or clip_range <= 0:
        raise ValueError("learning_rate and clip_range must be positive")
    advantages, returns = advantages_and_returns(rollout)
    observations = np.stack([turn.observation for turn in rollout.turns])
    masks = policy._validated_masks(
        np.stack([turn.action_mask for turn in rollout.turns]),
        (len(rollout.turns), ACTION_COUNT),
    )
    actions = np.asarray([turn.selected_action for turn in rollout.turns], dtype=np.int64)
    if np.any((actions < 0) | (actions >= ACTION_COUNT)) or not np.all(
        masks[np.arange(len(actions)), actions]
    ):
        raise ValueError("Rollout contains an illegal selected action")
    old_log_probabilities = np.asarray(
        [turn.log_probability for turn in rollout.turns], dtype=np.float32
    )
    if not np.isfinite(old_log_probabilities).all():
        raise ValueError("Rollout log probabilities must be finite")

    # The sensory adapter and spike simulator are fixed NumPy components. Their
    # outputs are features only; autograd sees only the two trainable heads.
    descending = np.stack([policy._descending_activity(row) for row in observations])
    model = policy.actor_critic
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    activity = torch.as_tensor(descending, device=device, dtype=dtype)
    legal = torch.as_tensor(masks, device=device, dtype=torch.bool)
    selected = torch.as_tensor(actions, device=device, dtype=torch.long)
    old_log_probs = torch.as_tensor(old_log_probabilities, device=device, dtype=dtype)
    policy_advantages = (
        normalized_policy_advantages(advantages) if normalize_advantages else advantages
    )
    advantage_tensor = torch.as_tensor(policy_advantages, device=device, dtype=dtype)
    return_tensor = torch.as_tensor(returns, device=device, dtype=dtype)

    actor_parameters = tuple(model.policy_head.parameters())
    critic_parameters = tuple(model.value_head.parameters())
    actor_before = [item.detach().clone() for item in actor_parameters]
    critic_before = [item.detach().clone() for item in critic_parameters]
    optimizer = torch.optim.Adam(
        [*actor_parameters, *critic_parameters], lr=learning_rate
    )
    optimizer.zero_grad(set_to_none=True)
    logits, values = model(activity)
    distribution = Categorical(logits=logits.masked_fill(~legal, float("-inf")))
    new_log_probs = distribution.log_prob(selected)
    ratio = torch.exp(new_log_probs - old_log_probs)
    policy_loss = -torch.minimum(
        ratio * advantage_tensor,
        ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantage_tensor,
    ).mean()
    value_loss = 0.5 * torch.square(values.squeeze(-1) - return_tensor).mean()
    entropy = distribution.entropy().mean()
    total_loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    if not all(
        torch.isfinite(item).item()
        for item in (policy_loss, value_loss, entropy, total_loss)
    ):
        raise FloatingPointError("PPO update produced a non-finite loss")
    total_loss.backward()
    if not all(
        item.grad is not None and torch.isfinite(item.grad).all().item()
        for item in (*actor_parameters, *critic_parameters)
    ):
        raise FloatingPointError("PPO update produced a non-finite or missing gradient")
    optimizer.step()

    actor_changed = any(
        not torch.equal(item, before)
        for item, before in zip(actor_parameters, actor_before, strict=True)
    )
    critic_changed = any(
        not torch.equal(item, before)
        for item, before in zip(critic_parameters, critic_before, strict=True)
    )
    fly_output_unchanged = np.array_equal(
        descending[0], policy._descending_activity(observations[0])
    )
    if not actor_changed or not critic_changed or not fly_output_unchanged:
        raise AssertionError("PPO update failed the heads-only/Fly CNS invariants")
    return PPOUpdateResult(
        policy_loss=float(policy_loss.item()),
        value_loss=float(value_loss.item()),
        entropy=float(entropy.item()),
        total_loss=float(total_loss.item()),
        actor_changed=actor_changed,
        critic_changed=critic_changed,
        fly_output_unchanged=fly_output_unchanged,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-url",
        default=LocalhostServerConfiguration.websocket_url,
        help="WebSocket URL of the already-running local Showdown server",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
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
    rollout = run_one_battle(environment, policy, seed=args.seed)
    result = update_once(policy, rollout)

    print(f"rollout turns: {len(rollout.turns)}")
    print(f"policy loss: {result.policy_loss:.6f}")
    print(f"value loss: {result.value_loss:.6f}")
    print(f"entropy: {result.entropy:.6f}")
    print(f"total loss: {result.total_loss:.6f}")
    print(f"actor parameters changed: {'yes' if result.actor_changed else 'no'}")
    print(f"critic parameters changed: {'yes' if result.critic_changed else 'no'}")
    print(f"Fly CNS output unchanged: {'yes' if result.fly_output_unchanged else 'no'}")


if __name__ == "__main__":
    main()
