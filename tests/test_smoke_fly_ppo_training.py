from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from flycns.actor_critic import FlyCNSActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_training import (
    BATTLES_PER_PHASE,
    FIXED_SEEDS,
    PhaseMetrics,
    TrainingSmokeResult,
    run_smoke,
    run_three_seed_smoke,
)
from smoke_fly_ppo_update import PPOUpdateResult
from smoke_fly_rollout import RolloutResult


class FlyPPOTrainingSmokeTest(unittest.TestCase):
    def test_three_fixed_seeds_start_from_fresh_reproducible_heads(self) -> None:
        starting_weights: list[torch.Tensor] = []
        policies: list[FlyCNSPPOPolicy] = []
        seen_seeds: list[int] = []

        def fake_run(
            policy: FlyCNSPPOPolicy,
            _environment_factory: object,
            *,
            checkpoint_path: Path,
            seed: int,
        ) -> TrainingSmokeResult:
            policies.append(policy)
            seen_seeds.append(seed)
            starting_weights.append(
                next(policy.actor_critic.policy_head.parameters()).detach().clone()
            )
            self.assertEqual(checkpoint_path.name, f"actor_critic_seed_{seed}.pt")
            index = FIXED_SEEDS.index(seed)
            before = PhaseMetrics(10, 4 + index, 6 - index, 0, float(index), 50.0 + 10 * index)
            after = PhaseMetrics(10, 5 + index, 5 - index, 0, float(index + 2), 45.0 + 10 * index)
            return TrainingSmokeResult(before, before, after, (None,) * 10, checkpoint_path)

        with tempfile.TemporaryDirectory() as directory:
            with patch("smoke_fly_ppo_training.run_smoke", side_effect=fake_run):
                result = run_three_seed_smoke(lambda: object(), checkpoint_dir=directory)
                repeat = run_three_seed_smoke(lambda: object(), checkpoint_dir=directory)

        self.assertEqual(seen_seeds, list(FIXED_SEEDS) * 2)
        self.assertEqual(len({id(policy.actor_critic) for policy in policies}), 6)
        self.assertFalse(torch.equal(starting_weights[0], starting_weights[1]))
        self.assertFalse(torch.equal(starting_weights[1], starting_weights[2]))
        for first, second in zip(starting_weights[:3], starting_weights[3:], strict=True):
            self.assertTrue(torch.equal(first, second))
        self.assertEqual(result.before.wins, 15)
        self.assertEqual(result.after.wins, 18)
        self.assertEqual(result.before.battles, 30)
        self.assertEqual(result.after.battles, 30)
        self.assertEqual(result.before.average_reward, 1.0)
        self.assertEqual(result.after.average_reward, 3.0)
        self.assertEqual(result.before.average_battle_length, 60.0)
        self.assertEqual(result.after.average_battle_length, 55.0)
        self.assertEqual(result.mean_win_improvement, 1.0)
        self.assertEqual(result.mean_reward_improvement, 2.0)
        self.assertEqual(result.mean_battle_length_change, -5.0)
        self.assertEqual(repeat.before, result.before)
        self.assertEqual(repeat.after, result.after)

    def test_exactly_ten_battles_per_phase_and_ten_updates(self) -> None:
        policy = FlyCNSPPOPolicy()
        environments: list[object] = []

        def make_environment() -> object:
            environment = object()
            environments.append(environment)
            return environment

        collected: list[tuple[object, bool]] = []

        def collect(environment: object, _policy: FlyCNSPPOPolicy, **kwargs: object) -> RolloutResult:
            self.assertIs(_policy, policy)
            collected.append((environment, bool(kwargs.get("deterministic", False))))
            index = len(collected) - 1
            return RolloutResult(
                turns=(None,) * 4,  # Only episode length is used by this orchestration test.
                total_reward=float(index),
                outcome="win" if index % 2 == 0 else "loss",
                illegal_action_attempted=False,
                actor_critic_weights_unchanged=True,
            )

        update_count = 0

        def update(_policy: FlyCNSPPOPolicy, _rollout: RolloutResult) -> PPOUpdateResult:
            nonlocal update_count
            self.assertIs(_policy, policy)
            update_count += 1
            with torch.no_grad():
                _policy.actor_critic.policy_head.linear.bias.add_(0.001)
                _policy.actor_critic.value_head.bias.add_(0.001)
            return PPOUpdateResult(0.0, 0.0, 0.0, 0.0, True, True, True)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "actor_critic.pt"
            with patch("smoke_fly_ppo_training.run_one_battle", side_effect=collect), patch(
                "smoke_fly_ppo_training.update_once", side_effect=update
            ):
                result = run_smoke(
                    policy, make_environment, checkpoint_path=checkpoint, seed=7
                )

            self.assertTrue(checkpoint.is_file())
            loaded = FlyCNSActorCritic.load_weights(checkpoint)
            for actual, saved in zip(
                policy.actor_critic.parameters(), loaded.parameters(), strict=True
            ):
                self.assertTrue(torch.equal(actual, saved))

        self.assertEqual(len(environments), 3 * BATTLES_PER_PHASE)
        self.assertEqual([item[0] for item in collected], environments)
        self.assertEqual(
            [item[1] for item in collected],
            [True] * 10 + [False] * 10 + [True] * 10,
        )
        self.assertEqual(update_count, BATTLES_PER_PHASE)
        self.assertEqual(len(result.updates), BATTLES_PER_PHASE)
        self.assertEqual((result.before.battles, result.training.battles, result.after.battles), (10, 10, 10))
        self.assertEqual((result.before.wins, result.before.losses), (5, 5))
        self.assertEqual(result.before.average_reward, 4.5)
        self.assertEqual(result.training.average_reward, 14.5)
        self.assertEqual(result.after.average_reward, 24.5)
        self.assertEqual(result.after.average_battle_length, 4.0)


if __name__ == "__main__":
    unittest.main()
