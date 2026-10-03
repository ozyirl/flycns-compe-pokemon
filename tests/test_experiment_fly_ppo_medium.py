from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiment_fly_ppo_medium import (
    CHECKPOINT_INTERVAL,
    EVALUATION_BATTLES,
    LOG_INTERVAL,
    TRAINING_BATTLES,
    CurvePoint,
    SeedExperiment,
    run_experiment,
    run_seed,
)
from flycns.actor_critic import FlyCNSActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_training import FIXED_SEEDS, PhaseMetrics
from smoke_fly_ppo_update import PPOUpdateResult
from smoke_fly_rollout import RolloutResult


class MediumFlyPPOExperimentTest(unittest.TestCase):
    def test_one_seed_runs_exact_counts_logs_every_ten_and_checkpoints_every_25(self) -> None:
        policy = FlyCNSPPOPolicy()
        environments: list[object] = []

        def make_environment() -> object:
            environment = object()
            environments.append(environment)
            return environment

        def collect(_environment: object, _policy: FlyCNSPPOPolicy, **kwargs: object) -> RolloutResult:
            self.assertIs(_policy, policy)
            return RolloutResult((None,) * 5, 2.0, "win", False, True)

        update_count = 0

        def update(_policy: FlyCNSPPOPolicy, _rollout: RolloutResult) -> PPOUpdateResult:
            nonlocal update_count
            self.assertIs(_policy, policy)
            update_count += 1
            return PPOUpdateResult(-0.2, 1.0, 0.5, 0.3, True, True, True)

        with tempfile.TemporaryDirectory() as directory:
            with patch("experiment_fly_ppo_medium.run_one_battle", side_effect=collect) as battle, patch(
                "experiment_fly_ppo_medium.update_once", side_effect=update
            ):
                result = run_seed(
                    policy, make_environment, seed=7, checkpoint_dir=Path(directory)
                )

            self.assertEqual(
                [path.name for path in result.checkpoints],
                [f"seed_7_battle_{battle_number:03d}.pt" for battle_number in (25, 50, 75, 100)],
            )
            self.assertEqual(len(result.checkpoints), TRAINING_BATTLES // CHECKPOINT_INTERVAL)
            for path in result.checkpoints:
                self.assertTrue(path.is_file())
                FlyCNSActorCritic.load_weights(path)

        self.assertEqual(battle.call_count, 2 * EVALUATION_BATTLES + TRAINING_BATTLES)
        self.assertEqual(len(environments), battle.call_count)
        self.assertEqual(
            [call.kwargs.get("deterministic", False) for call in battle.call_args_list],
            [True] * EVALUATION_BATTLES
            + [False] * TRAINING_BATTLES
            + [True] * EVALUATION_BATTLES,
        )
        self.assertEqual(
            [call.kwargs["seed"] for call in battle.call_args_list],
            list(range(7, 7 + battle.call_count)),
        )
        self.assertEqual(update_count, TRAINING_BATTLES)
        self.assertEqual(result.before.battles, EVALUATION_BATTLES)
        self.assertEqual(result.after.battles, EVALUATION_BATTLES)
        self.assertEqual(len(result.learning_curve), TRAINING_BATTLES // LOG_INTERVAL)
        self.assertEqual(
            [point.training_battles for point in result.learning_curve],
            list(range(10, TRAINING_BATTLES + 1, 10)),
        )
        for point in result.learning_curve:
            self.assertEqual(point.win_rate, 1.0)
            self.assertEqual(point.average_reward, 2.0)
            self.assertEqual(point.average_battle_length, 5.0)
            self.assertEqual(point.policy_entropy, 0.5)
            self.assertEqual(point.policy_loss, -0.2)
            self.assertEqual(point.value_loss, 1.0)

    def test_three_seeds_start_from_fresh_normalized_heads_and_aggregate_curve(self) -> None:
        policies: list[FlyCNSPPOPolicy] = []
        initial_weights: list[torch.Tensor] = []

        def fake_run(
            policy: FlyCNSPPOPolicy,
            _environment_factory: object,
            *,
            seed: int,
            checkpoint_dir: Path,
            progress: object,
        ) -> SeedExperiment:
            policies.append(policy)
            initial_weights.append(
                next(policy.actor_critic.policy_head.parameters()).detach().clone()
            )
            self.assertEqual(tuple(policy.actor_critic.input_normalization.parameters()), ())
            self.assertTrue(checkpoint_dir.is_dir())
            index = FIXED_SEEDS.index(seed)
            before = PhaseMetrics(30, 10 + index, 20 - index, 0, float(index), 50.0)
            after = PhaseMetrics(30, 12 + index, 18 - index, 0, float(index + 2), 45.0)
            curve = tuple(
                CurvePoint(battle, 0.2 + 0.1 * index, float(index), 40.0, 1.0, -0.2, 0.5)
                for battle in range(10, 101, 10)
            )
            return SeedExperiment(seed, before, after, curve, ())

        with tempfile.TemporaryDirectory() as directory:
            with patch("experiment_fly_ppo_medium.run_seed", side_effect=fake_run):
                result = run_experiment(lambda: object(), output_dir=Path(directory))

        self.assertEqual([run.seed for run in result.per_seed], list(FIXED_SEEDS))
        self.assertEqual(len({id(policy.actor_critic) for policy in policies}), 3)
        self.assertFalse(torch.equal(initial_weights[0], initial_weights[1]))
        self.assertFalse(torch.equal(initial_weights[1], initial_weights[2]))
        self.assertEqual(result.before.wins, 33)
        self.assertEqual(result.after.wins, 39)
        self.assertEqual(result.before.average_reward, 1.0)
        self.assertEqual(result.after.average_reward, 3.0)
        self.assertEqual(result.before.average_battle_length, 50.0)
        self.assertEqual(result.after.average_battle_length, 45.0)
        self.assertEqual(len(result.aggregate_curve), 10)
        self.assertAlmostEqual(result.aggregate_curve[0].win_rate, 0.3)
        self.assertAlmostEqual(result.aggregate_curve[0].average_reward, 1.0)


if __name__ == "__main__":
    unittest.main()
