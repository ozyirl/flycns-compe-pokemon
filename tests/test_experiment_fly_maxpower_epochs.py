from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiment_fly_maxpower_epochs import (
    EPOCHS_PER_BATCH,
    EXPECTED_OPTIMIZER_STEPS,
    policy_shift_metrics,
    train_multi_epoch,
)
from flycns.ppo_policy import BatchPolicyEvaluation
from smoke_fly_ppo_update import PPOUpdateResult
from smoke_fly_rollout import RolloutResult, TurnRecord


def one_turn_rollout(reward: float) -> RolloutResult:
    turn = TurnRecord(
        observation=np.zeros(186, dtype=np.float32),
        action_mask=np.ones(26, dtype=np.int8),
        selected_action=7,
        log_probability=-1.0,
        state_value=0.25,
        reward=reward,
        done=True,
    )
    return RolloutResult((turn,), reward, "win", False, True)


class MultiEpochMaxPowerExperimentTest(unittest.TestCase):
    def test_post_step_clip_fraction_and_approximate_kl(self) -> None:
        rollout = one_turn_rollout(1.0)
        new_log_probability = -1.0 + math.log(1.5)
        policy = SimpleNamespace(
            evaluate_batch=lambda observations, masks, actions: BatchPolicyEvaluation(
                action_log_probabilities=np.asarray([new_log_probability]),
                entropies=np.asarray([0.5]),
                state_values=np.asarray([0.0]),
            )
        )

        clip_fraction, approximate_kl = policy_shift_metrics(policy, rollout)

        self.assertEqual(clip_fraction, 1.0)
        self.assertAlmostEqual(approximate_kl, 0.5 - math.log(1.5))

    def test_same_rollout_batch_gets_eight_epochs_and_104_total_steps(self) -> None:
        rollout_seeds: list[int] = []
        update_batch_sizes: list[int] = []
        update_rollout_ids: list[int] = []

        def collect(_environment: object, _policy: object, *, seed: int) -> RolloutResult:
            rollout_seeds.append(seed)
            return one_turn_rollout(float(seed))

        def update(_policy: object, rollout: RolloutResult) -> PPOUpdateResult:
            update_batch_sizes.append(len(rollout.turns))
            update_rollout_ids.append(id(rollout))
            return PPOUpdateResult(0.1, 2.0, 1.0, 1.09, True, True, True)

        class FakeHeads:
            def save_weights(self, path: Path) -> None:
                path.write_bytes(f"after-step-{len(update_batch_sizes)}".encode())

        policy = SimpleNamespace(actor_critic=FakeHeads())
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("experiment_fly_maxpower_epochs.run_one_battle", side_effect=collect),
                patch("experiment_fly_maxpower_epochs.update_once", side_effect=update),
                patch("experiment_fly_maxpower_epochs.policy_shift_metrics", return_value=(0.2, 0.03)),
            ):
                steps, checkpoints = train_multi_epoch(
                    policy, object, output_dir=Path(directory), seed=7
                )

        self.assertEqual(rollout_seeds, list(range(7, 107)))
        self.assertEqual(EPOCHS_PER_BATCH, 8)
        self.assertEqual(EXPECTED_OPTIMIZER_STEPS, 104)
        self.assertEqual(len(steps), 104)
        self.assertEqual(update_batch_sizes, [8] * 96 + [4] * 8)
        self.assertEqual(
            [step.epoch for step in steps],
            list(range(1, 9)) * 13,
        )
        self.assertTrue(all(
            len(set(update_rollout_ids[start:start + 8])) == 1
            for start in range(0, 104, 8)
        ))
        self.assertEqual([item.after_battle for item in checkpoints], [20, 40, 60, 80, 100])
        self.assertEqual(
            [item.updated_through_battle for item in checkpoints], [16, 40, 56, 80, 100]
        )
        self.assertEqual([item.pending_rollouts for item in checkpoints], [4, 0, 4, 0, 0])
        self.assertEqual([item.optimizer_steps_total for item in checkpoints], [16, 40, 56, 80, 104])
        self.assertEqual([item.optimizer_steps_in_window for item in checkpoints], [16, 24, 16, 24, 24])
        self.assertTrue(all(item.value_loss == 2.0 for item in checkpoints))
        for item in checkpoints:
            self.assertAlmostEqual(item.policy_loss, 0.1)
            self.assertAlmostEqual(item.clip_fraction, 0.2)
            self.assertAlmostEqual(item.approximate_kl, 0.03)


if __name__ == "__main__":
    unittest.main()
