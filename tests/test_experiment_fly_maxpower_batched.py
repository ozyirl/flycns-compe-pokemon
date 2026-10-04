from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiment_fly_maxpower_batched import combine_rollouts, train_batched
from smoke_fly_ppo_update import PPOUpdateResult, advantages_and_returns
from smoke_fly_rollout import RolloutResult, TurnRecord


def one_turn_rollout(reward: float, value: float = 0.25) -> RolloutResult:
    turn = TurnRecord(
        observation=np.zeros(186, dtype=np.float32),
        action_mask=np.ones(26, dtype=np.int8),
        selected_action=7,
        log_probability=-1.0,
        state_value=value,
        reward=reward,
        done=True,
    )
    return RolloutResult((turn,), reward, "win", False, True)


class BatchedMaxPowerExperimentTest(unittest.TestCase):
    def test_gae_resets_at_every_concatenated_battle_boundary(self) -> None:
        first = one_turn_rollout(2.0, value=0.5)
        second = one_turn_rollout(-3.0, value=-0.25)
        combined = combine_rollouts((first, second))

        advantages, returns = advantages_and_returns(combined)
        first_advantages, first_returns = advantages_and_returns(first)
        second_advantages, second_returns = advantages_and_returns(second)

        self.assertEqual(len(combined.turns), 2)
        self.assertEqual([turn.done for turn in combined.turns], [True, True])
        np.testing.assert_array_equal(
            advantages, np.concatenate((first_advantages, second_advantages))
        )
        np.testing.assert_array_equal(
            returns, np.concatenate((first_returns, second_returns))
        )

    def test_exactly_thirteen_updates_and_five_unambiguous_checkpoints(self) -> None:
        rollout_seeds: list[int] = []
        update_batch_sizes: list[int] = []

        def collect(_environment: object, _policy: object, *, seed: int) -> RolloutResult:
            rollout_seeds.append(seed)
            return one_turn_rollout(float(seed))

        def update(_policy: object, rollout: RolloutResult) -> PPOUpdateResult:
            update_batch_sizes.append(len(rollout.turns))
            self.assertTrue(all(turn.done for turn in rollout.turns))
            return PPOUpdateResult(0.1, 2.0, 1.0, 1.09, True, True, True)

        class FakeHeads:
            def save_weights(self, path: Path) -> None:
                path.write_bytes(f"after-update-{len(update_batch_sizes)}".encode())

        policy = SimpleNamespace(actor_critic=FakeHeads())
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("experiment_fly_maxpower_batched.run_one_battle", side_effect=collect),
                patch("experiment_fly_maxpower_batched.update_once", side_effect=update),
            ):
                updates, checkpoints = train_batched(
                    policy, object, output_dir=Path(directory), seed=7
                )

        self.assertEqual(rollout_seeds, list(range(7, 107)))
        self.assertEqual(update_batch_sizes, [8] * 12 + [4])
        self.assertEqual(len(updates), 13)
        self.assertEqual([item.after_battle for item in checkpoints], [20, 40, 60, 80, 100])
        self.assertEqual(
            [item.updated_through_battle for item in checkpoints],
            [16, 40, 56, 80, 100],
        )
        self.assertEqual([item.pending_rollouts for item in checkpoints], [4, 0, 4, 0, 0])
        self.assertEqual([item.updates_in_window for item in checkpoints], [2, 3, 2, 3, 3])
        self.assertEqual([item.checkpoint.name for item in checkpoints], [
            "battle_020.pt", "battle_040.pt", "battle_060.pt", "battle_080.pt", "battle_100.pt"
        ])
        self.assertTrue(all(np.isfinite(item.advantage_std) for item in checkpoints))
        self.assertTrue(all(item.policy_entropy == 1.0 for item in checkpoints))
        self.assertTrue(all(item.value_loss == 2.0 for item in checkpoints))


if __name__ == "__main__":
    unittest.main()
