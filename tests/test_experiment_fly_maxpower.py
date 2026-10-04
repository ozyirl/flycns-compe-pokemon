from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiment_fly_maxpower import TRAINING_BATTLES, train_heads


class MaxPowerContinuationTest(unittest.TestCase):
    def test_uses_one_existing_ppo_update_per_battle(self) -> None:
        created: list[object] = []
        observed_seeds: list[int] = []
        updates: list[object] = []

        def environment_factory() -> object:
            environment = object()
            created.append(environment)
            return environment

        def rollout(environment: object, policy: object, *, seed: int) -> object:
            self.assertIs(environment, created[-1])
            self.assertIs(policy, policy_object)
            observed_seeds.append(seed)
            return SimpleNamespace(
                outcome="win" if seed % 2 else "loss",
                total_reward=float(seed),
                turns=(object(), object()),
                illegal_action_attempted=False,
            )

        def update(policy: object, result: object) -> None:
            self.assertIs(policy, policy_object)
            updates.append(result)

        policy_object = object()
        with (
            patch("experiment_fly_maxpower.run_one_battle", side_effect=rollout),
            patch("experiment_fly_maxpower.update_once", side_effect=update),
        ):
            result = train_heads(
                policy_object, environment_factory, seed=7, battles=TRAINING_BATTLES
            )

        self.assertEqual(len(created), 100)
        self.assertEqual(len(updates), 100)
        self.assertEqual(observed_seeds, list(range(7, 107)))
        self.assertEqual(result.battles, 100)
        self.assertEqual(result.updates, 100)
        self.assertEqual((result.wins, result.losses, result.ties), (50, 50, 0))
        self.assertEqual(result.average_battle_length, 2.0)
        self.assertEqual(result.illegal_actions, 0)


if __name__ == "__main__":
    unittest.main()
