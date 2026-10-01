from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from encoding import OBSERVATION_DIM
from flycns.action_decoder import ACTION_COUNT
from flycns.ppo_policy import PolicyDecision
from smoke_fly_rollout import run_one_battle


class FakeEnvironment:
    def __init__(self) -> None:
        self.env = SimpleNamespace(battle1=SimpleNamespace(won=True, lost=False))
        self.reset_calls = 0
        self.step_calls = 0
        self.closed = False
        self.observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        self.mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        self.mask[7] = 1

    def reset(self, *, seed: int) -> tuple[dict[str, np.ndarray], dict]:
        self.reset_calls += 1
        return {"observation": self.observation, "action_mask": self.mask}, {}

    def step(self, action: np.int64) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        self.step_calls += 1
        assert isinstance(action, np.int64)
        assert action == 7
        self.observation[0] = float(self.step_calls)
        return (
            {"observation": self.observation, "action_mask": self.mask},
            1.5,
            self.step_calls == 2,
            False,
            {},
        )

    def close(self) -> None:
        self.closed = True


class FakePolicy:
    def __init__(self) -> None:
        self.actor_critic = torch.nn.Linear(1, 1)

    def sample_action(self, observation: np.ndarray, action_mask: np.ndarray) -> PolicyDecision:
        return PolicyDecision(7, -0.5, 0.25, tuple([float("-inf")] * ACTION_COUNT))


class FlyRolloutSmokeTest(unittest.TestCase):
    def test_records_one_episode_without_weight_updates(self) -> None:
        environment = FakeEnvironment()
        policy = FakePolicy()

        result = run_one_battle(environment, policy, seed=7)

        self.assertEqual(environment.reset_calls, 1)
        self.assertEqual(environment.step_calls, 2)
        self.assertTrue(environment.closed)
        self.assertEqual(len(result.turns), 2)
        self.assertEqual(result.total_reward, 3.0)
        self.assertEqual(result.outcome, "win")
        self.assertFalse(result.illegal_action_attempted)
        self.assertTrue(result.actor_critic_weights_unchanged)
        self.assertEqual([turn.done for turn in result.turns], [False, True])
        self.assertEqual(result.turns[0].observation.shape, (OBSERVATION_DIM,))
        self.assertEqual(result.turns[0].action_mask.shape, (ACTION_COUNT,))
        self.assertEqual(result.turns[0].selected_action, 7)
        self.assertEqual(result.turns[0].log_probability, -0.5)
        self.assertEqual(result.turns[0].state_value, 0.25)
        self.assertEqual(result.turns[0].reward, 1.5)
        self.assertEqual(result.turns[0].observation[0], 0.0)


if __name__ == "__main__":
    unittest.main()
