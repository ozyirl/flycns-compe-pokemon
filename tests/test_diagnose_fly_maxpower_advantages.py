from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from diagnose_fly_maxpower_advantages import (
    inspect_rollout,
    run_diagnostic,
    summarize_outcome,
)
from encoding import OBSERVATION_DIM
from flycns.action_decoder import ACTION_COUNT
from flycns.ppo_policy import PolicyDecision
from smoke_fly_ppo_update import advantages_and_returns
from smoke_fly_rollout import RolloutResult, TurnRecord


class FakePolicy:
    def __init__(self) -> None:
        self.actor_critic = torch.nn.Linear(1, 1)
        self.predict_calls = 0

    def predict(
        self, observation: np.ndarray, action_mask: np.ndarray, *, deterministic: bool
    ) -> PolicyDecision:
        assert deterministic
        self.predict_calls += 1
        logits = [float("-inf")] * ACTION_COUNT
        logits[7] = logits[8] = 0.0
        return PolicyDecision(7, -math.log(2), 0.0, tuple(logits))


def sample_rollout(outcome: str = "win") -> RolloutResult:
    observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    mask = np.zeros(ACTION_COUNT, dtype=np.int8)
    mask[[7, 8]] = 1
    turns = (
        TurnRecord(observation, mask, 7, -math.log(2), 0.2, 1.0, False),
        TurnRecord(observation, mask, 7, -math.log(2), 0.4, 2.0, True),
    )
    return RolloutResult(turns, 3.0, outcome, False, True)


class FlyMaxPowerAdvantageDiagnosticTest(unittest.TestCase):
    def test_turn_rows_match_existing_gae_and_frozen_action_distribution(self) -> None:
        policy = FakePolicy()
        rollout = sample_rollout()
        advantages, returns = advantages_and_returns(rollout)

        battle = inspect_rollout(policy, rollout, battle_index=3)

        self.assertEqual(policy.predict_calls, 2)
        self.assertEqual((battle.battle, battle.outcome, battle.total_reward), (3, "win", 3.0))
        self.assertEqual(len(battle.turns), 2)
        for index, row in enumerate(battle.turns):
            self.assertEqual(row.turn, index + 1)
            self.assertEqual(row.selected_action, 7)
            self.assertAlmostEqual(row.reward, rollout.turns[index].reward)
            self.assertAlmostEqual(row.critic_value, rollout.turns[index].state_value)
            self.assertAlmostEqual(row.computed_return, returns[index])
            self.assertAlmostEqual(row.ppo_advantage, advantages[index])
            self.assertAlmostEqual(row.critic_prediction_error, row.critic_value - row.computed_return)
            self.assertAlmostEqual(row.policy_entropy, math.log(2))
            self.assertAlmostEqual(row.selected_action_probability, 0.5)

    def test_wins_and_losses_get_separate_battle_and_turn_averages(self) -> None:
        policy = FakePolicy()
        win = inspect_rollout(policy, sample_rollout("win"), battle_index=1)
        loss = inspect_rollout(policy, sample_rollout("loss"), battle_index=2)

        winning = summarize_outcome((win,))
        losing = summarize_outcome((loss,))

        self.assertEqual((winning.battles, winning.turns), (1, 2))
        self.assertEqual((losing.battles, losing.turns), (1, 2))
        self.assertEqual(winning.average_total_reward, 3.0)
        self.assertEqual(losing.average_total_reward, 3.0)
        self.assertGreater(winning.average_absolute_advantage, 0.0)
        self.assertGreater(winning.average_absolute_critic_prediction_error, 0.0)
        self.assertAlmostEqual(winning.average_policy_entropy, math.log(2))
        self.assertEqual(winning.action_frequency["7"], {"count": 2, "fraction": 1.0})
        self.assertEqual(winning.action_frequency["8"], {"count": 0, "fraction": 0.0})

    def test_rollout_collection_never_changes_weights(self) -> None:
        policy = FakePolicy()
        original = [parameter.detach().clone() for parameter in policy.actor_critic.parameters()]
        with patch(
            "diagnose_fly_maxpower_advantages.run_one_battle",
            side_effect=[sample_rollout("win"), sample_rollout("loss")],
        ) as collect:
            battles = run_diagnostic(policy, object, battles=2, seed=7)

        self.assertEqual(len(battles), 2)
        self.assertEqual([call.kwargs["seed"] for call in collect.call_args_list], [7, 8])
        for parameter, before in zip(policy.actor_critic.parameters(), original, strict=True):
            torch.testing.assert_close(parameter, before, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
