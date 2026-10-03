from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from diagnose_fly_policy_scale import inspect_turn, run_diagnostic, summarize_turns
from encoding import OBSERVATION_DIM
from flycns.action_decoder import ACTION_COUNT
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_rollout import RolloutResult, TurnRecord


def recorded_turn(mask: np.ndarray) -> TurnRecord:
    observation = np.linspace(-0.8, 1.0, OBSERVATION_DIM, dtype=np.float32)
    observation[15] = 1.0
    return TurnRecord(observation, mask, 7, 0.0, 0.0, 0.0, True)


class FlyPolicyScaleDiagnosticTest(unittest.TestCase):
    def test_inspection_uses_only_legal_logits_and_keeps_weights_fixed(self) -> None:
        policy = FlyCNSPPOPolicy()
        with torch.no_grad():
            policy.actor_critic.policy_head.linear.bias[5] = 1000.0
        before = [parameter.detach().clone() for parameter in policy.actor_critic.parameters()]
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[[2, 7, 11]] = 1

        scale = inspect_turn(policy, recorded_turn(mask))

        self.assertEqual(scale.legal_actions, 3)
        self.assertLess(scale.legal_logit_maximum, 1000.0)
        self.assertGreaterEqual(scale.spike_minimum, 0.0)
        self.assertLessEqual(scale.spike_minimum, scale.spike_mean)
        self.assertLessEqual(scale.spike_mean, scale.spike_maximum)
        self.assertGreaterEqual(scale.spike_l2_norm, 0.0)
        self.assertGreater(scale.policy_entropy, 0.0)
        self.assertGreaterEqual(scale.highest_legal_probability, 1.0 / 3.0)
        self.assertLessEqual(scale.highest_legal_probability, 1.0)
        for parameter, original in zip(policy.actor_critic.parameters(), before, strict=True):
            self.assertTrue(torch.equal(parameter, original))
            self.assertIsNone(parameter.grad)

    def test_one_legal_action_has_zero_entropy_and_probability_one(self) -> None:
        policy = FlyCNSPPOPolicy()
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[7] = 1

        scale = inspect_turn(policy, recorded_turn(mask))

        self.assertEqual(scale.legal_actions, 1)
        self.assertEqual(scale.legal_logit_minimum, scale.legal_logit_maximum)
        self.assertAlmostEqual(scale.policy_entropy, 0.0)
        self.assertEqual(scale.highest_legal_probability, 1.0)

    def test_five_rollouts_are_aggregated_without_training(self) -> None:
        policy = FlyCNSPPOPolicy()
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[[2, 7, 11]] = 1
        turn = recorded_turn(mask)
        rollout = RolloutResult((turn, turn), 0.0, "loss", False, True)
        environments: list[object] = []

        def make_environment() -> object:
            environment = object()
            environments.append(environment)
            return environment

        with patch("diagnose_fly_policy_scale.run_one_battle", return_value=rollout) as collect:
            result = run_diagnostic(policy, make_environment, battles=5, seed=13)

        self.assertEqual(collect.call_count, 5)
        self.assertEqual([call.kwargs["seed"] for call in collect.call_args_list], [13, 14, 15, 16, 17])
        self.assertEqual([call.args[0] for call in collect.call_args_list], environments)
        self.assertEqual(result.battles, 5)
        self.assertEqual(len(result.turns), 10)
        self.assertTrue(result.weights_unchanged)
        self.assertEqual(result.summaries, summarize_turns(result.turns))
        for summary in result.summaries.values():
            self.assertTrue(np.isfinite([summary.mean, summary.minimum, summary.maximum]).all())
            self.assertLessEqual(summary.minimum, summary.mean)
            self.assertLessEqual(summary.mean, summary.maximum)


if __name__ == "__main__":
    unittest.main()
