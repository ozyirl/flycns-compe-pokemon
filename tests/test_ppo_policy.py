from __future__ import annotations

import unittest

import numpy as np
import torch

from encoding import OBSERVATION_DIM
from flycns.action_decoder import ACTION_COUNT
from flycns.actor_critic import FlyCNSActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy


def observation() -> np.ndarray:
    vector = np.linspace(-0.8, 1.0, OBSERVATION_DIM, dtype=np.float32)
    vector[15] = 1.0
    return vector


class FlyCNSPPOPolicyTest(unittest.TestCase):
    def test_illegal_actions_cannot_be_selected(self) -> None:
        actor_critic = FlyCNSActorCritic()
        with torch.no_grad():
            actor_critic.policy_head.linear.weight.zero_()
            actor_critic.policy_head.linear.bias.zero_()
            actor_critic.policy_head.linear.bias[5] = 1000.0  # Illegal raw winner.
            actor_critic.policy_head.linear.bias[7] = 1.0
        policy = FlyCNSPPOPolicy(actor_critic=actor_critic)
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[[2, 7]] = 1

        decision = policy.predict(observation(), mask)
        sampled = [
            policy.sample_action(observation(), mask).selected_action_index
            for _ in range(20)
        ]

        self.assertEqual(decision.selected_action_index, 7)
        self.assertTrue(set(sampled).issubset({2, 7}))
        self.assertEqual(len(decision.policy_logits), ACTION_COUNT)
        self.assertEqual(decision.policy_logits[5], float("-inf"))
        self.assertTrue(np.isfinite(decision.state_value))
        expected_log_probability = torch.log_softmax(
            torch.tensor(decision.policy_logits), dim=0
        )[decision.selected_action_index]
        self.assertAlmostEqual(decision.action_log_probability, float(expected_log_probability))

    def test_deterministic_evaluation_repeats_without_changing_weights(self) -> None:
        policy = FlyCNSPPOPolicy()
        original_parameters = [
            parameter.detach().clone() for parameter in policy.actor_critic.parameters()
        ]
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[[1, 6, 9, 22]] = 1

        first = policy.predict(observation(), mask, deterministic=True)
        second = policy.predict(observation(), mask, deterministic=True)

        self.assertEqual(first, second)
        self.assertIn(first.selected_action_index, {1, 6, 9, 22})
        for parameter, original in zip(
            policy.actor_critic.parameters(), original_parameters, strict=True
        ):
            self.assertTrue(torch.equal(parameter, original))
            self.assertIsNone(parameter.grad)

    def test_empty_action_mask_is_rejected(self) -> None:
        policy = FlyCNSPPOPolicy()

        with self.assertRaisesRegex(ValueError, "at least one legal action"):
            policy.predict(observation(), np.zeros(ACTION_COUNT, dtype=np.int8))

    def test_batch_evaluates_legal_actions_without_changing_weights(self) -> None:
        policy = FlyCNSPPOPolicy()
        original_parameters = [
            parameter.detach().clone() for parameter in policy.actor_critic.parameters()
        ]
        observations = np.stack((observation(), observation()[::-1].copy(), observation()))
        masks = np.zeros((3, ACTION_COUNT), dtype=np.int8)
        masks[0, [2, 7]] = 1
        masks[1, 4] = 1
        masks[2, [1, 9, 22]] = 1
        selected = np.asarray([7, 4, 22], dtype=np.int64)

        result = policy.evaluate_batch(observations, masks, selected)

        self.assertEqual(result.action_log_probabilities.shape, (3,))
        self.assertEqual(result.entropies.shape, (3,))
        self.assertEqual(result.state_values.shape, (3,))
        self.assertTrue(np.isfinite(result.action_log_probabilities).all())
        self.assertTrue(np.isfinite(result.entropies).all())
        self.assertTrue(np.isfinite(result.state_values).all())
        self.assertAlmostEqual(float(result.action_log_probabilities[1]), 0.0)
        self.assertAlmostEqual(float(result.entropies[1]), 0.0)
        self.assertEqual(result.state_values[0], result.state_values[2])
        for parameter, original in zip(
            policy.actor_critic.parameters(), original_parameters, strict=True
        ):
            self.assertTrue(torch.equal(parameter, original))
            self.assertIsNone(parameter.grad)

    def test_batch_rejects_illegal_selected_action(self) -> None:
        policy = FlyCNSPPOPolicy()
        masks = np.zeros((2, ACTION_COUNT), dtype=np.int8)
        masks[:, 7] = 1

        with self.assertRaisesRegex(ValueError, "must be legal"):
            policy.evaluate_batch(
                np.stack((observation(), observation())),
                masks,
                [7, 5],
            )


if __name__ == "__main__":
    unittest.main()
