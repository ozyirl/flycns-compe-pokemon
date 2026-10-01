from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from flycns.action_decoder import ACTION_COUNT
from flycns.actor_critic import FlyCNSActorCritic
from flycns.trainable_decoder import DESCENDING_ACTIVITY_SIZE, TrainableActionDecoder


class FlyCNSActorCriticTest(unittest.TestCase):
    def test_single_activity_vector_produces_logits_and_value(self) -> None:
        model = FlyCNSActorCritic()

        policy_logits, state_value = model(torch.zeros(DESCENDING_ACTIVITY_SIZE))

        self.assertIsInstance(model.policy_head, TrainableActionDecoder)
        self.assertEqual(tuple(policy_logits.shape), (ACTION_COUNT,))
        self.assertEqual(tuple(state_value.shape), (1,))

    def test_batched_activity_vectors_work(self) -> None:
        model = FlyCNSActorCritic()
        activity = torch.zeros((4, DESCENDING_ACTIVITY_SIZE))

        policy_logits, state_value = model(activity)

        self.assertEqual(tuple(policy_logits.shape), (4, ACTION_COUNT))
        self.assertEqual(tuple(state_value.shape), (4, 1))

    def test_gradients_reach_policy_and_value_parameters(self) -> None:
        model = FlyCNSActorCritic()
        activity = torch.linspace(-1.0, 1.0, DESCENDING_ACTIVITY_SIZE)

        policy_logits, state_value = model(activity)
        (policy_logits.sum() + state_value.sum()).backward()

        for head in (model.policy_head, model.value_head):
            for parameter in head.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

    def test_saved_weights_reproduce_both_outputs(self) -> None:
        model = FlyCNSActorCritic()
        activity = torch.linspace(-1.0, 1.0, DESCENDING_ACTIVITY_SIZE)
        expected_logits, expected_value = model(activity)

        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "actor_critic.pt"
            model.save_weights(weights_path)
            reloaded = FlyCNSActorCritic.load_weights(weights_path)

        actual_logits, actual_value = reloaded(activity)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_value, expected_value, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
