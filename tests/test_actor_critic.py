from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from flycns.action_decoder import ACTION_COUNT
from flycns.actor_critic import FlyCNSActorCritic
from flycns.trainable_decoder import DESCENDING_ACTIVITY_SIZE, TrainableActionDecoder


class FlyCNSActorCriticTest(unittest.TestCase):
    def test_fixed_normalization_preserves_512_features_and_standardizes_each_observation(self) -> None:
        model = FlyCNSActorCritic()
        activity = torch.stack(
            (
                torch.linspace(-1.0, 1.0, DESCENDING_ACTIVITY_SIZE),
                torch.linspace(10.0, 20.0, DESCENDING_ACTIVITY_SIZE),
            )
        )

        normalized = model.input_normalization(activity)

        self.assertEqual(tuple(normalized.shape), (2, DESCENDING_ACTIVITY_SIZE))
        self.assertEqual(list(model.input_normalization.parameters()), [])
        torch.testing.assert_close(
            normalized.mean(dim=-1), torch.zeros(2), rtol=0.0, atol=1e-5
        )
        torch.testing.assert_close(
            normalized.std(dim=-1, unbiased=False),
            torch.ones(2),
            rtol=0.0,
            atol=1e-4,
        )
        logits, value = model(activity)
        torch.testing.assert_close(logits, model.policy_head(normalized))
        torch.testing.assert_close(value, model.value_head(normalized))

    def test_constant_and_zero_activity_remain_finite(self) -> None:
        model = FlyCNSActorCritic()
        activity = torch.stack(
            (torch.zeros(DESCENDING_ACTIVITY_SIZE), torch.full((DESCENDING_ACTIVITY_SIZE,), 7.0))
        )

        normalized = model.input_normalization(activity)
        logits, value = model(activity)

        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(normalized, torch.zeros_like(normalized), rtol=0.0, atol=1e-6)

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

    def test_pre_normalization_checkpoint_is_rejected(self) -> None:
        model = FlyCNSActorCritic()
        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "old_actor_critic.pt"
            torch.save(
                {
                    "format_version": 1,
                    "descending_activity_size": DESCENDING_ACTIVITY_SIZE,
                    "action_count": ACTION_COUNT,
                    "value_count": 1,
                    "state_dict": model.state_dict(),
                },
                weights_path,
            )

            with self.assertRaisesRegex(ValueError, "metadata does not match"):
                FlyCNSActorCritic.load_weights(weights_path)


if __name__ == "__main__":
    unittest.main()
