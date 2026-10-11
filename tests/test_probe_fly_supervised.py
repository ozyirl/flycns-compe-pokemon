from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from encoding import OBSERVATION_DIM
from flycns.action_decoder import ACTION_COUNT
from flycns.ppo_policy import DEFAULT_GRAPH_PATH
from flycns.trainable_decoder import DESCENDING_ACTIVITY_SIZE, TrainableActionDecoder
from probe_fly_supervised import (
    Examples,
    accuracy,
    collect_examples,
    frozen_features,
    masked_logits,
    split_battles,
    train_probe,
)


class ProbeFlySupervisedTest(unittest.TestCase):
    def test_collects_actual_teacher_orders_in_shared_action_space(self) -> None:
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[[6, 7]] = 1
        state = {"observation": np.zeros(OBSERVATION_DIM, dtype=np.float32), "action_mask": mask}
        battle = SimpleNamespace(wait=False)
        environment = Mock()
        environment.env.battle1 = battle
        environment.reset.return_value = (state, {})
        environment.step.return_value = (state, 0.0, True, False, {})
        environment.env.order_to_action.side_effect = [np.int64(7), np.int64(6)]
        teacher = Mock()
        teacher.choose_move.side_effect = ["first order", "second order"]

        examples = collect_examples(environment, teacher, battles=2, seed=3)

        np.testing.assert_array_equal(examples.actions, [7, 6])
        np.testing.assert_array_equal(examples.battle_ids, [0, 1])
        self.assertEqual(examples.observations.shape, (2, OBSERVATION_DIM))
        self.assertEqual(examples.action_masks.shape, (2, ACTION_COUNT))
        self.assertEqual(environment.env.order_to_action.call_count, 2)
        self.assertEqual(environment.step.call_count, 2)

    def test_rejects_illegal_teacher_label(self) -> None:
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[6] = 1
        state = {"observation": np.zeros(OBSERVATION_DIM, dtype=np.float32), "action_mask": mask}
        environment = Mock()
        environment.env.battle1 = SimpleNamespace(wait=False)
        environment.reset.return_value = (state, {})
        environment.env.order_to_action.return_value = np.int64(7)
        with self.assertRaisesRegex(ValueError, "illegal action 7"):
            collect_examples(environment, Mock(), battles=2, seed=3)
        environment.step.assert_not_called()

    def test_frozen_features_match_existing_simulator_dimensions(self) -> None:
        observation = np.linspace(-0.5, 0.5, OBSERVATION_DIM, dtype=np.float32)
        mask = np.zeros((1, ACTION_COUNT), dtype=np.int8)
        mask[0, 6] = 1
        examples = Examples(observation[None, :], mask, np.array([6]), np.array([0]))
        features = frozen_features(examples, DEFAULT_GRAPH_PATH)
        self.assertEqual(features.shape, (1, DESCENDING_ACTIVITY_SIZE))
        self.assertTrue(np.isfinite(features).all())

    def test_split_is_by_battle_not_turn(self) -> None:
        battle_ids = np.repeat(np.arange(10), 4)
        train, test = split_battles(battle_ids, seed=7, test_fraction=0.2)
        self.assertFalse(set(battle_ids[train]) & set(battle_ids[test]))
        self.assertEqual(len(train) + len(test), len(battle_ids))

    def test_only_legal_actions_can_win_masked_prediction(self) -> None:
        decoder = TrainableActionDecoder()
        with torch.no_grad():
            decoder.linear.weight.zero_()
            decoder.linear.bias.zero_()
            decoder.linear.bias[1] = 100.0
            decoder.linear.bias[7] = 2.0
        mask = torch.zeros((1, ACTION_COUNT), dtype=torch.bool)
        mask[0, 7] = True
        logits = masked_logits(decoder, torch.zeros((1, DESCENDING_ACTIVITY_SIZE)), mask)
        self.assertEqual(int(logits.argmax(dim=-1).item()), 7)
        self.assertEqual(accuracy(decoder, torch.zeros((1, DESCENDING_ACTIVITY_SIZE)), mask, torch.tensor([7]))[0], 1.0)

    def test_training_is_deterministic_and_learns_synthetic_rule(self) -> None:
        features = np.zeros((80, DESCENDING_ACTIVITY_SIZE), dtype=np.float32)
        features[:, 0] = np.where(np.arange(80) % 2 == 0, -2.0, 2.0)
        masks = np.zeros((80, ACTION_COUNT), dtype=np.int8)
        masks[:, [6, 7]] = 1
        actions = np.where(features[:, 0] < 0, 6, 7).astype(np.int64)
        examples = Examples(
            np.zeros((80, OBSERVATION_DIM), dtype=np.float32),
            masks, actions, np.repeat(np.arange(10), 8),
        )
        train, test = split_battles(examples.battle_ids, seed=7, test_fraction=0.2)
        first = train_probe(features, examples, train, test, seed=7, epochs=25, learning_rate=0.01, batch_size=16)
        second = train_probe(features, examples, train, test, seed=7, epochs=25, learning_rate=0.01, batch_size=16)
        self.assertEqual(first, second)
        self.assertGreater(first["test_accuracy"], 0.9)
        np.testing.assert_array_equal(features[:, 0], np.where(np.arange(80) % 2 == 0, -2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
