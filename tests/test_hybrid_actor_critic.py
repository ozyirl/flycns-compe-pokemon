from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from encoding import OBSERVATION_DIM
from flycns.action_decoder import ACTION_COUNT
from flycns.actor_critic import FlyCNSActorCritic
from flycns.hybrid_actor_critic import (
    MOVE_FEATURE_SIZE,
    MOVE_FEATURES_OFFSET,
    SWITCH_FEATURE_SIZE,
    SWITCH_FEATURES_OFFSET,
    FlyCNSHybridActorCritic,
    load_actor_critic,
)
from flycns.ppo_policy import FlyCNSPPOPolicy
from flycns.trainable_decoder import DESCENDING_ACTIVITY_SIZE


def _seeded_fly() -> FlyCNSActorCritic:
    torch.manual_seed(3)
    return FlyCNSActorCritic()


def _inputs(batch: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(5)
    activity = torch.rand(batch, DESCENDING_ACTIVITY_SIZE, generator=generator) * 10
    observation = torch.rand(batch, OBSERVATION_DIM, generator=generator)
    return activity, observation


class FlyCNSHybridActorCriticTest(unittest.TestCase):
    def test_warm_start_reproduces_fly_only_outputs(self) -> None:
        source = _seeded_fly()
        hybrid = FlyCNSHybridActorCritic.from_fly(source)
        activity, observation = _inputs()

        source_logits, source_value = source(activity)
        hybrid_logits, hybrid_value = hybrid(activity, observation)

        torch.testing.assert_close(hybrid_logits, source_logits)
        torch.testing.assert_close(hybrid_value, source_value)

    def test_requires_observation(self) -> None:
        activity, _ = _inputs()
        with self.assertRaisesRegex(ValueError, "observation"):
            FlyCNSHybridActorCritic()(activity)

    def test_move_slot_score_reaches_every_gimmick_variant_of_that_slot_only(self) -> None:
        hybrid = FlyCNSHybridActorCritic.from_fly(_seeded_fly())
        with torch.no_grad():
            hybrid.policy_head.move_scorer.weight[0, 0] = 1.0  # Base power input.
        activity, observation = _inputs(1)
        baseline, _ = hybrid(activity, observation)
        bumped = observation.clone()
        bumped[0, MOVE_FEATURES_OFFSET + 2 * MOVE_FEATURE_SIZE] += 0.08  # Slot 2, +20 base power.

        changed, _ = hybrid(activity, bumped)

        delta = (changed - baseline)[0]
        expected = torch.zeros(ACTION_COUNT)
        expected[[8, 12, 16, 20, 24]] = 1.0
        torch.testing.assert_close(delta, expected, rtol=0.0, atol=1e-5)

    def test_switch_slot_score_reaches_only_that_switch(self) -> None:
        hybrid = FlyCNSHybridActorCritic.from_fly(_seeded_fly())
        with torch.no_grad():
            hybrid.policy_head.switch_scorer.weight[0, 2] = 1.0  # HP fraction input.
        activity, observation = _inputs(1)
        baseline, _ = hybrid(activity, observation)
        bumped = observation.clone()
        bumped[0, SWITCH_FEATURES_OFFSET + 4 * SWITCH_FEATURE_SIZE + 2] += 0.5

        changed, _ = hybrid(activity, bumped)

        expected = torch.zeros(ACTION_COUNT)
        expected[4] = 0.5
        torch.testing.assert_close((changed - baseline)[0], expected, rtol=0.0, atol=1e-5)

    def test_parameter_groups_partition_the_model(self) -> None:
        hybrid = FlyCNSHybridActorCritic()
        raw = {id(item) for item in hybrid.raw_parameters()}
        fly = {id(item) for item in hybrid.fly_parameters()}

        self.assertFalse(raw & fly)
        self.assertEqual(raw | fly, {id(item) for item in hybrid.parameters()})

    def test_round_trip_and_dispatching_loader(self) -> None:
        hybrid = FlyCNSHybridActorCritic.from_fly(_seeded_fly())
        with torch.no_grad():
            hybrid.policy_head.move_scorer.weight.fill_(0.25)
        fly = _seeded_fly()
        activity, observation = _inputs()
        with tempfile.TemporaryDirectory() as directory:
            hybrid_path = Path(directory) / "hybrid.pt"
            fly_path = Path(directory) / "fly.pt"
            hybrid.save_weights(hybrid_path)
            fly.save_weights(fly_path)

            loaded_hybrid = load_actor_critic(hybrid_path)
            loaded_fly = load_actor_critic(fly_path)
            with self.assertRaisesRegex(ValueError, "metadata"):
                FlyCNSHybridActorCritic.load_weights(fly_path)

        self.assertIsInstance(loaded_hybrid, FlyCNSHybridActorCritic)
        self.assertIsInstance(loaded_fly, FlyCNSActorCritic)
        torch.testing.assert_close(loaded_hybrid(activity, observation), hybrid(activity, observation))

    def test_policy_predicts_legal_actions_with_hybrid_heads(self) -> None:
        policy = FlyCNSPPOPolicy(actor_critic=FlyCNSHybridActorCritic.from_fly(_seeded_fly()))
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[[1, 6, 7]] = 1

        decision = policy.predict(np.zeros(OBSERVATION_DIM, dtype=np.float32), mask)

        self.assertIn(decision.selected_action_index, (1, 6, 7))
        self.assertTrue(np.isfinite(decision.state_value))


if __name__ == "__main__":
    unittest.main()
