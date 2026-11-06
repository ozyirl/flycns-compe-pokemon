from __future__ import annotations

import unittest

import numpy as np
import torch

from encoding import OBSERVATION_DIM
from experiment_fly_hybrid import make_model, make_optimizer, ppo_epochs
from flycns.action_decoder import ACTION_COUNT
from flycns.actor_critic import FlyCNSActorCritic
from flycns.hybrid_actor_critic import FlyCNSHybridActorCritic
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_rollout import RolloutResult, TurnRecord


def _seeded_fly() -> FlyCNSActorCritic:
    torch.manual_seed(3)
    return FlyCNSActorCritic()


class ExperimentFlyHybridTest(unittest.TestCase):
    def test_make_model_rejects_unknown_arm(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown arm"):
            make_model("mlp", None)  # type: ignore[arg-type]

    def test_optimizer_gives_raw_parameters_their_own_learning_rate(self) -> None:
        model = FlyCNSHybridActorCritic()
        optimizer = make_optimizer(model, learning_rate=3e-4, raw_learning_rate=1e-2)

        self.assertEqual([group["lr"] for group in optimizer.param_groups], [3e-4, 1e-2])
        self.assertEqual(len(optimizer.param_groups[1]["params"]), len(model.raw_parameters()))
        fly_optimizer = make_optimizer(_seeded_fly(), learning_rate=3e-4, raw_learning_rate=1e-2)
        self.assertEqual([group["lr"] for group in fly_optimizer.param_groups], [3e-4])

    def test_ppo_epochs_train_raw_scorers_with_persistent_optimizer(self) -> None:
        model = FlyCNSHybridActorCritic.from_fly(_seeded_fly())
        policy = FlyCNSPPOPolicy(actor_critic=model)
        optimizer = make_optimizer(model, learning_rate=3e-4, raw_learning_rate=1e-2)
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[[6, 7]] = 1
        rng = np.random.default_rng(0)
        turns = []
        for index, reward in enumerate((1.0, -1.0, 2.0)):
            observation = rng.random(OBSERVATION_DIM).astype(np.float32)
            decision = policy.predict(observation, mask, deterministic=False)
            turns.append(TurnRecord(
                observation=observation, action_mask=mask, selected_action=decision.selected_action_index,
                log_probability=decision.action_log_probability, state_value=decision.state_value,
                reward=reward, done=index == 2,
            ))
        rollout = RolloutResult(tuple(turns), 2.0, "win", False, True)
        before = [item.detach().clone() for item in model.raw_parameters()]

        stats = ppo_epochs(policy, optimizer, rollout, epochs=2)

        self.assertTrue(all(np.isfinite(value) for value in stats.values()))
        self.assertTrue(any(
            not torch.equal(item, old) for item, old in zip(model.raw_parameters(), before, strict=True)
        ))
        self.assertEqual(len(optimizer.state), len(list(model.parameters())))



if __name__ == "__main__":
    unittest.main()
