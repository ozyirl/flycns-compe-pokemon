from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from encoding import MATCHUP_OBSERVATION_DIM, OBSERVATION_DIM
from experiment_fly_hybrid import BatchStats, make_model, make_optimizer, ppo_epochs, write_report
from flycns.action_decoder import ACTION_COUNT
from flycns.actor_critic import FlyCNSActorCritic
from flycns.hybrid_actor_critic import (
    FlyCNSContextualActorCritic,
    FlyCNSHybridActorCritic,
    FlyCNSMatchupActorCritic,
)
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_rollout import RolloutResult, TurnRecord


def _seeded_fly() -> FlyCNSActorCritic:
    torch.manual_seed(3)
    return FlyCNSActorCritic()


class ExperimentFlyHybridTest(unittest.TestCase):
    def test_make_model_rejects_unknown_arm(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown arm"):
            make_model("mlp", None)  # type: ignore[arg-type]

    def test_make_model_continues_hybrid_and_widens_fly_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fly_path = Path(directory) / "fly.pt"
            hybrid_path = Path(directory) / "hybrid.pt"
            _seeded_fly().save_weights(fly_path)
            hybrid = FlyCNSHybridActorCritic.from_fly(_seeded_fly())
            with torch.no_grad():
                hybrid.policy_head.move_scorer.weight.fill_(0.5)
            hybrid.save_weights(hybrid_path)

            self.assertIsInstance(make_model("fly", fly_path), FlyCNSActorCritic)
            self.assertIsInstance(make_model("hybrid", fly_path), FlyCNSHybridActorCritic)
            continued = make_model("hybrid", hybrid_path)
            with self.assertRaisesRegex(ValueError, "FlyCNSHybridActorCritic checkpoint"):
                make_model("fly", hybrid_path)
            contextual = make_model("contextual", hybrid_path)
            self.assertIsInstance(contextual, FlyCNSContextualActorCritic)
            self.assertIsInstance(make_model("contextual", fly_path), FlyCNSContextualActorCritic)
            self.assertIsInstance(make_model("matchup", hybrid_path), FlyCNSMatchupActorCritic)
            contextual_path = Path(directory) / "contextual.pt"
            contextual.save_weights(contextual_path)
            with self.assertRaisesRegex(ValueError, "FlyCNSContextualActorCritic checkpoint"):
                make_model("matchup", contextual_path)

        torch.testing.assert_close(
            continued.policy_head.move_scorer.weight, hybrid.policy_head.move_scorer.weight
        )

    def test_optimizer_gives_raw_parameters_their_own_learning_rate(self) -> None:
        model = FlyCNSHybridActorCritic()
        optimizer = make_optimizer(model, learning_rate=3e-4, raw_learning_rate=1e-2)

        self.assertEqual([group["lr"] for group in optimizer.param_groups], [3e-4, 1e-2])
        self.assertEqual(len(optimizer.param_groups[1]["params"]), len(model.raw_parameters()))
        fly_optimizer = make_optimizer(_seeded_fly(), learning_rate=3e-4, raw_learning_rate=1e-2)
        self.assertEqual([group["lr"] for group in fly_optimizer.param_groups], [3e-4])

    def test_ppo_epochs_train_raw_scorers_with_persistent_optimizer(self) -> None:
        hybrid = FlyCNSHybridActorCritic.from_fly(_seeded_fly())
        cases = (
            (hybrid, OBSERVATION_DIM, [6, 7]),
            (FlyCNSMatchupActorCritic.from_hybrid(hybrid), MATCHUP_OBSERVATION_DIM, [1, 2, 6]),
        )
        for model, width, legal_actions in cases:
            with self.subTest(model=type(model).__name__):
                policy = FlyCNSPPOPolicy(actor_critic=model)
                optimizer = make_optimizer(model, learning_rate=3e-4, raw_learning_rate=1e-2)
                mask = np.zeros(ACTION_COUNT, dtype=np.int8)
                mask[legal_actions] = 1
                rng = np.random.default_rng(0)
                turns = []
                for index, reward in enumerate((1.0, -1.0, 2.0)):
                    observation = rng.random(width).astype(np.float32)
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

    def test_write_report_serializes_rows_and_replaces_previous_file(self) -> None:
        batches = [BatchStats(8, 120, 3, 0.1, 2.0, 1.2, 0.05, 0.01)]
        report = {"arm": "contextual", "complete": False, "batches": batches, "evaluations": []}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_report(output, report)
            report["complete"] = True
            write_report(output, report)
            saved = json.loads((output / "results.json").read_text(encoding="utf-8"))
            leftovers = list(output.glob("*.tmp"))

        self.assertTrue(saved["complete"])
        self.assertEqual(saved["batches"][0]["wins"], 3)
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
