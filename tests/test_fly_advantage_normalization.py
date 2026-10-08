from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiment_fly_advantage_normalization import CheckpointResult, aggregate, training_environment
from experiment_fly_maxpower_epochs import train_multi_epoch
from flycns.ppo_policy import FlyCNSPPOPolicy
from smoke_fly_ppo_update import (
    PPOUpdateResult,
    advantages_and_returns,
    normalized_policy_advantages,
    update_once,
)
from smoke_fly_rollout import RolloutResult, TurnRecord
from tests.test_smoke_fly_ppo_update import recorded_rollout


def one_turn_rollout(seed: int) -> RolloutResult:
    turn = TurnRecord(
        observation=np.zeros(186, dtype=np.float32),
        action_mask=np.ones(26, dtype=np.int8),
        selected_action=7,
        log_probability=-1.0,
        state_value=0.25,
        reward=float(seed),
        done=True,
    )
    return RolloutResult((turn,), float(seed), "win", False, True)


class AdvantageNormalizationTest(unittest.TestCase):
    def test_training_environments_use_distinct_temporary_showdown_accounts(self) -> None:
        with (
            patch("experiment_fly_advantage_normalization.secrets.token_hex", side_effect=["aaaaa", "bbbbb"]),
            patch("experiment_fly_advantage_normalization.ShowdownEnv") as showdown,
            patch("experiment_fly_advantage_normalization.SingleAgentWrapper"),
            patch("experiment_fly_advantage_normalization.make_opponent") as opponent,
        ):
            training_environment(object())
            training_environment(object())
        names = [call.kwargs["account_configuration1"].username for call in showdown.call_args_list]
        self.assertEqual(names, ["fanaaaaaa", "fanbbbbba"])
        self.assertEqual(opponent.call_count, 2)

    def test_policy_copy_has_zero_mean_unit_std_and_does_not_change_raw(self) -> None:
        raw = np.asarray([-7.0, -2.0, 3.0, 12.0], dtype=np.float32)
        before = raw.copy()
        normalized = normalized_policy_advantages(raw)
        np.testing.assert_array_equal(raw, before)
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized.std()), 1.0, places=6)
        self.assertTrue(np.isfinite(normalized_policy_advantages(np.ones(4, dtype=np.float32))).all())

    def test_opt_in_changes_only_policy_loss_input_not_critic_targets(self) -> None:
        torch.manual_seed(7)
        reference = FlyCNSPPOPolicy()
        rollout = recorded_rollout(reference)
        raw_advantages, raw_returns = advantages_and_returns(rollout)
        original = [parameter.detach().clone() for parameter in reference.actor_critic.parameters()]
        baseline = FlyCNSPPOPolicy(actor_critic=copy.deepcopy(reference.actor_critic))
        normalized = FlyCNSPPOPolicy(actor_critic=copy.deepcopy(reference.actor_critic))

        old_result = update_once(baseline, rollout)
        new_result = update_once(normalized, rollout, normalize_advantages=True)

        self.assertAlmostEqual(old_result.value_loss, new_result.value_loss, places=5)
        self.assertAlmostEqual(old_result.entropy, new_result.entropy, places=5)
        self.assertAlmostEqual(old_result.policy_loss, -float(raw_advantages.mean()), places=4)
        self.assertAlmostEqual(new_result.policy_loss, 0.0, places=4)
        self.assertFalse(np.isclose(old_result.policy_loss, new_result.policy_loss))
        np.testing.assert_array_equal(advantages_and_returns(rollout)[0], raw_advantages)
        np.testing.assert_array_equal(advantages_and_returns(rollout)[1], raw_returns)
        for parameter, before in zip(reference.actor_critic.parameters(), original, strict=True):
            torch.testing.assert_close(parameter, before, rtol=0.0, atol=0.0)
        for old, new in zip(
            baseline.actor_critic.value_head.parameters(),
            normalized.actor_critic.value_head.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-7)

    def test_multi_epoch_mode_keeps_batch_counts_and_is_opt_in(self) -> None:
        def collect(_environment: object, _policy: object, *, seed: int) -> RolloutResult:
            return one_turn_rollout(seed)

        for normalize in (False, True):
            calls: list[dict[str, object]] = []

            def update(_policy: object, _rollout: RolloutResult, **kwargs: object) -> PPOUpdateResult:
                calls.append(kwargs)
                return PPOUpdateResult(0.1, 2.0, 1.0, 1.09, True, True, True)

            class FakeHeads:
                def save_weights(self, path: Path) -> None:
                    path.write_bytes(b"checkpoint")

            policy = SimpleNamespace(actor_critic=FakeHeads())
            with tempfile.TemporaryDirectory() as directory:
                with (
                    patch("experiment_fly_maxpower_epochs.run_one_battle", side_effect=collect),
                    patch("experiment_fly_maxpower_epochs.update_once", side_effect=update),
                    patch("experiment_fly_maxpower_epochs.policy_shift_metrics", return_value=(0.2, 0.01)),
                ):
                    steps, snapshots = train_multi_epoch(
                        policy, object, output_dir=Path(directory), seed=7,
                        normalize_advantages=normalize,
                    )
            self.assertEqual(len(calls), 104)
            self.assertEqual(len(steps), 104)
            self.assertEqual(len(snapshots), 5)
            self.assertTrue(all(call == ({"normalize_advantages": True} if normalize else {}) for call in calls))
            self.assertTrue(all(row.advantage_normalized_for_policy == normalize for row in snapshots))
            self.assertTrue(all(abs(row.normalized_advantage_mean) < 1e-5 for row in snapshots))
            self.assertTrue(all(abs(row.normalized_advantage_std - 1.0) < 1e-5 for row in snapshots))

    def test_aggregate_counts_wins_and_averages_training_statistics(self) -> None:
        base = dict(
            mode="normalized", after_battle=100, updated_through_battle=100,
            pending_rollouts=0, optimizer_steps=104, checkpoint=Path("candidate.pt"),
            checkpoint_sha256="hash", losses=0, ties=0, average_turns=30.0,
            policy_entropy=1.2, policy_loss=0.1, value_loss=2.0,
            raw_advantage_mean=-1.0, raw_advantage_std=5.0,
            normalized_advantage_mean=0.0, normalized_advantage_std=1.0,
            clip_fraction=0.2, approximate_kl=0.01, fly_illegal_actions=0,
            opposing_illegal_actions=0,
        )
        first = CheckpointResult(seed=7, wins=20, win_rate=0.4, average_reward=-4.0, **base)
        second = CheckpointResult(seed=17, wins=30, win_rate=0.6, average_reward=2.0, **base)
        result = aggregate([first, second])
        self.assertEqual(result["wins"], 50)
        self.assertEqual(result["evaluation_battles"], 100)
        self.assertEqual(result["win_rate"], 0.5)
        self.assertEqual(result["average_reward"], -1.0)


if __name__ == "__main__":
    unittest.main()
