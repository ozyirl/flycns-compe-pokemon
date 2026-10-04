from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from poke_env.environment import SinglesEnv
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer

from encoding import OBSERVATION_DIM
from env import DEFAULT_BATTLE_FORMAT, make_opponent
from flycns.action_decoder import ACTION_COUNT
from flycns.actor_critic import FlyCNSActorCritic
from flycns.player import FlyCNSPlayer
from flycns.ppo_policy import PolicyDecision


class FlyCNSPlayerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = Path(self.directory.name) / "fly.pt"
        FlyCNSActorCritic().save_weights(self.checkpoint)

    def test_uses_existing_observation_mask_and_order_conversion(self) -> None:
        player = FlyCNSPlayer(
            self.checkpoint, battle_format=DEFAULT_BATTLE_FORMAT, start_listening=False
        )
        self.assertIsInstance(player, Player)
        self.assertTrue(all(not p.requires_grad for p in player.policy.actor_critic.parameters()))
        self.assertFalse(player.policy.actor_critic.training)
        original = [p.detach().clone() for p in player.policy.actor_critic.parameters()]
        battle = object()
        observation = np.arange(OBSERVATION_DIM, dtype=np.float32)
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[7] = 1
        decision = PolicyDecision(7, -0.1, 0.2, tuple([0.0] * ACTION_COUNT))
        order = object()
        with (
            patch("flycns.player.embed_battle", return_value=observation) as embed,
            patch.object(SinglesEnv, "get_action_mask", return_value=mask.tolist()) as get_mask,
            patch.object(SinglesEnv, "action_to_order", return_value=order) as to_order,
            patch.object(player.policy, "predict", return_value=decision) as predict,
        ):
            self.assertIs(player.choose_move(battle), order)

        embed.assert_called_once_with(battle)
        get_mask.assert_called_once_with(battle)
        args, kwargs = predict.call_args
        np.testing.assert_array_equal(args[0], observation)
        np.testing.assert_array_equal(args[1], mask)
        self.assertEqual(kwargs, {"deterministic": True})
        args, kwargs = to_order.call_args
        self.assertIsInstance(args[0], np.int64)
        self.assertEqual(args[0], 7)
        self.assertIs(args[1], battle)
        self.assertEqual(kwargs, {"strict": True})
        self.assertEqual(player.choices_made, 1)
        self.assertEqual(player.illegal_action_count, 0)
        for before, after in zip(original, player.policy.actor_critic.parameters(), strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_rejects_illegal_policy_choice(self) -> None:
        player = FlyCNSPlayer(
            self.checkpoint, battle_format=DEFAULT_BATTLE_FORMAT, start_listening=False
        )
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[7] = 1
        with (
            patch("flycns.player.embed_battle", return_value=np.zeros(OBSERVATION_DIM)),
            patch.object(SinglesEnv, "get_action_mask", return_value=mask),
            patch.object(
                player.policy,
                "predict",
                return_value=PolicyDecision(8, 0.0, 0.0, tuple([0.0] * ACTION_COUNT)),
            ),
            patch.object(SinglesEnv, "action_to_order") as to_order,
        ):
            with self.assertRaisesRegex(ValueError, "illegal action 8"):
                player.choose_move(object())
        to_order.assert_not_called()
        self.assertEqual(player.illegal_action_count, 1)

    def test_factory_selects_fly_without_changing_existing_opponents(self) -> None:
        player = make_opponent("fly", fly_checkpoint=self.checkpoint)
        self.assertIsInstance(player, FlyCNSPlayer)
        self.assertEqual(player.checkpoint_path, self.checkpoint)
        self.assertIsInstance(make_opponent("random"), RandomPlayer)
        self.assertIsInstance(make_opponent("max-power"), MaxBasePowerPlayer)
        with self.assertRaisesRegex(ValueError, "requires fly_checkpoint"):
            make_opponent("fly")


if __name__ == "__main__":
    unittest.main()
