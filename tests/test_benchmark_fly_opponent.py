from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmark_fly_opponent import baseline_action, run_matchup, scripted_action
from env import DEFAULT_BATTLE_FORMAT
from flycns.action_decoder import ACTION_COUNT
from flycns.actor_critic import FlyCNSActorCritic
from flycns.player import FlyCNSPlayer


class FakeBattle:
    def __init__(self) -> None:
        self.wait = False
        self.won = False
        self.lost = False
        self.turn = 0


class FakeEnvironment:
    def __init__(self, fly: FlyCNSPlayer) -> None:
        self.opponent = fly
        self.env = SimpleNamespace(battle1=None, battle2=None, _reward_buffer={})
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, *, seed: int) -> tuple[dict, dict]:
        self.reset_calls += 1
        assert seed == 6 + self.reset_calls
        self.env.battle1 = FakeBattle()
        self.env.battle2 = FakeBattle()
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[7] = 1
        return {"action_mask": mask}, {}

    def step(self, action: np.int64) -> tuple[dict, float, bool, bool, dict]:
        self.step_calls += 1
        assert isinstance(action, np.int64)
        assert action == 7
        fly_battle = self.env.battle2
        fly_battle.won = self.reset_calls % 2 == 1
        fly_battle.lost = not fly_battle.won
        fly_battle.turn = 10 + self.reset_calls
        self.env._reward_buffer[fly_battle] = 2.5 * self.reset_calls
        return {}, -999.0, True, False, {}


class FakeModel:
    def predict(self, observation: dict, *, deterministic: bool) -> tuple[np.int64, None]:
        assert deterministic
        return np.int64(7), None


class FakePlayer:
    def choose_move(self, battle: FakeBattle) -> object:
        return battle


class FlyOpponentBenchmarkTest(unittest.TestCase):
    def test_scores_fly_side_reward_and_turns_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fly.pt"
            FlyCNSActorCritic().save_weights(checkpoint)
            fly = FlyCNSPlayer(
                checkpoint, battle_format=DEFAULT_BATTLE_FORMAT, start_listening=False
            )
            environment = FakeEnvironment(fly)
            result = run_matchup(
                "baseline", environment, baseline_action(FakeModel()), battles=3, seed=7
            )

        self.assertEqual(environment.reset_calls, 3)
        self.assertEqual(environment.step_calls, 3)
        self.assertEqual((result.wins, result.losses, result.ties), (2, 1, 0))
        self.assertAlmostEqual(result.win_rate, 2 / 3)
        self.assertEqual(result.rewards, (2.5, 5.0, 7.5))
        self.assertEqual(result.average_reward, 5.0)
        self.assertEqual(result.turns, (11, 12, 13))
        self.assertEqual(result.average_battle_length, 12.0)
        self.assertEqual((result.fly_illegal_actions, result.opposing_illegal_actions), (0, 0))

    def test_scripted_order_uses_environment_action_converter(self) -> None:
        battle = FakeBattle()
        converter = SimpleNamespace(order_to_action=lambda order, b, strict: np.int64(7))
        environment = SimpleNamespace(env=SimpleNamespace(battle1=battle, order_to_action=converter.order_to_action))
        self.assertEqual(scripted_action(FakePlayer())(environment, {}), 7)


if __name__ == "__main__":
    unittest.main()
