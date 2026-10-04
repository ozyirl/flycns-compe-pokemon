from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from encoding import OBSERVATION_DIM
from env import DEFAULT_BATTLE_FORMAT
from flycns.action_decoder import ACTION_COUNT
from flycns.actor_critic import FlyCNSActorCritic
from flycns.player import FlyCNSPlayer
from smoke_fly_opponent import SMOKE_BATTLES, run_ten_battles


class FakeEnvironment:
    def __init__(self, opponent: FlyCNSPlayer) -> None:
        self.opponent = opponent
        self.env = SimpleNamespace(battle1=None)
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, *, seed: int) -> tuple[dict, dict]:
        self.reset_calls += 1
        assert seed == 6 + self.reset_calls
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[7] = 1
        return {"observation": np.zeros(OBSERVATION_DIM), "action_mask": mask}, {}

    def step(self, action: np.int64) -> tuple[dict, float, bool, bool, dict]:
        self.step_calls += 1
        assert isinstance(action, np.int64)
        assert action == 7
        self.env.battle1 = SimpleNamespace(
            won=self.step_calls % 2 == 0, lost=self.step_calls % 2 == 1
        )
        return {}, 1.5, True, False, {}


class FakeModel:
    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, observation: dict, *, deterministic: bool) -> tuple[np.int64, None]:
        assert deterministic
        self.predict_calls += 1
        return np.int64(7), None


class FlyOpponentSmokeTest(unittest.TestCase):
    def test_exactly_ten_read_only_battles_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fly.pt"
            FlyCNSActorCritic().save_weights(checkpoint)
            opponent = FlyCNSPlayer(
                checkpoint, battle_format=DEFAULT_BATTLE_FORMAT, start_listening=False
            )
            environment = FakeEnvironment(opponent)
            model = FakeModel()

            result = run_ten_battles(model, environment, seed=7)

        self.assertEqual(environment.reset_calls, SMOKE_BATTLES)
        self.assertEqual(environment.step_calls, SMOKE_BATTLES)
        self.assertEqual(model.predict_calls, SMOKE_BATTLES)
        self.assertEqual((result.wins, result.losses, result.ties), (5, 5, 0))
        self.assertEqual(result.rewards, (1.5,) * SMOKE_BATTLES)
        self.assertEqual(result.trainer_illegal_actions, 0)
        self.assertEqual(result.fly_illegal_actions, 0)
        self.assertTrue(result.fly_weights_unchanged)


if __name__ == "__main__":
    unittest.main()
