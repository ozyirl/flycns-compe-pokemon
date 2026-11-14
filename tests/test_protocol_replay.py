from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from encoding import embed_battle, embed_matchups
from flycns.player import decide, load_frozen_policy
from flycns.protocol_replay import order_to_choice, replay_battle


CHECKPOINT = Path(__file__).resolve().parents[1] / "deploy/fly_matchup.pt"
# One battle this checkpoint played live on a local Showdown server, recorded as its
# protocol stream plus the observation and choice of each of its first ten decisions.
FIXTURE = json.loads((Path(__file__).parent / "fixtures/replay_battle.json").read_text(encoding="utf-8"))


class ProtocolReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_frozen_policy(CHECKPOINT)

    def test_replay_reproduces_live_observations_and_choices(self) -> None:
        for decision in FIXTURE["decisions"]:
            with self.subTest(prefix=decision["prefix"]):
                battle, request = replay_battle(FIXTURE["protocol"][: decision["prefix"]], FIXTURE["format"])
                observation = np.concatenate([embed_battle(battle), embed_matchups(battle)])
                order, _ = decide(self.policy, battle)

                np.testing.assert_array_equal(observation, np.asarray(decision["observation"], dtype=np.float32))
                self.assertEqual(order_to_choice(order, battle, request), decision["choice"])

    def test_replay_requires_a_request(self) -> None:
        protocol = [line for line in FIXTURE["protocol"] if not line.startswith("|request|")]

        with self.assertRaisesRegex(ValueError, "no request"):
            replay_battle(protocol, FIXTURE["format"])

    def test_replay_requires_the_requesting_players_name(self) -> None:
        protocol = [line for line in FIXTURE["protocol"] if not line.startswith("|player|p1|")]

        with self.assertRaisesRegex(ValueError, "no \\|player\\| line for p1"):
            replay_battle(protocol, FIXTURE["format"])


if __name__ == "__main__":
    unittest.main()
