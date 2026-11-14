from __future__ import annotations

import json
import unittest
from pathlib import Path

import httpx
import numpy as np

from encoding import OBSERVATION_DIM
from flycns.action_decoder import ACTION_COUNT
from serve_fly import app


REPLAY = json.loads((Path(__file__).parent / "fixtures/replay_battle.json").read_text(encoding="utf-8"))


class FlyAPITest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_health_reports_loaded_graph(self) -> None:
        response = await self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual({route.path for route in app.routes}, {"/health", "/predict", "/v1/choose"})
        self.assertEqual(
            response.json(),
            {"status": "ok", "neuron_count": 4274, "version": "0.2.0", "model": "fly_matchup.pt"},
        )

    async def test_choose_replays_the_protocol_and_returns_a_showdown_choice(self) -> None:
        decision = REPLAY["decisions"][2]

        response = await self.client.post(
            "/v1/choose",
            json={"format": REPLAY["format"], "protocol": REPLAY["protocol"][: decision["prefix"]]},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["choice"], decision["choice"])
        self.assertEqual(set(payload), {"choice", "action_index", "turn", "state_value"})

    async def test_choose_without_gimmicks_picks_a_plain_move_or_switch(self) -> None:
        decision = REPLAY["decisions"][2]
        self.assertEqual(decision["choice"], "move 2 terastallize")

        response = await self.client.post(
            "/v1/choose",
            json={
                "format": REPLAY["format"],
                "protocol": REPLAY["protocol"][: decision["prefix"]],
                "gimmicks": False,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertRegex(response.json()["choice"], r"^(move [1-4]|switch [1-6])$")
        self.assertLess(response.json()["action_index"], 10)

    async def test_choose_rejects_unusable_requests(self) -> None:
        protocol = REPLAY["protocol"][: REPLAY["decisions"][0]["prefix"]]
        cases = {
            "no request": [line for line in protocol if not line.startswith("|request|")],
            "wait": [*protocol, '|request|{"wait":true,"side":{"id":"p1","name":"Recorder 1","pokemon":[]}}'],
            "format": protocol,
        }
        for name, lines in cases.items():
            with self.subTest(name):
                battle_format = "gen8randombattle" if name == "format" else REPLAY["format"]
                response = await self.client.post(
                    "/v1/choose", json={"format": battle_format, "protocol": lines}
                )

                self.assertEqual(response.status_code, 422, response.text)

    async def test_predict_returns_scores_and_never_selects_an_illegal_action(self) -> None:
        observation = np.linspace(-0.8, 1.0, OBSERVATION_DIM).tolist()
        action_mask = [0] * ACTION_COUNT
        legal_actions = {2, 7, 23}
        for action in legal_actions:
            action_mask[action] = 1

        response = await self.client.post(
            "/predict",
            json={"observation": observation, "action_mask": action_mask},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(len(payload["raw_scores"]), ACTION_COUNT)
        self.assertEqual(len(payload["legal_masked_scores"]), ACTION_COUNT)
        self.assertIn(payload["hypothetical_selected_action_index"], legal_actions)
        for action, masked_score in enumerate(payload["legal_masked_scores"]):
            if action in legal_actions:
                self.assertEqual(masked_score, payload["raw_scores"][action])
            else:
                self.assertIsNone(masked_score)
        self.assertIn("total_cns_spikes", payload["diagnostics"])
        self.assertEqual(len(payload["diagnostics"]["top_descending_neurons"]), 10)

    async def test_predict_rejects_a_mask_without_legal_actions(self) -> None:
        response = await self.client.post(
            "/predict",
            json={
                "observation": [0.0] * OBSERVATION_DIM,
                "action_mask": [0] * ACTION_COUNT,
            },
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
