from __future__ import annotations

import unittest

import httpx
import numpy as np

from encoding import OBSERVATION_DIM
from flycns.action_decoder import ACTION_COUNT
from serve_fly import app


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
        self.assertEqual({route.path for route in app.routes}, {"/health", "/predict"})
        self.assertEqual(
            response.json(),
            {"status": "ok", "neuron_count": 4274, "version": "0.1.0"},
        )

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
