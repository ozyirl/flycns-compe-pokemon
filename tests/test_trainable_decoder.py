from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from flycns.action_decoder import ACTION_COUNT
from flycns.trainable_decoder import (
    DESCENDING_ACTIVITY_SIZE,
    TrainableActionDecoder,
)


class TrainableActionDecoderTest(unittest.TestCase):
    def test_single_activity_vector_produces_action_logits(self) -> None:
        decoder = TrainableActionDecoder()

        logits = decoder(torch.zeros(DESCENDING_ACTIVITY_SIZE))

        self.assertEqual(tuple(logits.shape), (ACTION_COUNT,))

    def test_gradients_reach_decoder_parameters(self) -> None:
        decoder = TrainableActionDecoder()
        activity = torch.linspace(-1.0, 1.0, DESCENDING_ACTIVITY_SIZE)

        decoder(activity).square().sum().backward()

        for parameter in decoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

    def test_saved_weights_reload_identically(self) -> None:
        torch.manual_seed(7)
        decoder = TrainableActionDecoder()
        activity = torch.linspace(-1.0, 1.0, DESCENDING_ACTIVITY_SIZE)
        expected_logits = decoder(activity).detach()

        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "decoder.pt"
            decoder.save_weights(weights_path)
            reloaded = TrainableActionDecoder.load_weights(weights_path)

        torch.testing.assert_close(
            reloaded(activity), expected_logits, rtol=0.0, atol=0.0
        )


if __name__ == "__main__":
    unittest.main()
