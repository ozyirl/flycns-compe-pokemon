from __future__ import annotations

import unittest

import numpy as np
import torch

from flycns.action_decoder import ACTION_COUNT
from flycns.trainable_decoder import DESCENDING_ACTIVITY_SIZE, TrainableActionDecoder
from smoke_fly_trainable_decoder import run_smoke


class FlyTrainableDecoderIntegrationTest(unittest.TestCase):
    def test_real_spike_counts_reach_decoder_without_changing_weights(self) -> None:
        decoder = TrainableActionDecoder()
        original_parameters = [parameter.detach().clone() for parameter in decoder.parameters()]

        result = run_smoke(decoder=decoder)

        self.assertGreater(result.sensory_neurons_stimulated, 0)
        self.assertGreater(result.active_descending_neurons, 0)
        self.assertEqual(result.descending_activity.shape, (DESCENDING_ACTIVITY_SIZE,))
        self.assertEqual(
            int(np.count_nonzero(result.descending_activity)),
            result.active_descending_neurons,
        )
        self.assertEqual(result.logits.shape, (ACTION_COUNT,))
        self.assertTrue(np.isfinite(result.logits).all())
        self.assertEqual(result.highest_scoring_action_index, int(np.argmax(result.logits)))
        for parameter, original in zip(decoder.parameters(), original_parameters, strict=True):
            self.assertTrue(torch.equal(parameter, original))
            self.assertIsNone(parameter.grad)

    def test_default_untrained_decoder_is_repeatable(self) -> None:
        first = run_smoke()
        second = run_smoke()

        np.testing.assert_array_equal(first.descending_activity, second.descending_activity)
        np.testing.assert_array_equal(first.logits, second.logits)
        self.assertEqual(first.highest_scoring_action_index, second.highest_scoring_action_index)


if __name__ == "__main__":
    unittest.main()
