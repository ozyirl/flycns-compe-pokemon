from __future__ import annotations

import unittest

import torch

from smoke_train_decoder import run_smoke


class DecoderLearningSmokeTest(unittest.TestCase):
    def test_target_probability_rises_and_only_decoder_parameters_change(self) -> None:
        result = run_smoke()

        self.assertEqual(result.target_action, 7)
        self.assertGreater(result.after_probability, result.before_probability)
        self.assertEqual(set(result.changed_parameters), {"linear.weight", "linear.bias"})

    def test_fixed_seed_repeats_identically_without_changing_global_rng(self) -> None:
        rng_before = torch.random.get_rng_state().clone()

        first = run_smoke(seed=7)
        second = run_smoke(seed=7)

        self.assertEqual(first, second)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))


if __name__ == "__main__":
    unittest.main()
