from __future__ import annotations

import unittest

import numpy as np

from encoding import OBSERVATION_DIM
from smoke_fly_representation_diversity import observation_profiles, run_smoke


class FlyRepresentationDiversityTest(unittest.TestCase):
    def test_five_feature_profiles_produce_distinct_descending_activity(self) -> None:
        profiles = observation_profiles()
        self.assertEqual(len(profiles), 5)
        self.assertTrue(all(vector.shape == (OBSERVATION_DIM,) for vector in profiles.values()))

        result = run_smoke()

        self.assertEqual(len(result.representations), 5)
        self.assertTrue(all(item.differs_from_all_others for item in result.representations))
        for item in result.representations:
            self.assertEqual(item.descending_spikes.shape, (512,))
            self.assertEqual(
                item.active_descending_neurons,
                int(np.count_nonzero(item.descending_spikes)),
            )
            self.assertEqual(item.total_descending_spikes, int(item.descending_spikes.sum()))
            self.assertGreater(item.total_descending_spikes, 0)
        self.assertEqual(result.cosine_similarity.shape, (5, 5))
        self.assertTrue(np.isfinite(result.cosine_similarity).all())
        np.testing.assert_allclose(result.cosine_similarity, result.cosine_similarity.T)
        np.testing.assert_allclose(np.diag(result.cosine_similarity), 1.0)
        self.assertLess(result.cosine_similarity[0, 3], 0.9)


if __name__ == "__main__":
    unittest.main()
