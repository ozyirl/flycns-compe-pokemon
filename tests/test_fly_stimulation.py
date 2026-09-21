from __future__ import annotations

import unittest

import numpy as np

from flycns.config import FlyGraphConfig
from flycns.action_decoder import ACTION_COUNT, DescendingActionDecoder
from flycns.graph import FlyGraph, NeuronRecord
from flycns.stimulation import BattleSensoryAdapter, FlySpikeSimulator


def diagnostic_graph() -> FlyGraph:
    neurons = [
        NeuronRecord(1, "sensory", "sensory", "", "sensory-a", "ACH", 1.0),
        NeuronRecord(2, "sensory", "sensory", "", "sensory-b", "ACH", 1.0),
        NeuronRecord(3, "central", "cb_intrinsic", "", "central", "ACH", 1.0),
        NeuronRecord(4, "descending", "descending_neuron", "", "DN-test", "ACH", 1.0),
    ]
    return FlyGraph(
        neurons=neurons,
        edge_pre=np.asarray([0, 1, 2], dtype=np.int64),
        edge_post=np.asarray([2, 2, 3], dtype=np.int64),
        edge_synapses=np.asarray([10, 10, 10], dtype=np.int32),
        edge_weights=np.asarray([0.6, 0.6, 1.0], dtype=np.float32),
        receptor_names=np.asarray(["nicotinic acetylcholine"] * 3),
        config=FlyGraphConfig(),
    )


class FlyStimulationTest(unittest.TestCase):
    def test_adapter_accepts_ppo_observation_dict_without_using_action_mask(self) -> None:
        graph = diagnostic_graph()
        adapter = BattleSensoryAdapter(graph, observation_dim=2)
        observation = {
            "observation": np.asarray([0.75, -0.5], dtype=np.float32),
            "action_mask": np.zeros(26, dtype=np.int8),
        }

        stimulation = adapter.adapt(observation)

        np.testing.assert_allclose(stimulation, [0.75, 0.0])
        self.assertEqual(stimulation.shape, (2,))

    def test_fixed_simulation_reports_central_and_descending_spikes(self) -> None:
        graph = diagnostic_graph()
        report = FlySpikeSimulator(
            graph,
            steps=8,
            membrane_decay=0.8,
            sensory_gain=1.5,
            recurrent_gain=2.0,
        ).run(np.asarray([1.0, 1.0], dtype=np.float32))

        self.assertEqual(report.sensory_neurons_stimulated, 2)
        self.assertGreater(report.total_cns_spikes, 0)
        self.assertEqual(report.active_central_neurons, 1)
        self.assertEqual(report.active_descending_neurons, 1)
        self.assertEqual(report.top_descending(1)[0].cell_type, "DN-test")
        self.assertGreater(report.top_descending(1)[0].spikes, 0)


class DescendingActionDecoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root_ids = np.arange(10_000, 10_512, dtype=np.int64)
        self.spikes = np.arange(512, dtype=np.int32) % 17

    def test_illegal_actions_are_excluded_from_ranking_and_choice(self) -> None:
        decoder = DescendingActionDecoder(self.root_ids)
        scores = decoder.score(self.spikes)
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        legal_indices = [2, 7, 23]
        mask[legal_indices] = 1
        scores[5] = scores.max() + 1_000.0  # Highest raw score is deliberately illegal.

        ranked = decoder.rank_legal(scores, mask)
        choice = decoder.hypothetical_choice(scores, mask)

        self.assertEqual({item.action for item in ranked}, set(legal_indices))
        self.assertIn(choice.action, legal_indices)
        self.assertNotEqual(choice.action, 5)
        self.assertEqual(len(ranked), len(legal_indices))

    def test_identical_activity_produces_identical_scores(self) -> None:
        first_decoder = DescendingActionDecoder(self.root_ids)
        second_decoder = DescendingActionDecoder(self.root_ids)

        first_scores = first_decoder.score(self.spikes)
        repeated_scores = first_decoder.score(self.spikes.copy())
        second_scores = second_decoder.score(self.spikes)

        np.testing.assert_array_equal(first_scores, repeated_scores)
        np.testing.assert_array_equal(first_scores, second_scores)


if __name__ == "__main__":
    unittest.main()
