from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from flycns.config import FlyGraphConfig
from flycns.graph import FlyGraph, build_graph
from flycns.model import FlyConnectomeFeaturesExtractor


class FlyGraphTest(unittest.TestCase):
    def test_build_save_load_and_forward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            neurons_path = root / "neurons.csv.gz"
            connections_path = root / "connections.csv.gz"
            graph_path = root / "graph.npz"
            fields = [
                "Root ID",
                "Predicted NT type",
                "Predicted NT confidence",
                "Super Class",
                "Class",
                "Primary Cell Type",
            ]
            rows = [
                ["1", "ACH", "1", "cb_sensory", "olfactory", "ORN"],
                ["2", "ACH", "1", "cb_intrinsic", "Kenyon_Cell", "KC"],
                ["3", "DA", "1", "cb_intrinsic", "DAN", "PAM"],
                ["4", "GABA", "1", "cb_intrinsic", "CX", "central"],
                ["5", "ACH", "1", "descending_neuron", "", "DN"],
            ]
            with gzip.open(neurons_path, "wt", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            with gzip.open(connections_path, "wt", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"]
                )
                writer.writerows(
                    [["1", "2", "A", "5", ""], ["2", "4", "B", "3", ""], ["4", "5", "C", "4", ""]]
                )

            config = FlyGraphConfig(
                sensory_neurons=1,
                kenyon_neurons=1,
                modulatory_neurons=1,
                central_neurons=2,
                descending_neurons=1,
            )
            graph = build_graph(neurons_path, connections_path, config)
            graph.save(graph_path)
            loaded = FlyGraph.load(graph_path)
            self.assertEqual(len(loaded.neurons), 5)
            self.assertEqual(loaded.edge_pre.size, 3)

            observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(-1, 1, (6,), dtype=np.float32),
                    "action_mask": spaces.Box(0, 1, (26,), dtype=np.int8),
                }
            )
            extractor = FlyConnectomeFeaturesExtractor(
                observation_space,
                graph_path=str(graph_path),
                features_dim=8,
                telemetry=True,
            )
            output = extractor(
                {
                    "observation": torch.zeros((2, 6)),
                    "action_mask": torch.ones((2, 26)),
                }
            )
            self.assertEqual(tuple(output.shape), (2, 8))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(extractor.activity_snapshot())


if __name__ == "__main__":
    unittest.main()
