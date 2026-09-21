from __future__ import annotations

import unittest
from pathlib import Path

from fly_viewer.catalog import load_catalog


class FlyViewerCatalogTest(unittest.TestCase):
    def test_generated_catalog_matches_selected_graph(self) -> None:
        catalog_path = Path("artifacts/fly_viewer_catalog.json.gz")
        if not catalog_path.exists():
            self.skipTest("viewer catalog has not been built")
        catalog = load_catalog(catalog_path)
        self.assertEqual(catalog["mode"], "visualization-only")
        self.assertEqual(catalog["summary"]["neurons"], 4274)
        self.assertEqual(len(catalog["neurons"]), 4274)
        self.assertEqual(len(catalog["edges"]), catalog["summary"]["edges"])
        self.assertTrue(catalog["capabilities"]["synapse_points"])
        self.assertTrue(catalog["capabilities"]["morphology"])
        self.assertIn("flyem-male-cns/v1.0", catalog["sources"]["morphology"])
        cell_types = {neuron["cell_type"] for neuron in catalog["neurons"]}
        self.assertTrue({"KCg-m", "PAM04", "DNp01"}.issubset(cell_types))


if __name__ == "__main__":
    unittest.main()
