"""Build visualization-only metadata for the selected Fly CNS graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from fly_viewer.catalog import build_catalog, save_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("artifacts/fly_connectome.npz"))
    parser.add_argument(
        "--connections", type=Path, default=Path("connections_princeton.csv.gz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/fly_viewer_catalog.json.gz")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = build_catalog(args.graph, args.connections)
    save_catalog(catalog, args.output)
    summary = catalog["summary"]
    print(f"saved viewer catalog: {args.output}")
    print(
        f"{summary['neurons']:,} neurons · {summary['edges']:,} edges · "
        f"{summary['synapses']:,} synapses"
    )
    print("mode: visualization-only (no training or decision code)")


if __name__ == "__main__":
    main()
