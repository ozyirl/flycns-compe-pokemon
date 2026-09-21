"""Build the local sparse fly-connectome artifact; this does not train a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flycns.config import FlyGraphConfig
from flycns.graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=Path, default=Path("neurons.csv.gz"))
    parser.add_argument(
        "--connections", type=Path, default=Path("connections_princeton.csv.gz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/fly_connectome.npz")
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_graph(
        args.neurons, args.connections, FlyGraphConfig(seed=args.seed)
    )
    graph.save(args.output)
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(graph.summary(), indent=2) + "\n")
    print(f"saved graph: {args.output}")
    print(f"saved manifest: {manifest}")
    print(json.dumps(graph.summary(top_cell_types=3), indent=2))


if __name__ == "__main__":
    main()
