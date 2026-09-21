"""Inspect the cells, transmitters, and modeled receptors used by a fly graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flycns.graph import POPULATION_ORDER, FlyGraph
from flycns.receptors import RECEPTOR_CHANNELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "graph", nargs="?", type=Path, default=Path("artifacts/fly_connectome.npz")
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--cells", type=int, default=5, help="sample cells per population")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = FlyGraph.load(args.graph)
    if args.json:
        print(json.dumps(graph.summary(), indent=2))
        return

    summary = graph.summary()
    print(
        f"Fly graph: {summary['neurons']:,} neurons, {summary['edges']:,} edges, "
        f"{summary['synapses']:,} source synapses"
    )
    print("\nPopulations")
    for population in POPULATION_ORDER:
        print(f"  {population:12} {summary['populations'].get(population, 0):>5,}")

    print("\nModeled receptor channels (assumptions, not measured expression)")
    receptor_edges = summary["modeled_receptor_edges"]
    for channel in RECEPTOR_CHANNELS.values():
        count = receptor_edges.get(channel.receptor, 0)
        print(
            f"  {channel.transmitter:7} -> {channel.receptor:30} "
            f"{channel.effect:26} gain={channel.gain:>5.2f} edges={count:,}"
        )

    print("\nSample selected cells")
    for population in POPULATION_ORDER:
        print(f"  [{population}]")
        cells = [n for n in graph.neurons if n.population == population][: args.cells]
        for neuron in cells:
            identity = neuron.cell_type or neuron.cell_class or "untyped"
            print(
                f"    {neuron.root_id:<10} {identity:<24.24} "
                f"NT={neuron.neurotransmitter:<7} confidence={neuron.nt_confidence:.2f}"
            )


if __name__ == "__main__":
    main()
