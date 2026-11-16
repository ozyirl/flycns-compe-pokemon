"""Build the neuron layout the battle frontend's brain view draws, from MaleCNS somas.

Each graph neuron gets the position of its cell body from the public MaleCNS v1.0
body annotations (14 MB, download once):
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather
Reading it needs pyarrow, which the training environment does not include.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from flycns.brain_diagnostics import encode_bytes, graph_fingerprint


ANNOTATIONS_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
# MaleCNS coordinates are 8 nm voxels: x across the body, y dorsal to ventral, z
# anterior to posterior (the ventral nerve cord lies at large z).
VOXEL_NM = 8
POSITION_RANGE = 32_000
MISSING = -32_768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help=f"Local copy of {ANNOTATIONS_URL}")
    parser.add_argument("--graph", type=Path, default=Path("artifacts/fly_connectome.npz"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def soma_positions(annotations: Path) -> dict[int, list[int]]:
    try:
        from pyarrow import feather
    except ImportError as error:
        raise SystemExit("Reading the annotations needs pyarrow: pip install pyarrow") from error
    table = feather.read_table(annotations, columns=["bodyId", "somaLocation", "tosomaLocation"])
    bodies, somas, tosomas = (table.column(name).to_pylist() for name in table.column_names)
    # Neurons without a traced soma sometimes record where the soma would be instead.
    return {body: soma or tosoma for body, soma, tosoma in zip(bodies, somas, tosomas) if soma or tosoma}


def main() -> None:
    args = parse_args()
    with np.load(args.graph, allow_pickle=False) as data:
        root_ids = data["root_ids"].astype(np.int64)
        population_names = data["populations"].astype(str)
    positions = soma_positions(args.annotations)

    populations = sorted(set(population_names))
    raw = np.full((root_ids.size, 3), np.nan)
    for index, root_id in enumerate(root_ids):
        if int(root_id) in positions:
            raw[index] = positions[int(root_id)]
    placed = ~np.isnan(raw[:, 0])
    low, high = np.nanmin(raw, axis=0), np.nanmax(raw, axis=0)
    centre, half_extent = (low + high) / 2, (high - low).max() / 2
    scaled = np.full(raw.shape, MISSING, dtype="<i2")
    scaled[placed] = np.round((raw[placed] - centre) / half_extent * POSITION_RANGE)

    layout = {
        "source": ANNOTATIONS_URL,
        "graph": graph_fingerprint(root_ids),
        "neuron_count": int(root_ids.size),
        "placed": int(placed.sum()),
        # Positions span -POSITION_RANGE..POSITION_RANGE along the longest axis.
        "position_range": POSITION_RANGE,
        "missing": MISSING,
        "extent_um": round(float((high - low).max()) * VOXEL_NM / 1000, 1),
        "populations": populations,
        "population": encode_bytes(np.array([populations.index(name) for name in population_names])),
        # Little-endian int16 x, y, z per neuron in graph order; MISSING where unplaced.
        "positions": base64.b64encode(scaled.tobytes()).decode("ascii"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(layout, separators=(",", ":")), encoding="utf-8")
    print(f"saved brain layout: {args.output}")
    print(f"{layout['placed']:,} of {layout['neuron_count']:,} neurons placed · {layout['extent_um']} µm across")


if __name__ == "__main__":
    main()
