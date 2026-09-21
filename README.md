# FlyCNS Pokémon Showdown

This project trains Pokémon Showdown policies and is now scaffolded to compare a
plain MLP baseline with a sparse policy derived from the Janelia MaleCNS
connectome.

## Fly architecture

The initial fly model uses five visible populations:

- sensory and visual-projection neurons receive the 186 battle features;
- Kenyon cells provide a learning-and-memory population;
- dopamine, serotonin, and octopamine cells provide a modulatory population;
- central-brain, optic-lobe, and VNC intrinsic cells provide recurrent relays;
- descending neurons form the action-facing readout.

Connections and synapse counts come from the local MaleCNS v1.0 data. Presynaptic
neurotransmitters also come from the annotations. Notably, the source predictor labels
most Kenyon cells as DA; the scaffold preserves that value but uses the curated `DAN`
cell class—not DA prediction alone—to define the modulatory population. The receptor
names and effects are explicit modeling assumptions because the supplied CSV files do
not contain measured postsynaptic receptor expression. The graph is fixed by default;
the Pokémon-to-fly input adapter and fly-to-PPO readout are trainable.

Build and inspect the graph without training:

```bash
.venv/bin/python build_fly.py
.venv/bin/python inspect_fly.py
```

The build produces a local `artifacts/fly_connectome.npz` plus a readable manifest.
The default artifact contains roughly 4,000 deterministically selected neurons. Every
cell retains its real root ID, population, type, transmitter, and confidence so a UI
can expose the active cells rather than treating the policy as a black box.

The extractor also supports opt-in activity telemetry through
`FlyConnectomeFeaturesExtractor(..., telemetry=True)` and
`activity_snapshot()`. This is the data hook for a later brain-activity view like the
visual references; the current source tables do not include 3D coordinates.

Training is wired but should only be started deliberately:

```bash
.venv/bin/python train.py --architecture fly --output models/fly-baseline
```

## Visualization-only mode

The standalone viewer never imports the Showdown environment, policies, PPO, or
training entry point. It streams real public MaleCNS v1.0 morphology into
Neuroglancer and uses only the selected graph catalog for search and connection
highlighting.

```bash
.venv/bin/python build_fly_viewer.py
.venv/bin/python visualize_fly.py --open
```

Then visit `http://127.0.0.1:8765`. The control panel can search root IDs and cell
types, color the five populations, highlight incoming/outgoing partners, and toggle
the public EM, whole-CNS, neuropil, and selected-neuron synapse layers. Synapse points
stream from the public v1.0 annotation layer; the multi-gigabyte point table is not
downloaded into this repository.

Remote scientific layers:

- MaleCNS v1.0 neuron segmentation, meshes, skeletons, and properties:
  `precomputed://gs://flyem-male-cns/v1.0/segmentation`
- MaleCNS CLAHE EM imagery: `precomputed://gs://flyem-male-cns/em/em-clahe-jpeg`
- Brain/VNC shells, neuropils, and synapses from the official MaleCNS Neuroglancer scene
