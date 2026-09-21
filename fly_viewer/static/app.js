"use strict";

const NGL_BASE = "https://neuroglancer-demo.appspot.com/";
const POPULATION_ORDER = ["sensory", "kenyon", "modulatory", "central", "descending"];
const BRAIN_NEUROPIL_SEGMENTS = [
  ...Array.from({length: 86}, (_, index) => String(index + 1)), "93", "94", "96",
];
const VNC_NEUROPIL_SEGMENTS = Array.from({length: 23}, (_, index) => String(index + 5));
const state = {
  catalog: null,
  neuronsById: new Map(),
  incoming: new Map(),
  outgoing: new Map(),
  enabledPopulations: new Set(POPULATION_ORDER),
  matches: [],
  focusIds: new Set(),
};

const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]);
const labelFor = (neuron) => neuron.cell_type || neuron.cell_class || neuron.super_class || `Neuron ${neuron.id}`;

async function init() {
  const response = await fetch("/api/catalog");
  if (!response.ok) throw new Error(`Catalog request failed: ${response.status}`);
  state.catalog = await response.json();
  for (const neuron of state.catalog.neurons) state.neuronsById.set(String(neuron.id), neuron);
  for (const edge of state.catalog.edges) {
    const pre = String(edge.pre), post = String(edge.post);
    if (!state.outgoing.has(pre)) state.outgoing.set(pre, []);
    if (!state.incoming.has(post)) state.incoming.set(post, []);
    state.outgoing.get(pre).push(edge);
    state.incoming.get(post).push(edge);
  }
  for (const edges of [...state.incoming.values(), ...state.outgoing.values()]) {
    edges.sort((a, b) => b.synapses - a.synapses);
  }
  renderSummary();
  renderPopulations();
  renderLegend();
  bindEvents();
  updateScene();
}

function renderSummary() {
  const {neurons, edges, synapses} = state.catalog.summary;
  $("datasetSummary").textContent = `${neurons.toLocaleString()} neurons · ${edges.toLocaleString()} edges · ${synapses.toLocaleString()} synapses`;
}

function renderPopulations() {
  const counts = state.catalog.summary.populations;
  $("populationToggles").innerHTML = POPULATION_ORDER.map((population) => `
    <label class="population-row">
      <input type="checkbox" data-population="${population}" checked />
      <span class="dot" style="background:${state.catalog.population_colors[population]}"></span>
      <span>${population[0].toUpperCase()}${population.slice(1)}</span>
      <span class="count">${(counts[population] || 0).toLocaleString()}</span>
    </label>`).join("");
}

function renderLegend() {
  $("legend").innerHTML = POPULATION_ORDER.map((population) => `
    <span class="legend-item"><span class="dot" style="background:${state.catalog.population_colors[population]}"></span>${population}</span>`).join("");
}

function bindEvents() {
  let timer;
  $("search").addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(runSearch, 100);
  });
  $("clearSearch").addEventListener("click", () => {
    $("search").value = "";
    runSearch();
  });
  $("showMatches").addEventListener("click", () => selectMany(state.matches.map((n) => String(n.id))));
  $("clearSelection").addEventListener("click", clearSelection);
  $("reloadScene").addEventListener("click", updateScene);
  $("openExternal").addEventListener("click", (event) => {
    if (!event.currentTarget.href) event.preventDefault();
  });
  $("populationToggles").addEventListener("change", (event) => {
    const population = event.target.dataset.population;
    if (!population) return;
    event.target.checked ? state.enabledPopulations.add(population) : state.enabledPopulations.delete(population);
    updateScene();
  });
  $("toggleAll").addEventListener("click", () => {
    const turnOn = state.enabledPopulations.size !== POPULATION_ORDER.length;
    state.enabledPopulations = new Set(turnOn ? POPULATION_ORDER : []);
    document.querySelectorAll("[data-population]").forEach((input) => input.checked = turnOn);
    $("toggleAll").textContent = turnOn ? "Hide all" : "Show all";
    updateScene();
  });
  for (const id of ["brainToggle", "neuropilToggle", "emToggle", "synapseToggle", "incomingToggle", "outgoingToggle"]) {
    $(id).addEventListener("change", updateScene);
  }
  $("neuroglancer").addEventListener("load", () => {
    $("viewerLoading").classList.add("hidden");
    $("sceneStatus").textContent = "Scene loaded";
  });
}

function runSearch() {
  const query = $("search").value.trim().toLowerCase();
  if (!query) {
    state.matches = [];
    $("results").innerHTML = "";
    $("matchCount").textContent = `Type to search ${state.catalog.neurons.length.toLocaleString()} cells`;
    $("showMatches").hidden = true;
    return;
  }
  const exact = [], partial = [];
  for (const neuron of state.catalog.neurons) {
    const values = [String(neuron.id), neuron.cell_type, neuron.cell_class, neuron.super_class, neuron.population].map((v) => (v || "").toLowerCase());
    if (values.some((value) => value === query)) exact.push(neuron);
    else if (values.some((value) => value.includes(query))) partial.push(neuron);
  }
  state.matches = [...exact, ...partial];
  $("matchCount").textContent = `${state.matches.length.toLocaleString()} match${state.matches.length === 1 ? "" : "es"}`;
  $("showMatches").hidden = state.matches.length < 2;
  $("results").innerHTML = state.matches.slice(0, 80).map(resultMarkup).join("") || `<p class="muted">No selected neuron matches “${escapeHtml(query)}”.</p>`;
  $("results").querySelectorAll("button[data-id]").forEach((button) => button.addEventListener("click", () => selectOne(button.dataset.id)));
}

function resultMarkup(neuron) {
  return `<button class="result" data-id="${neuron.id}">
    <span class="dot" style="background:${state.catalog.population_colors[neuron.population]}"></span>
    <span><strong>${escapeHtml(labelFor(neuron))}</strong><small>${escapeHtml(neuron.cell_class || neuron.super_class)} · ${escapeHtml(neuron.population)}</small></span>
    <code>${neuron.id}</code>
  </button>`;
}

function selectOne(id) {
  state.focusIds = new Set([String(id)]);
  renderSelection();
  updateScene();
}

function selectMany(ids) {
  state.focusIds = new Set(ids);
  renderSelection();
  updateScene();
}

function clearSelection() {
  state.focusIds.clear();
  renderSelection();
  updateScene();
}

function renderSelection() {
  const panel = $("selectionPanel"), content = $("selectionContent"), connections = $("connections");
  $("clearSelection").hidden = state.focusIds.size === 0;
  if (!state.focusIds.size) {
    panel.classList.add("empty");
    content.innerHTML = "No neuron selected.";
    connections.innerHTML = "";
    return;
  }
  panel.classList.remove("empty");
  if (state.focusIds.size > 1) {
    const populations = {};
    for (const id of state.focusIds) {
      const neuron = state.neuronsById.get(id);
      populations[neuron.population] = (populations[neuron.population] || 0) + 1;
    }
    content.innerHTML = `<div class="selection-card"><strong>${state.focusIds.size.toLocaleString()} matched cells</strong><p class="hint">${Object.entries(populations).map(([p,n]) => `${p}: ${n}`).join(" · ")}</p></div>`;
  } else {
    const id = [...state.focusIds][0], neuron = state.neuronsById.get(id);
    content.innerHTML = `<div class="selection-card">
      <div class="selection-title"><span class="dot" style="background:${state.catalog.population_colors[neuron.population]}"></span><strong>${escapeHtml(labelFor(neuron))}</strong></div>
      <dl class="meta-grid">
        <dt>Root ID</dt><dd>${neuron.id}</dd><dt>Population</dt><dd>${escapeHtml(neuron.population)}</dd>
        <dt>Class</dt><dd>${escapeHtml(neuron.cell_class || "—")}</dd><dt>Super class</dt><dd>${escapeHtml(neuron.super_class || "—")}</dd>
        <dt>Transmitter</dt><dd>${escapeHtml(neuron.neurotransmitter)} (${Math.round(neuron.nt_confidence * 100)}% predicted confidence)</dd>
      </dl></div>`;
  }
  renderConnections();
}

function renderConnections() {
  const incoming = mergedEdges("incoming"), outgoing = mergedEdges("outgoing");
  $("connections").innerHTML = connectionSection("Incoming", incoming, "pre") + connectionSection("Outgoing", outgoing, "post");
  $("connections").querySelectorAll("button[data-id]").forEach((button) => button.addEventListener("click", () => selectOne(button.dataset.id)));
}

function mergedEdges(direction) {
  const result = new Map();
  for (const id of state.focusIds) {
    const edges = (direction === "incoming" ? state.incoming : state.outgoing).get(id) || [];
    for (const edge of edges) {
      const key = `${edge.pre}:${edge.post}`;
      if (!result.has(key)) result.set(key, edge);
    }
  }
  return [...result.values()].sort((a, b) => b.synapses - a.synapses);
}

function connectionSection(title, edges, partnerKey) {
  const items = edges.slice(0, 30).map((edge) => {
    const partner = state.neuronsById.get(String(edge[partnerKey]));
    const regions = edge.neuropils.slice(0, 3).map(([name, count]) => `${name} ${count}`).join(" · ") || "region unassigned";
    return `<button class="connection" data-id="${edge[partnerKey]}"><span><strong>${escapeHtml(labelFor(partner))}</strong><small>${escapeHtml(regions)} · ${escapeHtml(edge.receptor)}</small></span><span class="weight">${edge.synapses} syn</span></button>`;
  }).join("");
  return `<div class="connection-heading"><span>${title}</span><span>${edges.length.toLocaleString()} edges</span></div><div class="connection-list">${items || '<span class="muted">No selected-graph connections.</span>'}</div>`;
}

function updateScene() {
  if (!state.catalog) return;
  const scene = buildScene();
  const url = `${NGL_BASE}#!${encodeURIComponent(JSON.stringify(scene))}`;
  $("openExternal").href = url;
  $("viewerLoading").classList.remove("hidden");
  $("sceneStatus").textContent = `Loading ${visibleSegmentCount(scene).toLocaleString()} morphologies…`;
  $("neuroglancer").src = url;
}

function buildScene() {
  const sources = state.catalog.sources;
  const incomingIds = new Set(), outgoingIds = new Set();
  for (const edge of mergedEdges("incoming")) incomingIds.add(String(edge.pre));
  for (const edge of mergedEdges("outgoing")) outgoingIds.add(String(edge.post));
  for (const id of state.focusIds) { incomingIds.delete(id); outgoingIds.delete(id); }
  if (!$("incomingToggle").checked) incomingIds.clear();
  if (!$("outgoingToggle").checked) outgoingIds.clear();
  const highlighted = new Set([...state.focusIds, ...incomingIds, ...outgoingIds]);
  const layers = [];
  if ($("emToggle").checked) layers.push({
    type:"image",
    source:{url:sources.em, subsources:{default:true}, enableDefaultSubsources:false},
    name:"EM imagery",
    visible:true,
    opacity:0.55,
  });
  if ($("brainToggle").checked) {
    layers.push(referenceLayer("brain outline", sources.brain_outline, ["1", "2", "3"], "#8fa0aa", 6));
    layers.push(referenceLayer("VNC outline", sources.vnc_outline, ["1"], "#788994", 6));
  }
  if ($("neuropilToggle").checked) {
    layers.push(referenceLayer("brain neuropils", sources.brain_neuropils, BRAIN_NEUROPIL_SEGMENTS, "#5f7180", 3, 2904176341));
    layers.push(referenceLayer("VNC neuropils", sources.vnc_neuropils, VNC_NEUROPIL_SEGMENTS, "#5f7180", 3, 190917043));
  }
  for (const population of POPULATION_ORDER) {
    if (!state.enabledPopulations.has(population)) continue;
    const segments = state.catalog.neurons.filter((n) => n.population === population && !highlighted.has(String(n.id))).map((n) => String(n.id));
    if (segments.length) layers.push(neuronLayer(population, segments, state.catalog.population_colors[population], sources, state.focusIds.size ? 0.16 : 0.68));
  }
  if (incomingIds.size) layers.push(neuronLayer("incoming partners", [...incomingIds], "#d49cff", sources, 0.95));
  if (outgoingIds.size) layers.push(neuronLayer("outgoing partners", [...outgoingIds], "#ffcf5c", sources, 0.95));
  if (state.focusIds.size) layers.push(neuronLayer("selected neuron", [...state.focusIds], "#ffffff", sources, 1));
  if ($("synapseToggle").checked && state.focusIds.size) {
    layers.push(synapseLayer("outgoing synapses", sources.synapses, "body_pre", "#ffcf5c"));
    layers.push(synapseLayer("incoming synapses", sources.synapses, "body_post", "#d49cff"));
  }
  const selectedLayer = state.focusIds.size ? "selected neuron" : (layers.find((layer) => layer.type === "segmentation" && POPULATION_ORDER.includes(layer.name))?.name || layers[0]?.name);
  return {
    dimensions:{x:[8e-9,"m"], y:[8e-9,"m"], z:[8e-9,"m"]},
    position:[48686.5,27515.5,24721.5],
    layers,
    projectionScale:134522.22897188037,
    showAxisLines:false,
    showSlices:$("emToggle").checked,
    gpuMemoryLimit:2000000000,
    systemMemoryLimit:4000000000,
    selectedLayer:selectedLayer ? {layer:selectedLayer, visible:true} : undefined,
    layout:$("emToggle").checked ? "4panel" : "3d",
  };
}

function neuronLayer(name, segments, color, sources, alpha) {
  const source = [
    {url:sources.morphology, subsources:{default:true, mesh:true}, enableDefaultSubsources:false},
    ...sources.morphology_properties.map((url) => ({url, subsources:{default:true}, enableDefaultSubsources:false})),
  ];
  return {type:"segmentation", source, tab:"segments", segmentDefaultColor:color, objectAlpha:alpha, segments, name};
}

function referenceLayer(name, source, segments, color, silhouette, colorSeed) {
  return {
    type:"segmentation",
    source:{url:source, subsources:{default:true, properties:true, mesh:true}, enableDefaultSubsources:false},
    pick:false,
    tab:"rendering",
    selectedAlpha:0,
    saturation:colorSeed ? 1 : 0,
    meshSilhouetteRendering:silhouette,
    segmentDefaultColor:color,
    colorSeed,
    segments,
    name,
  };
}

function synapseLayer(name, source, property, color) {
  return {
    type:"annotation",
    source,
    annotationColor:color,
    linkedSegmentationLayer:{body_pre:"selected neuron", body_post:"selected neuron"},
    filterBySegmentation:[property],
    name,
  };
}

function visibleSegmentCount(scene) {
  return scene.layers.reduce((sum, layer) => sum + (layer.segments?.length || 0), 0);
}

init().catch((error) => {
  console.error(error);
  $("sceneStatus").textContent = "Viewer failed to initialize";
  $("viewerLoading").innerHTML = `<strong>Could not load the local catalog</strong><span>${escapeHtml(error.message)}</span>`;
});
