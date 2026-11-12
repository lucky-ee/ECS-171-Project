"use strict";
//get form inputs and output elements
const form = document.querySelector("#prediction-form");
const treeSelect = document.querySelector("#tree-select");
const statusLine = document.querySelector("#status");
const errorLine = document.querySelector("#error");
const svg = document.querySelector("#tree");
const models = JSON.parse(document.querySelector("#models-data").textContent);
const cleanModel = models.find(model => model.id === "clean");
const handlingSelect = document.querySelector("#handling-select");
const seedSelect = document.querySelector("#seed-select");
let selectedModel = cleanModel;
let selectedRate = 0;
let selectedSeed = cleanModel.seed;
let selectedHandling = "mcar";
let requestNumber = 0;
let treeSize = {width: 1, height: 1};
let lastPrediction = null;

const percent = value => `${(100 * value).toFixed(3)}%`;
const score = value => Number(value).toPrecision(7);
const setText = (id, text) => { document.getElementById(id).textContent = text; };

function setStatus(message) {
  //only show status text when there is something to report
  statusLine.textContent = message;
  statusLine.hidden = !message;
}

function readInputs() {
  //get complete inputs for the selected model
  const values = {};
  for (const input of form.querySelectorAll("[name]")) values[input.name] = input.value;
  return {values, tree: Number(treeSelect.value), model: selectedModel.id};
}

function percentageChange(selected, clean) {
  //compare each score with the clean baseline
  if (!Number.isFinite(selected) || !Number.isFinite(clean) || clean === 0) return "—";
  const change = 100 * (selected - clean) / clean;
  const rounded = change.toFixed(3);
  return `${Number(rounded) > 0 ? "+" : ""}${Number(rounded) === 0 ? "0.000" : rounded}%`;
}

function showModelInfo(model) {
  //show saved test scores for the selected training condition
  setText("model-description", `${model.label} · seed ${model.seed} · ${model.tree_count} trees`);
  document.querySelector("#model-description").title = model.filename;
  function metricRow(id, label, metrics) {
    const row = document.getElementById(id);
    row.replaceChildren();
    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = label;
    row.append(heading);
    for (const metric of ["average_precision", "precision", "recall", "f1", "accuracy"]) {
      const cell = document.createElement("td");
      cell.textContent = !metrics ? "Unavailable" : metric === "average_precision" ? metrics[metric].toFixed(3) : percent(metrics[metric]);
      row.append(cell);
    }
  }
  metricRow("clean-metrics", `Clean training · seed ${model.seed}`, model.clean_metrics);
  metricRow("selected-metrics", `${model.label} · seed ${model.seed}`, model.metrics);
  const changes = document.querySelector("#percentage-change");
  changes.replaceChildren();
  const heading = document.createElement("th");
  heading.scope = "row";
  heading.textContent = "Percentage Change";
  heading.title = "(selected model − clean model) / clean model × 100";
  changes.append(heading);
  for (const metric of ["average_precision", "precision", "recall", "f1", "accuracy"]) {
    const cell = document.createElement("td");
    cell.textContent = percentageChange(model.metrics?.[metric], model.clean_metrics?.[metric]);
    changes.append(cell);
  }
}

function selectModel(rate) {
  //keep form values when switching the training condition
  const candidate = rate === 0 ? cleanModel : models.find(model => model.missing_pct === rate && model.mode === selectedHandling && model.seed === selectedSeed);
  if (!candidate) {
    errorLine.textContent = "That model has not been saved. Choose another handling method or seed.";
    errorLine.hidden = false;
    return;
  }
  selectedModel = candidate;
  selectedRate = rate;
  handlingSelect.disabled = seedSelect.disabled = rate === 0;
  handlingSelect.value = rate === 0 ? "mcar" : selectedHandling;
  seedSelect.value = String(rate === 0 ? cleanModel.seed : selectedSeed);
  for (const tab of document.querySelectorAll("[data-rate]")) {
    const active = Number(tab.dataset.rate) === rate;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  }
  document.querySelector("#experiment-results").setAttribute("aria-labelledby", `model-tab-${rate}`);
  treeSelect.replaceChildren();
  for (let index = 0; index < candidate.tree_count; index++) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${index + 1} of ${candidate.tree_count}`;
    treeSelect.append(option);
  }
  ++requestNumber;
  lastPrediction = null;
  document.querySelector("#predict").disabled = false;
  document.querySelector("#results-panel").hidden = true;
  document.querySelector("#tree-panel").hidden = true;
  document.querySelector("#route-panel").hidden = true;
  showModelInfo(candidate);
  setStatus("Model changed. Enter valid inputs to show its prediction.");
  predict();
}

function svgElement(tag, attributes, text) {
  //make an svg element for the tree
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}

function applyZoom() {
  //resize tree using the zoom slider
  const scale = Number(document.querySelector("#zoom").value) / 100;
  svg.style.width = `${treeSize.width * scale}px`;
  svg.style.height = `${treeSize.height * scale}px`;
  setText("zoom-value", `${Math.round(scale * 100)}%`);
  document.querySelector("#zoom-out").disabled = Number(document.querySelector("#zoom").value) <= 25;
  document.querySelector("#zoom-in").disabled = Number(document.querySelector("#zoom").value) >= 125;
}

function drawTree(tree, path) {
  //clear the old tree and place the new nodes
  svg.replaceChildren();
  const nodes = [], edges = [];
  let leafNumber = 0, maxDepth = 0;
  function layout(node, depth) {
    //space out leaves and center parents above their children
    node.y = 22 + depth * 150;
    maxDepth = Math.max(maxDepth, depth);
    if (node.children) {
      node.children.forEach(child => { layout(child, depth + 1); edges.push({parent: node, child}); });
      node.x = (node.children[0].x + node.children[1].x) / 2;
    } else node.x = 126 + leafNumber++ * 244;
    nodes.push(node);
  }
  layout(tree, 0);
  treeSize = {width: Math.max(leafNumber * 244, 260), height: maxDepth * 150 + 126};
  svg.setAttribute("viewBox", `0 0 ${treeSize.width} ${treeSize.height}`);
  //draw branches and highlight the selected path
  for (const {parent, child} of edges) {
    const decision = path.decisions.find(item => item.node === parent.id && item.child === child.id);
    const ordinary = child.id === parent.yes ? "yes (<)" : "no (≥)";
    const isDefault = child.id === parent.missing;
    svg.append(svgElement("path", {
      d: `M ${parent.x} ${parent.y + 78} L ${child.x} ${child.y}`,
      class: `edge${decision ? " active" : ""}${decision?.branch === "missing" ? " missing" : ""}`
    }));
    const label = `${ordinary}${isDefault ? " / missing" : ""} → ${child.id}`;
    svg.append(svgElement("text", {x: (parent.x + child.x) / 2, y: (parent.y + 78 + child.y) / 2 - 7,
      "text-anchor": "middle", class: "edge-label"}, label));
  }
  for (const node of nodes) {
    //show split feature and threshold
    // show score for leaf nodes
    const group = svgElement("g", {class: `node${path.nodes.includes(node.id) ? " active" : ""}`});
    group.append(svgElement("rect", {x: node.x - 114, y: node.y, width: 228, height: 78, rx: 8}));
    group.append(svgElement("text", {x: node.x, y: node.y + 17, "text-anchor": "middle", class: "node-id"}, `Node ${node.id}`));
    const leaf = Object.hasOwn(node, "leaf");
    group.append(svgElement("text", {x: node.x, y: node.y + 40, "text-anchor": "middle"}, leaf ? "Leaf contribution" : node.feature));
    group.append(svgElement("text", {x: node.x, y: node.y + 62, "text-anchor": "middle"}, leaf ? score(node.leaf) : `< ${Number(node.threshold).toPrecision(9)} ?`));
    svg.append(group);
  }
  applyZoom();
  const container = document.querySelector("#tree-container");
  container.scrollLeft = tree.x * Number(document.querySelector("#zoom").value) / 100 - container.clientWidth / 2;
  container.scrollTop = 0;
}

function render(data) {
  //update predictions and path explanation
  lastPrediction = data;
  document.querySelector("#results-panel").hidden = false;
  document.querySelector("#tree-panel").hidden = false;
  document.querySelector("#route-panel").hidden = false;
  showModelInfo(data.model);
  for (const key of ["baseline", "selected"]) {
    const result = key === "baseline" ? data.clean_reference : data.selected;
    setText(`${key}-probability`, percent(result.probability));
    setText(`${key}-class`, result.failure ? "Failure (1)" : "No failure (0)");
  }
  setText("baseline-model-label", `Clean-trained model · seed ${data.clean_reference.seed}`);
  setText("selected-model-label", `${data.model.label} · seed ${data.model.seed}`);
  setText("baseline-threshold", `Failure threshold: ${percent(data.clean_reference.threshold)}`);
  setText("selected-threshold", `Failure threshold: ${percent(data.threshold)}`);
  const change = 100 * (data.selected.probability - data.clean_reference.probability);
  setText("difference", `${change >= 0 ? "+" : ""}${change.toFixed(3)} percentage points vs clean model`);
  const path = data.selected.path;
  setText("leaf-summary", `· leaf ${path.leaf_id} · score ${score(path.leaf_score)}`);
  const decisions = document.querySelector("#decisions");
  decisions.replaceChildren();
  for (const decision of path.decisions) {
    const item = document.createElement("li");
    item.textContent = decision.explanation;
    if (decision.branch === "missing") item.className = "missing";
    decisions.append(item);
  }
  const leaf = document.createElement("li");
  leaf.textContent = `Reach leaf ${path.leaf_id}: contribution ${score(path.leaf_score)} to the ensemble margin (not a probability).`;
  decisions.append(leaf);
  drawTree(data.tree, path);
}

async function predict() {
  //check inputs before making predictions
  if (!form.reportValidity()) return;
  const currentRequest = ++requestNumber;
  errorLine.hidden = true;
  setStatus("Calculating…");
  document.querySelector("#predict").disabled = true;
  try {
    //send inputs to flask and show predictions
    const response = await fetch("/api/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(readInputs())});
    const data = await response.json();
    if (currentRequest !== requestNumber) return;
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    render(data);
    setStatus("");
  } catch (error) {
    if (currentRequest !== requestNumber) return;
    errorLine.textContent = error.message;
    errorLine.hidden = false;
    setStatus("Prediction failed. Previous results have not been updated.");
  } finally {
    if (currentRequest === requestNumber) document.querySelector("#predict").disabled = false;
  }
}

form.addEventListener("submit", event => { event.preventDefault(); predict(); });
form.addEventListener("input", () => {
  //ignore old predictions when inputs change
  ++requestNumber;
  document.querySelector("#predict").disabled = false;
  setStatus("Inputs changed. Press Predict to update the results and route.");
});
form.addEventListener("reset", () => {
  //reset tree and zoom with the form
  ++requestNumber;
  setStatus("Restoring the default model and inputs…");
  document.querySelector("#zoom").value = "50";
  handlingSelect.value = "mcar";
  selectedHandling = "mcar";
  selectedSeed = cleanModel.seed;
  //wait for form reset before predicting again
  setTimeout(() => selectModel(0), 0);
});
treeSelect.addEventListener("change", () => {
  //check inputs before changing trees
  if (!form.checkValidity()) {
    if (lastPrediction) treeSelect.value = String(lastPrediction.tree_index);
    form.reportValidity();
    return;
  }
  predict();
});
document.querySelector("#zoom").addEventListener("input", applyZoom);
document.querySelector("#zoom-out").addEventListener("click", () => {
  document.querySelector("#zoom").stepDown();
  applyZoom();
});
document.querySelector("#zoom-in").addEventListener("click", () => {
  document.querySelector("#zoom").stepUp();
  applyZoom();
});
for (const tab of document.querySelectorAll("[data-rate]")) tab.addEventListener("click", () => selectModel(Number(tab.dataset.rate)));
document.querySelector(".model-tabs").addEventListener("keydown", event => {
  //support arrow keys for the model tabs
  const tabs = [...document.querySelectorAll("[data-rate]")];
  const index = tabs.indexOf(document.activeElement);
  if (index < 0 || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  tabs[next].focus();
  tabs[next].click();
});
handlingSelect.addEventListener("change", () => {
  selectedHandling = handlingSelect.value;
  selectModel(selectedRate);
});
seedSelect.addEventListener("change", () => {
  selectedSeed = Number(seedSelect.value);
  selectModel(selectedRate);
});
selectModel(0);
