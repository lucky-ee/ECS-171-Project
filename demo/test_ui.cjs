//test page javascript using jsdom and flask
//start flask and install jsdom first
// setup commands are in DEMO_README.md
const assert = require("node:assert/strict");
const {JSDOM, VirtualConsole} = require("./.venv/frontend-tests/node_modules/jsdom");
const base = "http://127.0.0.1:5000";

async function main() {
  const errors = [];
  const requests = [];
  const console = new VirtualConsole();
  console.on("jsdomError", error => errors.push(error));
  const dom = await JSDOM.fromURL(base, {
    resources: "usable", runScripts: "dangerously", virtualConsole: console,
    beforeParse(window) {
      //save inputs and send requests to the local server
      window.fetch = (url, options) => {
        requests.push(JSON.parse(options.body));
        return fetch(new URL(url, base), options);
      };
    }
  });
  const {window} = dom;
  const document = window.document;
  const find = selector => document.querySelector(selector);
  async function ready() {
    //wait for predictions to finish
    for (let attempt = 0; attempt < 200; attempt++) {
      if (find("#status")?.hidden && !find("#predict").disabled && !find("#results-panel").hidden && find("#selected-probability").textContent !== "—") return;
      if (find("#error") && !find("#error").hidden) throw new Error(find("#error").textContent);
      await new Promise(resolve => setTimeout(resolve, 25));
    }
    throw new Error("Page did not finish predicting.");
  }
  function input(selector, value) {
    find(selector).value = value;
    find(selector).dispatchEvent(new window.Event("input", {bubbles: true}));
  }
  try {
    //check default prediction and tree
    await ready();
    const originalProbability = find("#selected-probability").textContent;
    assert.equal(originalProbability, find("#baseline-probability").textContent);
    assert.ok(document.querySelectorAll("#tree .node").length > 0);
    const models = JSON.parse(find("#models-data").textContent);
    const clean = models.find(model => model.id === "clean");
    assert.equal(find("#tree-select").options.length, clean.tree_count);
    assert.equal(document.querySelectorAll('[role="tab"]').length, 5);
    assert.equal(document.querySelectorAll('input[type="checkbox"]').length, 0);
    assert.ok(!find("#tree-container").contains(find("#decisions")));
    assert.ok(!find("#tree-panel").contains(find("#decisions")));
    assert.ok(find("#route-panel").contains(find("#decisions")));
    assert.ok(find(".inputs").contains(find("#results-panel")));
    assert.ok(find("#tree-panel").compareDocumentPosition(find("#route-panel")) & window.Node.DOCUMENT_POSITION_FOLLOWING);
    assert.ok(find(".inputs").compareDocumentPosition(find("#route-panel")) & window.Node.DOCUMENT_POSITION_FOLLOWING);
    assert.ok(find("#results").firstElementChild.contains(find("#baseline-probability")));
    assert.ok(find("#results").lastElementChild.contains(find("#selected-probability")));
    assert.equal(window.getComputedStyle(find("#results")).gridTemplateColumns, "1fr");
    assert.equal(find("#tree-heading").textContent, "XGBoost Tree");
    assert.equal(find("#preprocessing-note"), null);
    assert.ok(!document.body.textContent.includes("Comparisons use float32 values"));
    assert.equal(window.getComputedStyle(document.body).overflow, "hidden");
    assert.equal(window.getComputedStyle(find("#tree-container")).overflow, "auto");
    assert.equal(find("header").textContent.trim(), "ECS 171 Live Demo AI4I");
    assert.equal(find("#zoom").value, "50");
    assert.equal(find("#zoom-value").textContent, "50%");
    assert.equal(find("#status").textContent, "");
    assert.match(find("#difference").textContent, /percentage points vs clean model$/);
    assert.equal(find("#metrics-difference"), null);
    assert.equal(find("#percentage-change th").textContent, "Percentage Change");
    assert.equal(document.querySelectorAll(".metrics-table tr").length, 4);
    assert.ok([...document.querySelectorAll("#percentage-change td")].every(cell => cell.textContent === "0.000%"));
    assert.equal(window.percentageChange(.9, .8), "+12.500%");
    assert.equal(window.percentageChange(.7, .8), "-12.500%");
    assert.equal(window.percentageChange(0, .8), "-100.000%");
    assert.equal(window.percentageChange(.8, 0), "—");
    assert.equal(window.percentageChange(undefined, .8), "—");
    for (const removed of ["01 / YOUR EXAMPLE", "Saved clean-test scores", "Use the same complete inputs", "These demo inputs", "Showing the current example", "Selected path", "Leaf score ≠ failure probability", "Same inputs, each model", "Missingness is in training and validation", ", same seed"]) {
      assert.ok(!document.body.textContent.includes(removed), removed);
    }

    //switch training conditions without changing the example
    for (const rate of [10, 20, 30, 40]) {
      find(`[data-rate="${rate}"]`).click();
      await ready();
      const selected = models.find(model => model.mode === "mcar" && model.missing_pct === rate && model.seed === 42);
      assert.equal(requests.at(-1).model, selected.id);
      assert.equal(find("#tree-select").options.length, selected.tree_count);
      assert.equal(find("#Torque").value, "42.8");
      assert.equal(find("#baseline-probability").textContent, originalProbability);
      assert.match(find("#selected-metrics").textContent, new RegExp(selected.metrics.average_precision.toFixed(3)));
      const metrics = ["average_precision", "precision", "recall", "f1", "accuracy"];
      const changes = [...document.querySelectorAll("#percentage-change td")];
      for (let index = 0; index < metrics.length; index++) {
        const metric = metrics[index];
        const change = 100 * (selected.metrics[metric] - selected.clean_metrics[metric]) / selected.clean_metrics[metric];
        const expected = `${change > 0 ? "+" : ""}${change.toFixed(3)}%`;
        assert.equal(changes[index].textContent, expected);
      }
    }

    //check imputation and another seed use their own model
    find("#handling-select").value = "mcar_imputed";
    find("#handling-select").dispatchEvent(new window.Event("change"));
    await ready();
    assert.equal(requests.at(-1).model, "mcar_imputed_40pct_seed42");
    assert.match(find("#tree").textContent, /standardized/);
    find("#seed-select").value = "43";
    find("#seed-select").dispatchEvent(new window.Event("change"));
    await ready();
    assert.equal(requests.at(-1).model, "mcar_imputed_40pct_seed43");
    assert.match(find("#clean-metrics").textContent, /seed 43/);
    find('[data-rate="0"]').click();
    await ready();
    assert.equal(find("#handling-select").value, "mcar");
    assert.equal(find("#seed-select").value, "42");
    find('[data-rate="40"]').click();
    await ready();
    assert.equal(requests.at(-1).model, "mcar_imputed_40pct_seed43");
    find("#tree-select").value = String(find("#tree-select").options.length - 1);
    find("#tree-select").dispatchEvent(new window.Event("change"));
    await ready();
    find('[data-rate="10"]').click();
    await ready();
    assert.equal(requests.at(-1).tree, 0);

    find('button[type="reset"]').click();
    await ready();
    assert.equal(requests.at(-1).model, "clean");
    assert.equal(find("#handling-select").disabled, true);
    assert.equal(find("#seed-select").value, "42");

    //select the last tree before early stopping
    find("#tree-select").value = String(clean.tree_count - 1);
    find("#tree-select").dispatchEvent(new window.Event("change", {bubbles: true}));
    await ready();
    assert.equal(requests.at(-1).tree, clean.tree_count - 1);
    assert.equal(find("#tree-select").value, String(clean.tree_count - 1));

    //check zoom buttons and slider limits
    find("#zoom-in").click();
    assert.equal(find("#zoom-value").textContent, "55%");
    find("#zoom-out").click();
    assert.equal(find("#zoom-value").textContent, "50%");
    find("#zoom").value = "25";
    find("#zoom").dispatchEvent(new window.Event("input"));
    assert.equal(find("#zoom-out").disabled, true);
    find("#zoom").value = "125";
    find("#zoom").dispatchEvent(new window.Event("input"));
    assert.equal(find("#zoom-in").disabled, true);

    //check invalid inputs are blocked before sending
    const count = requests.length;
    input("#Torque", "-2");
    find("#predict").click();
    assert.equal(find("#prediction-form").checkValidity(), false);
    assert.equal(requests.length, count);
    input("#Torque", "");
    find("#predict").click();
    assert.equal(requests.length, count);

    //check reset restores inputs, tree, zoom, and prediction
    find('button[type="reset"]').click();
    await ready();
    assert.equal(find("#Torque").value, "42.8");
    assert.equal(find("#Type").value, "L");
    assert.equal(find("#tree-select").value, "0");
    assert.equal(find("#zoom-value").textContent, "50%");
    assert.equal(find("#selected-probability").textContent, originalProbability);

    //compare another complete example across models
    input("#Type", "H");
    input("#Torque", "50");
    find("#predict").click();
    await ready();
    assert.equal(requests.at(-1).values.Type, "H");
    assert.equal(requests.at(-1).values.Torque, "50");

    find('[data-rate="20"]').click();
    await ready();
    find("#handling-select").value = "mcar_imputed";
    find("#handling-select").dispatchEvent(new window.Event("change"));
    await ready();
    assert.equal(requests.at(-1).values.Torque, "50");
    assert.ok(requests.every(request => !("missing" in request)));
    assert.equal(document.querySelectorAll("#decisions li.missing").length, 0);
    assert.deepEqual(errors, []);
    process.stdout.write("PASS: model tabs, handling, seeds, saved metrics, complete inputs, pinned route, validation, zoom, reset; no JavaScript errors.\n");
  } finally {
    window.close();
  }
}

main().catch(error => { process.stderr.write(`${error.stack}\n`); process.exitCode = 1; });
