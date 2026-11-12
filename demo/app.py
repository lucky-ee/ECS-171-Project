#ai4i live demo using the saved model
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from artifact_loader import load_artifact

#paths for the saved models and test results
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/xgboost/ai4i_clean.joblib"
#default example and labels
FIELDS = [
    {"name": "Type", "label": "Product Type", "default": "L"},
    {"name": "Air_temperature", "label": "Air temperature (K)", "default": 298.1},
    {"name": "Process_temperature", "label": "Process temperature (K)", "default": 308.6},
    {"name": "Rotational_speed", "label": "Rotational speed (rpm)", "default": 1551},
    {"name": "Torque", "label": "Torque (Nm)", "default": 42.8},
    {"name": "Tool_wear", "label": "Tool wear (minutes)", "default": 0},
]
LABELS = {field["name"]: field["label"] for field in FIELDS}


def read_test_results():
    #read existing experiment results without changing them
    tables = {}
    for mode in ("clean", "mcar", "mcar_imputed"):
        path = ROOT / "results/xgboost" / mode / "test.csv"
        tables[mode] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return tables


TEST_RESULTS = read_test_results()


def test_metrics(mode, seed, missing_pct):
    table = TEST_RESULTS[mode]
    if table.empty:
        return None
    rows = table[(table["dataset"] == "ai4i") & (table["seed"] == seed)]
    if mode != "clean":
        rows = rows[np.isclose(rows["missing_rate"], missing_pct / 100)]
    if len(rows) != 1:
        return None
    row = rows.iloc[0]
    metrics = {key: float(row[key]) for key in ("average_precision", "precision", "recall", "f1", "accuracy")}
    metrics["threshold"] = float(row["threshold"])
    metrics["best_round"] = int(row["best_round"]) if "best_round" in row else None
    return metrics


def load_models():
    #load every saved ai4i missingness model once at startup
    paths = [("clean", MODEL_PATH, "clean", 0, None)]
    pattern = re.compile(r"ai4i_(mcar|mcar_imputed)_(10|20|30|40)pct_seed(\d+)\.joblib")
    for path in sorted(MODEL_PATH.parent.glob("ai4i_mcar*.joblib")):
        match = pattern.fullmatch(path.name)
        if match:
            mode, rate, seed = match.groups()
            paths.append((path.stem.removeprefix("ai4i_"), path, mode, int(rate), int(seed)))
    loaded = {}
    for model_id, path, mode, rate, seed in paths:
        saved = load_artifact(path)
        processor, classifier = saved["preprocessor"], saved["model"]
        raw_columns = list(processor.feature_names_in_)
        names = list(processor.get_feature_names_out())
        if set(raw_columns) != set(LABELS) or classifier.n_features_in_ != len(names):
            raise RuntimeError(f"The input schema does not match {path.name}.")
        seed = int(classifier.get_params()["random_state"]) if seed is None else seed
        count = classifier.best_iteration + 1
        metrics = test_metrics(mode, seed, rate)
        threshold = float(saved["threshold"])
        #hide results if they appear to belong to a different saved run
        if metrics and (not np.isclose(metrics["threshold"], threshold, atol=1e-6, rtol=0)
                        or metrics["best_round"] not in (None, count)):
            metrics = None
        imputed = mode == "mcar_imputed"
        if imputed:
            note = ("Saved preprocessing fills missing inputs, standardizes sensor values, and one-hot encodes Product Type. "
                    "Tree thresholds use standardized values or 0/1 indicators. Filled inputs do not take NaN branches.")
        else:
            categories = processor.named_transformers_["categorical"].categories_[0]
            codes = ", ".join(f"{category} = {index}" for index, category in enumerate(categories))
            note = f"Sensor values keep their original units. Product Type codes: {codes}. Missing inputs remain NaN."
        loaded[model_id] = {
            "id": model_id, "filename": path.name, "mode": mode, "missing_pct": rate,
            "seed": seed, "threshold": threshold, "tree_count": count, "metrics": metrics,
            "clean_metrics": test_metrics("clean", seed, 0), "preprocessing_note": note,
            "label": "Clean training" if mode == "clean" else f"{rate}% missing · {'imputation' if imputed else 'native handling'}",
            "preprocessor": processor, "model": classifier, "raw_columns": raw_columns,
            "feature_names": names,
            "trees": [json.loads(tree) for tree in classifier.get_booster().get_dump(dump_format="json")[:count]],
        }
    return loaded


MODELS = load_models()
CLEAN = MODELS["clean"]
#keep the clean model available for the original checks
artifact = {key: CLEAN[key] for key in ("preprocessor", "model", "threshold")}
preprocessor, model = CLEAN["preprocessor"], CLEAN["model"]
THRESHOLD, TREE_COUNT, TREES = CLEAN["threshold"], CLEAN["tree_count"], CLEAN["trees"]
RAW_COLUMNS, FEATURE_NAMES = CLEAN["raw_columns"], CLEAN["feature_names"]


def model_info(bundle):
    #send labels and test results to the page without sending model objects
    return {key: value for key, value in bundle.items()
            if key not in ("preprocessor", "model", "trees", "raw_columns", "feature_names")}

#set up flask
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024


def feature_index(split, bundle=None):
    #match tree features to the transformed column order
    # f0 is the first column after preprocessing
    if split.startswith("f") and split[1:].isdigit():
        return int(split[1:])
    return (bundle or CLEAN)["feature_names"].index(split)


def readable_feature(index, bundle=None):
    # remove prefixes like numeric__ from the labels
    bundle = bundle or CLEAN
    raw = bundle["feature_names"][index].split("__", 1)[-1]
    if raw.startswith("Type_"):
        return f"Product Type = {raw.removeprefix('Type_')} (0/1)"
    if bundle["mode"] == "mcar_imputed" and raw != "Type":
        return raw.replace("_", " ") + " (standardized)"
    return LABELS[raw] + (" (encoded)" if raw == "Type" else "")


def walk_tree(tree, row, bundle=None):
    #follow the inputs from root to leaf
    node = tree
    path, decisions = [], []
    while True:
        path.append(node["nodeid"])
        # stop at the leaf and save its score
        # leaf score is not the failure probability
        if "leaf" in node:
            return {"nodes": path, "decisions": decisions,
                    "leaf_id": node["nodeid"], "leaf_score": node["leaf"]}
        index = feature_index(node["split"], bundle)
        # use float32 like xgboost for the split comparisons
        value = np.float32(row[index])
        cutoff = np.float32(node["split_condition"])
        missing = bool(np.isnan(value))
        # missing inputs follow the saved missing branch
        branch = "missing" if missing else ("yes" if value < cutoff else "no")
        child = node[branch]
        name = readable_feature(index, bundle)
        # explain each decision on the selected path
        if missing:
            explanation = f"{name} is missing (NaN): follow recorded missing child {child}; no comparison is made."
        else:
            operator = "<" if branch == "yes" else "≥"
            explanation = f"{name}: {float(value):.9g} {operator} {float(cutoff):.9g}; follow {branch} child {child}."
        decisions.append({"node": node["nodeid"], "child": child, "branch": branch,
                          "explanation": explanation})
        node = next(child_node for child_node in node["children"] if child_node["nodeid"] == child)


def display_tree(node, bundle=None):
    #send tree labels and branches to javascript
    result = {"id": node["nodeid"]}
    if "leaf" in node:
        result["leaf"] = node["leaf"]
    else:
        result.update(feature=readable_feature(feature_index(node["split"], bundle), bundle),
                      threshold=float(np.float32(node["split_condition"])),
                      yes=node["yes"], no=node["no"], missing=node["missing"],
                      children=[display_tree(child, bundle) for child in node["children"]])
    return result


def parse_payload(payload):
    #check inputs before making predictions
    if not isinstance(payload, dict) or not isinstance(payload.get("values"), dict):
        raise ValueError("Provide an object with all six input values.")
    values = payload["values"]
    model_id = payload.get("model", "clean")
    if not isinstance(model_id, str) or model_id not in MODELS:
        raise ValueError("Choose one of the available AI4I models.")
    bundle = MODELS[model_id]
    if set(values) != set(RAW_COLUMNS):
        raise ValueError("Provide exactly the six AI4I inputs.")
    missing = payload.get("missing", [])
    if not isinstance(missing, list) or any(not isinstance(key, str) or key not in RAW_COLUMNS for key in missing):
        raise ValueError("Missing inputs must be a list of AI4I column names.")
    tree_index = payload.get("tree", 0)
    # only allow trees up to the best iteration
    if type(tree_index) is not int or not 0 <= tree_index < bundle["tree_count"]:
        raise ValueError(f"Tree must be an integer from 0 to {bundle['tree_count'] - 1}.")
    clean = {}
    for key in RAW_COLUMNS:
        value = values[key]
        if key == "Type":
            if value not in ("H", "L", "M"):
                raise ValueError("Product Type must be H, L, or M.")
            clean[key] = value
        else:
            # convert form values to numbers and check them
            if isinstance(value, bool) or not isinstance(value, (int, float, str)) or value == "":
                raise ValueError(f"{LABELS[key]} must be a finite, nonnegative number.")
            try:
                number = float(value)
            except (ValueError, OverflowError):
                raise ValueError(f"{LABELS[key]} must be a number.") from None
            if not np.isfinite(number) or not 0 <= number <= float(np.finfo(np.float32).max):
                raise ValueError(f"{LABELS[key]} must be finite, nonnegative, and fit in float32.")
            if key in ("Air_temperature", "Process_temperature") and number == 0:
                raise ValueError("Temperatures in kelvin must be greater than zero.")
            clean[key] = number
    return clean, missing, tree_index, bundle


def predict_example(values, missing, tree_index, bundle=None):
    bundle = bundle or CLEAN
    processor, classifier = bundle["preprocessor"], bundle["model"]
    # keep the complete inputs for comparison
    complete = pd.DataFrame([values], columns=RAW_COLUMNS)
    complete["Type"] = complete["Type"].astype(object)
    # make a copy for the missing inputs
    # test data stays unchanged
    selected = complete.copy()
    for column in missing:
        selected.loc[0, column] = np.nan
    # apply saved preprocessing to both examples
    # do not fit again
    transformed = processor.transform(pd.concat([complete, selected], ignore_index=True))
    # column 1 is the failure probability
    probabilities = classifier.predict_proba(transformed, iteration_range=(0, bundle["tree_count"]))[:, 1]
    results = []
    # use the saved threshold for the predicted class
    # get the path through the selected tree
    for row, probability in zip(transformed, probabilities):
        results.append({"probability": float(probability), "failure": bool(probability >= bundle["threshold"]),
                        "path": walk_tree(bundle["trees"][tree_index], row, bundle)})
    #compare both models on the same raw inputs
    clean_probability = float(model.predict_proba(preprocessor.transform(selected), iteration_range=(0, TREE_COUNT))[0, 1])
    return {"baseline": results[0], "selected": results[1], "threshold": bundle["threshold"],
            "clean_reference": {"probability": clean_probability, "failure": clean_probability >= THRESHOLD,
                                "threshold": THRESHOLD, "seed": CLEAN["seed"]},
            "model": model_info(bundle), "tree_index": tree_index, "tree_count": bundle["tree_count"],
            "missing": missing, "tree": display_tree(bundle["trees"][tree_index], bundle)}


@app.get("/")
def index():
    #load page with inputs and tree options
    return render_template("index.html", fields=FIELDS, tree_count=TREE_COUNT,
                           models=[model_info(bundle) for bundle in MODELS.values()],
                           rates=sorted({bundle["missing_pct"] for bundle in MODELS.values()}),
                           seeds=sorted({bundle["seed"] for bundle in MODELS.values()}))


@app.post("/api/predict")
def predict():
    #get form data and send back predictions
    try:
        values, missing, tree_index, bundle = parse_payload(request.get_json(silent=True))
    except ValueError as error:
        return jsonify(error=str(error)), 400
    return jsonify(predict_example(values, missing, tree_index, bundle))


if __name__ == "__main__":
    #run locally without reloading the model
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
