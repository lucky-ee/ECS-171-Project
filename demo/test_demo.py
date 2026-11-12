#run from demo/: python -m unittest -v test_demo.py
# use demo inputs to check predictions
import hashlib
import itertools
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import app as demo
from artifact_loader import load_artifact, portable_model_bytes


DEFAULTS = {field["name"]: field["default"] for field in demo.FIELDS}


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #load another copy of the model for comparison
        cls.direct = load_artifact(demo.MODEL_PATH)
        cls.client = demo.app.test_client()

    def post(self, **changes):
        #start with default values and change inputs for each test
        payload = {"values": DEFAULTS.copy(), "missing": [], "tree": 0}
        payload.update(changes)
        return self.client.post("/api/predict", json=payload)

    def test_page_and_local_assets(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertNotIn(b'data-missing=', page.data)
        self.assertNotIn(b'type="checkbox"', page.data)
        self.assertIn(f'{demo.TREE_COUNT} of {demo.TREE_COUNT}'.encode(), page.data)
        self.assertNotIn(b'https://', page.data)
        for asset in ("style.css", "app.js"):
            with self.client.get("/static/" + asset) as response:
                self.assertEqual(response.status_code, 200)

    def test_probabilities_match_separate_direct_inference(self):
        #compare api output with the saved model predictions
        before = hashlib.sha256(demo.MODEL_PATH.read_bytes()).digest()
        for category in ("H", "L", "M"):
            for missing in ([], ["Type"], ["Torque"], list(demo.RAW_COLUMNS)):
                values = dict(DEFAULTS, Type=category)
                response = self.post(values=values, missing=missing, tree=demo.TREE_COUNT - 1)
                self.assertEqual(response.status_code, 200)
                data = response.get_json()
                complete = pd.DataFrame([values], columns=demo.RAW_COLUMNS)
                complete["Type"] = complete["Type"].astype(object)
                selected = complete.copy()
                for column in missing:
                    selected.loc[0, column] = np.nan
                for label, raw in (("baseline", complete), ("selected", selected)):
                    transformed = self.direct["preprocessor"].transform(raw)
                    probability = float(self.direct["model"].predict_proba(transformed)[0, 1])
                    self.assertEqual(data[label]["probability"], probability)
                    self.assertEqual(data[label]["failure"], probability >= float(self.direct["threshold"]))
        self.assertEqual(before, hashlib.sha256(demo.MODEL_PATH.read_bytes()).digest())

    def test_every_tree_all_missing_masks_and_categories(self):
        #try all 64 missing input combinations for each type
        rows = []
        for category in ("H", "L", "M"):
            for mask in itertools.product((False, True), repeat=6):
                row = dict(DEFAULTS, Type=category)
                for column, missing in zip(demo.RAW_COLUMNS, mask):
                    if missing:
                        row[column] = np.nan
                rows.append(row)
        raw = pd.DataFrame(rows, columns=demo.RAW_COLUMNS)
        transformed = self.direct["preprocessor"].transform(raw)
        expected = self.direct["model"].apply(transformed)
        self.assertEqual(expected.shape, (192, demo.TREE_COUNT))
        for row_index, row in enumerate(transformed):
            for tree_index, tree in enumerate(demo.TREES):
                path = demo.walk_tree(tree, row)
                self.assertEqual(path["leaf_id"], expected[row_index, tree_index])
        for index, name in enumerate(demo.FEATURE_NAMES):
            raw_name = name.split("__", 1)[-1]
            np.testing.assert_array_equal(np.isnan(transformed[:, index]), raw[raw_name].isna())

    def test_float32_boundaries_and_random_examples(self):
        #check paths for random inputs and values near a split
        rng = np.random.default_rng(171)
        rows = np.column_stack([rng.uniform(290, 310, 100), rng.uniform(300, 320, 100),
                                rng.uniform(1100, 3000, 100), rng.uniform(0, 100, 100),
                                rng.uniform(0, 300, 100), rng.integers(0, 3, 100)])
        rows[rng.random(rows.shape) < .25] = np.nan
        leaves = self.direct["model"].apply(rows)
        for row_index, row in enumerate(rows):
            for tree_index, tree in enumerate(demo.TREES):
                self.assertEqual(demo.walk_tree(tree, row)["leaf_id"], leaves[row_index, tree_index])
        baseline = demo.preprocessor.transform(pd.DataFrame([DEFAULTS], columns=demo.RAW_COLUMNS))[0]
        samples, indices = [], []
        for tree_index, tree in enumerate(demo.TREES):
            if "leaf" in tree:
                continue
            cutoff = np.float32(tree["split_condition"])
            # include values that round to the same float32 cutoff
            candidates = [np.nextafter(cutoff, np.float32(-np.inf)), cutoff,
                          np.nextafter(cutoff, np.float32(np.inf)),
                          np.nextafter(float(cutoff), -np.inf), np.nextafter(float(cutoff), np.inf)]
            for value in candidates:
                row = baseline.copy()
                row[demo.feature_index(tree["split"])] = value
                samples.append(row)
                indices.append(tree_index)
        expected = self.direct["model"].apply(np.array(samples))
        for row_index, (row, tree_index) in enumerate(zip(samples, indices)):
            self.assertEqual(demo.walk_tree(demo.TREES[tree_index], row)["leaf_id"], expected[row_index, tree_index])

    def test_recorded_missing_child_not_assumed_right(self):
        #check missing inputs can follow the yes branch
        tree = {"nodeid": 0, "split": "f0", "split_condition": 2, "yes": 8, "no": 3,
                "missing": 8, "children": [{"nodeid": 3, "leaf": 1}, {"nodeid": 8, "leaf": -1}]}
        path = demo.walk_tree(tree, [np.nan])
        self.assertEqual(path["leaf_id"], 8)
        self.assertEqual(path["decisions"][0]["branch"], "missing")

    def test_validation_and_tree_bounds(self):
        for tree in (-1, demo.TREE_COUNT, 1.5, True, "0", None):
            self.assertEqual(self.post(tree=tree).status_code, 400)
        for missing in ("Torque", ["unknown"], [None], {}):
            self.assertEqual(self.post(missing=missing).status_code, 400)
        for key, value in (("Type", "X"), ("Type", None), ("Torque", -1), ("Torque", ""),
                           ("Torque", "NaN"), ("Torque", "Infinity"), ("Torque", "1e100"),
                           ("Torque", False), ("Torque", []), ("Air_temperature", 0)):
            values = dict(DEFAULTS, **{key: value})
            self.assertEqual(self.post(values=values).status_code, 400)
        self.assertEqual(self.post(values={}).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", data="{", content_type="application/json").status_code, 400)
        self.assertEqual(self.post(tree=demo.TREE_COUNT - 1).status_code, 200)

    def test_inference_never_fits_or_reloads(self):
        with patch.object(demo.preprocessor, "fit", side_effect=AssertionError("fit called")), \
             patch.object(demo.preprocessor, "fit_transform", side_effect=AssertionError("fit_transform called")), \
             patch.object(demo, "load_artifact", side_effect=AssertionError("reload called")):
            self.assertEqual(self.post(missing=["Torque", "Type"]).status_code, 200)

    def test_all_saved_models_match_direct_predictions_and_paths(self):
        #check every model with complete and missing demo inputs
        self.assertEqual(len(demo.MODELS), 25)
        for model_id, bundle in demo.MODELS.items():
            direct = load_artifact(demo.MODEL_PATH.parent / bundle["filename"])
            rows = []
            for category in ("H", "L", "M"):
                for missing in ([], ["Type"], ["Torque"], list(demo.RAW_COLUMNS)):
                    raw = dict(DEFAULTS, Type=category)
                    for column in missing:
                        raw[column] = np.nan
                    rows.append(raw)
            raw = pd.DataFrame(rows, columns=bundle["raw_columns"])
            transformed = direct["preprocessor"].transform(raw)
            leaves = direct["model"].apply(transformed)
            self.assertEqual(leaves.shape[1], bundle["tree_count"])
            for row_index, row in enumerate(transformed):
                for tree_index, tree in enumerate(bundle["trees"]):
                    self.assertEqual(demo.walk_tree(tree, row, bundle)["leaf_id"], leaves[row_index, tree_index])
            for missing in ([], ["Type", "Torque"], list(demo.RAW_COLUMNS)):
                response = self.post(model=model_id, missing=missing, tree=bundle["tree_count"] - 1)
                self.assertEqual(response.status_code, 200)
                data = response.get_json()
                example = pd.DataFrame([DEFAULTS], columns=bundle["raw_columns"])
                example["Type"] = example["Type"].astype(object)
                for column in missing:
                    example.loc[0, column] = np.nan
                expected = float(direct["model"].predict_proba(direct["preprocessor"].transform(example))[0, 1])
                self.assertEqual(data["selected"]["probability"], expected)
                self.assertEqual(data["selected"]["failure"], expected >= float(direct["threshold"]))
                clean_expected = float(self.direct["model"].predict_proba(self.direct["preprocessor"].transform(example))[0, 1])
                self.assertEqual(data["clean_reference"]["probability"], clean_expected)
                self.assertEqual(data["threshold"], float(direct["threshold"]))
            self.assertEqual(self.post(model=model_id, tree=bundle["tree_count"]).status_code, 400)

    def test_test_metrics_and_feature_labels(self):
        #match saved scores to the selected model and training seed
        for bundle in demo.MODELS.values():
            metrics = bundle["metrics"]
            self.assertIsNotNone(metrics)
            self.assertAlmostEqual(metrics["threshold"], bundle["threshold"], places=6)
            table = demo.TEST_RESULTS[bundle["mode"]]
            rows = table[(table["dataset"] == "ai4i") & (table["seed"] == bundle["seed"])]
            if bundle["mode"] != "clean":
                rows = rows[np.isclose(rows["missing_rate"], bundle["missing_pct"] / 100)]
            self.assertEqual(metrics["average_precision"], float(rows.iloc[0]["average_precision"]))
            if bundle["mode"] == "mcar_imputed":
                self.assertIn("standardized", demo.readable_feature(0, bundle))
                self.assertIn("Product Type = H (0/1)", demo.readable_feature(5, bundle))
                self.assertEqual(len(bundle["feature_names"]), 8)
            else:
                self.assertEqual(len(bundle["feature_names"]), 6)
        for invalid in ("../models/adult_clean.joblib", "unknown", None, []):
            self.assertEqual(self.post(model=invalid).status_code, 400)

    def test_no_model_load_or_fit_when_switching(self):
        #switching models should use the copies already loaded at startup
        with patch.object(demo, "load_artifact", side_effect=AssertionError("reload called")):
            for model_id, bundle in demo.MODELS.items():
                with patch.object(bundle["preprocessor"], "fit", side_effect=AssertionError("fit called")), \
                     patch.object(bundle["preprocessor"], "fit_transform", side_effect=AssertionError("fit_transform called")):
                    self.assertEqual(self.post(model=model_id).status_code, 200)

    def test_portable_model_is_verbatim_and_equivalent(self):
        #check windows loading keeps the same model and predictions
        import joblib
        from xgboost import Booster, DMatrix
        captured = []

        def capture(booster, state):
            captured.append(bytearray(state["handle"]))
            booster.handle = None

        with patch.object(Booster, "__setstate__", capture):
            joblib.load(demo.MODEL_PATH)
        portable = portable_model_bytes(captured[0])
        self.assertIn(portable, captured[0])
        direct = Booster(model_file=portable)
        self.assertEqual(direct.get_dump(dump_format="json"), demo.model.get_booster().get_dump(dump_format="json"))
        raw = pd.DataFrame([DEFAULTS], columns=demo.RAW_COLUMNS)
        transformed = self.direct["preprocessor"].transform(raw)
        probability = direct.predict(DMatrix(transformed), iteration_range=(0, demo.TREE_COUNT))[0]
        self.assertEqual(float(probability), self.post().get_json()["selected"]["probability"])
        with self.assertRaises(ValueError):
            portable_model_bytes(b"invalid snapshot")


if __name__ == "__main__":
    unittest.main()
