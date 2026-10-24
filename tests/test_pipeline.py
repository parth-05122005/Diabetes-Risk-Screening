"""End-to-end tests:  python -m unittest discover tests

Trains on a 40,000-person stratified sample of the diabetes dataset to keep the run short.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app  # noqa: E402
from medai_cdss.data import Dataset, load  # noqa: E402
from medai_cdss.explain import shapley_values  # noqa: E402
from medai_cdss.fhir import FhirError, features_to_bundle, observations_to_features  # noqa: E402
from medai_cdss.model import RiskPredictor, pick_threshold, train  # noqa: E402


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full = load("diabetes")
        X, _, y, _ = train_test_split(full.X, full.y, train_size=40_000, stratify=full.y, random_state=0)
        cls.ds = Dataset(X, y, full.spec)
        cls.res = train(cls.ds, log=lambda *_: None)
        cls.pred = RiskPredictor(cls.res.bundle)
        cls.features = cls.res.bundle["features"]
        cls.client = create_app(cls.pred, examples=[]).test_client()

    def test_held_out_performance(self):
        m = self.res.bundle["metrics"]
        self.assertGreater(m["auroc"], 0.78)
        self.assertGreater(m["sensitivity"], 0.72)
        self.assertLess(m["brier"], 0.11)

    def test_threshold_hits_target_sensitivity(self):
        y = np.array([1] * 20 + [0] * 20)
        p = np.concatenate([np.linspace(0.3, 0.99, 20), np.linspace(0.01, 0.5, 20)])
        t = pick_threshold(y, p, 0.95)
        self.assertGreaterEqual((p[y == 1] >= t).mean(), 0.95)

    def test_bands_are_ordered(self):
        b = self.res.bundle
        self.assertLess(b["threshold"], b["high_risk"])
        self.assertEqual(self.pred.band(b["threshold"] / 2), "Low")
        self.assertEqual(self.pred.band(min(0.999, b["high_risk"] + 0.01)), "High")

    def test_shapley_is_additive(self):
        x = self.res.X_test.values[0]
        phi, base = shapley_values(self.pred._proba, x, self.res.bundle["background"], 256)
        p = self.pred._proba(x[None, :])[0]
        # Exact additivity holds against the sampled references; allow Monte-Carlo noise.
        self.assertAlmostEqual(base + phi.sum(), p, delta=0.05)

    def test_high_risk_profile_is_flagged(self):
        r = self.pred.predict({"HighBP": 1, "HighChol": 1, "BMI": 40, "GenHlth": 5, "Age": 11,
                               "DiffWalk": 1, "HeartDiseaseorAttack": 1, "PhysActivity": 0})
        self.assertTrue(r["flagged"])
        top = {c["name"] for c in r["contributions"][:6]}
        self.assertTrue({"GenHlth", "BMI", "HighBP"} & top)
        self.assertEqual(next(c for c in r["contributions"] if c["name"] == "GenHlth")["display"], "Poor")

    def test_input_validation(self):
        r = self.pred.predict({"BMI": 200})
        self.assertEqual(len(r["imputed"]), 20)
        self.assertTrue(any("outside the training range" in w for w in r["warnings"]))
        with self.assertRaises(ValueError):
            self.pred.predict({"BMI": float("nan")})
        with self.assertRaises(ValueError):
            self.pred.predict({"Age": 42})  # must be an age-band code 1-13

    def test_fairness_audit_present(self):
        groups = {(r["attribute"], r["group"]) for r in self.res.bundle["fairness"]}
        self.assertIn(("Sex", "Female"), groups)
        self.assertIn(("Age", "65+"), groups)

    def test_fhir_round_trip(self):
        feats = self.res.X_test.iloc[3].to_dict()
        bundle = features_to_bundle(feats, self.features, "p1")
        values, ref = observations_to_features(bundle, self.features)
        self.assertEqual(ref, "Patient/p1")
        self.assertEqual(values, feats)
        # BMI is matched by its LOINC code alone.
        loinc_only = {"resourceType": "Bundle", "entry": [{"resource": {
            "resourceType": "Observation", "code": {"coding": [{"system": "http://loinc.org", "code": "39156-5"}]},
            "valueQuantity": {"value": 33.0}}}]}
        self.assertEqual(observations_to_features(loinc_only, self.features)[0], {"BMI": 33.0})
        with self.assertRaises(FhirError):
            observations_to_features({"resourceType": "Patient"}, self.features)

    def test_api(self):
        feats = self.res.X_test.iloc[0].to_dict()
        r = self.client.post("/api/predict", json={"features": feats})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["fhir"]["resourceType"], "RiskAssessment")
        self.assertEqual(len(body["contributions"]), 21)

        r = self.client.post("/api/fhir", json=features_to_bundle(feats, self.features, "abc"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["fhir"]["subject"]["reference"], "Patient/abc")

        self.assertEqual(self.client.post("/api/predict", json={"x": 1}).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", json={"features": {"Age": 99}}).status_code, 400)
        self.assertEqual(self.client.post("/api/fhir", json={"resourceType": "Bundle"}).status_code, 400)
        self.assertEqual(self.client.get("/api/meta").status_code, 200)


if __name__ == "__main__":
    unittest.main()
