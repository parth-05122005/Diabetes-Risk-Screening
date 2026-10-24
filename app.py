"""MedAI-nEXUS CDSS web app and REST API.

    python app.py            -> http://127.0.0.1:5000

API
    GET  /api/meta           model card, feature metadata, fairness audit, example patients
    POST /api/predict        {"features": {"BMI": 31, "HighBP": 1, ...}, "patient": "Patient/123"}
    POST /api/fhir           a FHIR R4 Bundle of Observations -> prediction + RiskAssessment
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from medai_cdss.fhir import FhirError, observations_to_features, risk_assessment
from medai_cdss.model import RiskPredictor, json_safe

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
EXAMPLES_PATH = ROOT / "samples" / "example_patients.json"


def create_app(predictor: RiskPredictor | None = None, examples: list | None = None) -> Flask:
    app = Flask(__name__)
    if predictor is None:
        if not MODEL_PATH.exists():
            raise SystemExit("No trained model found. Run `python train.py` first.")
        predictor = RiskPredictor.load(MODEL_PATH)
    if examples is None:
        examples = json.loads(EXAMPLES_PATH.read_text()) if EXAMPLES_PATH.exists() else []
    b = predictor.b

    def respond(values: dict, patient_ref: str | None):
        result = predictor.predict(values)
        result["fhir"] = risk_assessment(result, patient_ref, b["model_name"])
        return jsonify(json_safe(result))

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/meta")
    def meta():
        return jsonify(json_safe({
            "model_name": b["model_name"],
            "features": b["features"],
            "threshold": b["threshold"],
            "high_risk": b["high_risk"],
            "target_sensitivity": b["target_sensitivity"],
            "metrics": b["metrics"],
            "model_comparison": b["model_comparison"],
            "fairness": b["fairness"],
            "dataset": b["dataset"],
            "positive_label": b["positive_label"],
            "outcome_text": b["outcome_text"],
            "examples": examples,
        }))

    @app.post("/api/predict")
    def predict():
        body = request.get_json(silent=True) or {}
        values = body.get("features")
        if not isinstance(values, dict):
            return jsonify({"error": "Body must be {\"features\": {name: value}}"}), 400
        try:
            return respond(values, body.get("patient"))
        except (TypeError, ValueError) as e:
            return jsonify({"error": str(e)}), 400

    @app.post("/api/fhir")
    def fhir():
        try:
            values, patient_ref = observations_to_features(request.get_json(silent=True), b["features"])
            return respond(values, patient_ref)
        except (FhirError, TypeError, ValueError) as e:
            return jsonify({"error": str(e)}), 400

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
