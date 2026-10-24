"""HL7 FHIR R4 interoperability.

Input:  a FHIR ``Bundle`` of ``Observation`` resources, one per feature.
Output: a FHIR ``RiskAssessment`` resource carrying the prediction and the
        top explanatory factors, ready to be POSTed to any FHIR server.

Features with a LOINC code (e.g. BMI = 39156-5) are matched on it. Every
feature also has a local code (its key, e.g. ``highbp``) in the project
CodeSystem; matching is case-insensitive and the display text is accepted as
a fallback. Values may be ``valueQuantity``, ``valueInteger`` or
``valueBoolean``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from .data import feature_key

CODE_SYSTEM = "https://medai-nexus.example.org/fhir/CodeSystem/cdss-features"
LOINC = "http://loinc.org"


class FhirError(ValueError):
    pass


def _value(res: dict):
    if "valueQuantity" in res:
        return res["valueQuantity"].get("value")
    if "valueInteger" in res:
        return res["valueInteger"]
    if "valueBoolean" in res:
        return int(bool(res["valueBoolean"]))
    return None


def observations_to_features(bundle: dict, features: list[dict]) -> tuple[dict, str | None]:
    """Extract {feature name: value} and the patient reference from a Bundle."""
    if not isinstance(bundle, dict) or bundle.get("resourceType") != "Bundle":
        raise FhirError("Expected a FHIR Bundle resource")

    lookup = {}
    for f in features:
        lookup[feature_key(f["name"])] = f["name"]
        lookup[feature_key(f.get("label") or f["name"])] = f["name"]
        if f.get("loinc"):
            lookup[f["loinc"]] = f["name"]

    values: dict[str, float] = {}
    patient_ref = None
    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        if res.get("resourceType") == "Patient" and res.get("id"):
            patient_ref = patient_ref or f"Patient/{res['id']}"
        if res.get("resourceType") != "Observation":
            continue
        patient_ref = patient_ref or res.get("subject", {}).get("reference")
        val = _value(res)
        if val is None:
            continue
        code = res.get("code", {})
        codings = code.get("coding", [])
        candidates = [c.get("code", "") for c in codings]
        candidates += [c.get("display", "") for c in codings] + [code.get("text", "")]
        for cand in candidates:
            name = lookup.get(cand) or lookup.get(feature_key(cand)) if cand else None
            if name:
                values[name] = float(val)
                break
    if not values:
        raise FhirError("No Observation in the Bundle matched a model feature")
    return values, patient_ref


def features_to_bundle(values: dict, features: list[dict], patient_id: str = "example",
                       category: str = "survey") -> dict:
    """Build a Bundle of Observations (used for samples and tests)."""
    meta = {f["name"]: f for f in features}
    entries = [{"resource": {"resourceType": "Patient", "id": patient_id}}]
    for name, value in values.items():
        f = meta[name]
        coding = [{"system": CODE_SYSTEM, "code": feature_key(name), "display": f.get("label", name)}]
        if f.get("loinc"):
            coding.insert(0, {"system": LOINC, "code": f["loinc"], "display": f.get("label", name)})
        obs = {
            "resourceType": "Observation",
            "status": "final",
            "category": [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": category}]}],
            "code": {"coding": coding, "text": f.get("label", name)},
            "subject": {"reference": f"Patient/{patient_id}"},
        }
        if f.get("kind", "numeric") == "numeric":
            q = {"value": round(float(value), 6)}
            if f.get("unit"):
                q["unit"] = f["unit"]
            obs["valueQuantity"] = q
        else:
            obs["valueInteger"] = int(value)
        entries.append({"resource": obs})
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def risk_assessment(result: dict, patient_ref: str | None, model_name: str) -> dict:
    """Wrap a prediction in a FHIR R4 RiskAssessment."""
    top = result["contributions"][:5]
    basis = "; ".join(
        f"{c['label']} = {c['display']} {'raises' if c['impact'] > 0 else 'lowers'} risk by "
        f"{abs(c['impact']) * 100:.1f} pts"
        for c in top
    )
    return {
        "resourceType": "RiskAssessment",
        "id": str(uuid.uuid4()),
        "status": "preliminary",
        "subject": {"reference": patient_ref or "Patient/unknown"},
        "occurrenceDateTime": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": {"text": model_name},
        "prediction": [{
            "outcome": {"text": result["positive_label"]},
            "probabilityDecimal": round(result["probability"], 4),
            "qualitativeRisk": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                    "code": {"Low": "low", "Moderate": "moderate", "High": "high"}[result["band"]],
                    "display": f"{result['band']} risk",
                }]
            },
            "rationale": basis,
        }],
        "mitigation": result["recommendation"],
        "note": [{"text": "Research prototype - decision support only, not a diagnosis."}],
    }
