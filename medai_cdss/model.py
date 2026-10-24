"""Training, model selection, calibration, fairness audit and the bedside predictor."""

from __future__ import annotations

import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Dataset, feature_key
from .explain import shapley_values

SEED = 42
HIGH_RISK_SPECIFICITY = 0.95  # "High" band = top of the risk scale, where >= 95% of negatives fall below


def json_safe(obj):
    """Make metrics JSON-safe (NaN -> null) for browsers and report files."""
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def candidate_models(n_rows: int) -> dict:
    large = n_rows > 20_000
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, random_state=SEED)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200 if large else 400,
            min_samples_leaf=25 if large else 2,
            max_samples=0.5 if large else None,
            random_state=SEED, n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=500, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED,
        ),
    }


def evaluate(y_true, prob, threshold) -> dict:
    y_true = np.asarray(y_true)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, prob)) if both else float("nan"),
        "auprc": float(average_precision_score(y_true, prob)) if both else float("nan"),
        "brier": float(brier_score_loss(y_true, prob)),
        "accuracy": float((tp + tn) / len(y_true)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "ppv": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "npv": float(tn / (tn + fn)) if tn + fn else float("nan"),
        "flag_rate": float(pred.mean()),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": float(threshold),
    }


def pick_threshold(y_true, prob, target_sensitivity) -> float:
    """Highest threshold that still reaches the target sensitivity (maximises specificity)."""
    pos = np.sort(prob[np.asarray(y_true) == 1])
    k = int(np.floor((1 - target_sensitivity) * len(pos)))
    return float(np.clip(pos[k], 0.001, 0.999))


def pick_high_cutoff(y_true, prob, specificity=HIGH_RISK_SPECIFICITY) -> float:
    neg = prob[np.asarray(y_true) == 0]
    return float(np.quantile(neg, specificity))


def subgroup_metrics(X: pd.DataFrame, y, prob, fairness: dict, threshold: float) -> list[dict]:
    """Performance per demographic subgroup (sex, age, income ...) on the test set."""
    rows = []
    y = np.asarray(y)
    for col, cfg in fairness.items():
        for label, (lo, hi) in cfg["groups"].items():
            mask = ((X[col] >= lo) & (X[col] <= hi)).values
            if mask.sum() < 30:
                continue
            m = evaluate(y[mask], prob[mask], threshold)
            rows.append({"attribute": cfg["name"], "group": label,
                         **{k: m[k] for k in ("n", "prevalence", "auroc", "sensitivity", "specificity",
                                              "ppv", "flag_rate", "brier")}})
    return rows


@dataclass
class TrainingResult:
    bundle: dict
    X_test: pd.DataFrame
    y_test: pd.Series
    test_prob: np.ndarray


def train(ds: Dataset, log=print) -> TrainingResult:
    spec = ds.spec
    # 70 / 15 / 15: train models, validate (select model + set thresholds), test (report only).
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        ds.X, ds.y, test_size=0.15, stratify=ds.y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=SEED)
    log(f"  Split: train={len(X_train):,}  validation={len(X_val):,}  test={len(X_test):,}")

    rows = []
    for name, est in candidate_models(len(ds.X)).items():
        t0 = time.time()
        est.fit(X_train, y_train)
        prob = est.predict_proba(X_val)[:, 1]
        m = evaluate(y_val, prob, pick_threshold(y_val, prob, spec.target_sensitivity))
        rows.append({"model": name, "auroc": m["auroc"], "auprc": m["auprc"], "brier": m["brier"],
                     "specificity_at_target": m["specificity"], "fit_seconds": round(time.time() - t0, 1)})
        log(f"  {name:<20} val AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
            f"Brier={m['brier']:.4f}  ({time.time() - t0:.0f}s)")
    comparison = sorted(rows, key=lambda r: (-r["auroc"], r["brier"]))
    best_name = comparison[0]["model"]
    log(f"  Selected: {best_name}")

    # Calibrate so the output is a trustworthy probability, not just a ranking score.
    method = "isotonic" if len(X_train) > 5_000 else "sigmoid"
    model = CalibratedClassifierCV(clone(candidate_models(len(ds.X))[best_name]), method=method, cv=3)
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = pick_threshold(y_val, val_prob, spec.target_sensitivity)
    high = max(pick_high_cutoff(y_val, val_prob), threshold + 0.01)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_prob, threshold)
    log(f"  Test: AUROC={test_metrics['auroc']:.4f}  sens={test_metrics['sensitivity']:.3f}  "
        f"spec={test_metrics['specificity']:.3f}  threshold={threshold:.3f}  high>={high:.3f}")

    rng = np.random.default_rng(SEED)
    background = X_train.values[rng.choice(len(X_train), size=min(100, len(X_train)), replace=False)]
    features = []
    for c in ds.X.columns:
        meta = spec.features.get(c) or {"label": c, "group": "Features", "kind": "numeric"}
        features.append({
            **meta, "name": c, "key": feature_key(c),
            "median": float(X_train[c].median()),
            "p05": float(X_train[c].quantile(0.05)), "p95": float(X_train[c].quantile(0.95)),
            "min": float(X_train[c].min()), "max": float(X_train[c].max()),
        })

    bundle = {
        "model": model,
        "model_name": f"{best_name} ({method}-calibrated)",
        "feature_names": list(ds.X.columns),
        "features": features,
        "threshold": threshold,
        "high_risk": high,
        "target_sensitivity": spec.target_sensitivity,
        "background": background,
        "metrics": test_metrics,
        "model_comparison": comparison,
        "fairness": subgroup_metrics(X_test, y_test, test_prob, spec.fairness, threshold),
        "dataset": {"key": spec.key, "name": spec.name, "source": spec.source, "citation": spec.citation,
                    "license": spec.license, "n": int(len(ds.X)), "n_features": int(ds.X.shape[1]),
                    "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
                    "prevalence": float(ds.y.mean())},
        "positive_label": spec.positive_label,
        "negative_label": spec.negative_label,
        "outcome_text": spec.outcome_text,
        "recommendations": spec.recommendations,
        "fhir_category": spec.fhir_category,
    }
    return TrainingResult(bundle, X_test, y_test, test_prob)


class RiskPredictor:
    """Loads a trained bundle and produces explained, banded risk predictions."""

    def __init__(self, bundle: dict):
        self.b = bundle
        self.names = bundle["feature_names"]

    @classmethod
    def load(cls, path) -> "RiskPredictor":
        return cls(joblib.load(path))

    def _proba(self, X: np.ndarray) -> np.ndarray:
        return self.b["model"].predict_proba(pd.DataFrame(X, columns=self.names))[:, 1]

    def band(self, p: float) -> str:
        if p < self.b["threshold"]:
            return "Low"
        return "High" if p >= self.b["high_risk"] else "Moderate"

    @staticmethod
    def _display(f: dict, v: float) -> str:
        opts = f.get("options")
        if opts:
            return str(opts.get(int(v), opts.get(str(int(v)), v)))
        return f"{v:g}" + (f" {f['unit']}" if f.get("unit") else "")

    def predict(self, values: dict, n_permutations: int = 128) -> dict:
        lookup = {str(k).lower(): v for k, v in values.items()}
        x, imputed, warnings = [], [], []
        for f in self.b["features"]:
            v = lookup.get(f["name"].lower(), lookup.get(f["key"]))
            if v is None or v == "":
                x.append(f["median"])
                imputed.append(f["name"])
                continue
            v = float(v)
            if not np.isfinite(v):
                raise ValueError(f"{f['label']} must be a finite number")
            opts = f.get("options")
            if opts and int(v) not in {int(k) for k in opts}:
                raise ValueError(f"{f['label']} must be one of {sorted(int(k) for k in opts)}")
            if not opts and (v < f["min"] or v > f["max"]):
                warnings.append(f"{f['label']} = {v:g} is outside the training range "
                                f"[{f['min']:g}, {f['max']:g}]; prediction may be unreliable.")
            x.append(v)
        x = np.array(x)

        p = float(self._proba(x[None, :])[0])
        phi, baseline = shapley_values(self._proba, x, self.b["background"], n_permutations)
        contributions = sorted(
            ({"name": f["name"], "label": f["label"], "value": float(v), "display": self._display(f, v),
              "impact": float(s), "imputed": f["name"] in imputed}
             for f, v, s in zip(self.b["features"], x, phi)),
            key=lambda c: -abs(c["impact"]),
        )
        band = self.band(p)
        if len(imputed) > len(self.names) // 3:
            warnings.append(f"{len(imputed)} of {len(self.names)} inputs were missing and imputed "
                            "with training medians; interpret with caution.")
        return {
            "probability": p,
            "band": band,
            "flagged": p >= self.b["threshold"],
            "recommendation": self.b["recommendations"][band],
            "baseline": baseline,
            "contributions": contributions,
            "imputed": imputed,
            "warnings": warnings,
            "threshold": self.b["threshold"],
            "high_risk": self.b["high_risk"],
            "positive_label": self.b["positive_label"],
            "outcome_text": self.b["outcome_text"],
            "model_name": self.b["model_name"],
        }
