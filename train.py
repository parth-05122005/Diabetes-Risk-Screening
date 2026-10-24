"""Train, evaluate and export the MedAI-nEXUS CDSS model.

    python train.py                               # CDC Diabetes Health Indicators (253,680 people)
    python train.py --dataset breast-cancer       # small FNA image-feature dataset
    python train.py --csv data.csv --target outcome   # any open tabular dataset

Writes:
    artifacts/model.joblib       trained, calibrated model + metadata
    reports/metrics.json         test metrics, model comparison, fairness audit, importance
    reports/*.png                ROC, calibration, global importance, fairness (poster-ready)
    samples/*.json               example patients and FHIR Bundles from the test set
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from medai_cdss.data import REGISTRY, load, load_csv  # noqa: E402
from medai_cdss.explain import global_importance  # noqa: E402
from medai_cdss.fhir import features_to_bundle  # noqa: E402
from medai_cdss.model import SEED, RiskPredictor, json_safe, train  # noqa: E402

ROOT = Path(__file__).parent


def plot_reports(res, predictor: RiskPredictor, out: Path) -> list[dict]:
    out.mkdir(exist_ok=True)
    y, p = res.y_test.values, res.test_prob
    b = res.bundle
    m = b["metrics"]

    fig, ax = plt.subplots(figsize=(5, 5))
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {m['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.scatter([1 - m["specificity"]], [m["sensitivity"]], zorder=3, color="crimson",
               label=f"Operating point (t={m['threshold']:.2f})")
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", title=f"ROC curve (test set, n={len(y):,})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac, "o-", lw=2, label=f"Model (Brier = {m['brier']:.3f})")
    lim = max(mean_pred.max(), frac.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="grey", lw=1, label="Perfect calibration")
    ax.set(xlabel="Predicted risk", ylabel="Observed frequency", title="Calibration (test set)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=160)
    plt.close(fig)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(res.X_test), size=min(300, len(res.X_test)), replace=False)
    imp = global_importance(predictor._proba, res.X_test.values[idx], b["background"])
    labels = np.array([f["label"] for f in b["features"]])
    order = np.argsort(imp)[-12:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(labels[order], imp[order] * 100, color="#3b6fd8")
    ax.set(xlabel="Mean |Shapley value| (risk percentage points)", title="Global feature importance")
    fig.tight_layout()
    fig.savefig(out / "global_importance.png", dpi=160)
    plt.close(fig)

    if b["fairness"]:
        fair = b["fairness"]
        names = [f"{r['attribute']}: {r['group']}" for r in fair]
        x = np.arange(len(fair))
        fig, ax = plt.subplots(figsize=(max(6, len(fair) * 0.8), 4.5))
        ax.bar(x - 0.2, [r["sensitivity"] for r in fair], 0.4, label="Sensitivity", color="#d6336c")
        ax.bar(x + 0.2, [r["specificity"] for r in fair], 0.4, label="Specificity", color="#1c7ed6")
        ax.set_xticks(x, names, rotation=30, ha="right")
        ax.set(ylim=(0, 1), title="Fairness audit: performance by subgroup (test set)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "fairness.png", dpi=160)
        plt.close(fig)

    ranked = sorted(zip(predictor.names, labels, imp), key=lambda t: -t[2])
    return [{"feature": n, "label": lab, "mean_abs_shap": float(v)} for n, lab, v in ranked]


def write_samples(res, out: Path) -> None:
    """Pick test patients at the 10th/50th/90th risk percentile of each true class."""
    out.mkdir(exist_ok=True)
    b = res.bundle
    examples = []
    for label, tag in ((1, b["positive_label"]), (0, b["negative_label"])):
        mask = (res.y_test == 1).values if label else (res.y_test == 0).values
        ids, probs = res.y_test.index[mask], res.test_prob[mask]
        order = np.argsort(probs)
        for q in (0.9, 0.5, 0.1):
            i = order[int(q * (len(order) - 1))]
            examples.append({"id": f"test-{ids[i]}", "label": tag, "features": res.X_test.loc[ids[i]].to_dict()})
    (out / "example_patients.json").write_text(json.dumps(examples, indent=2))
    for ex, name in ((examples[0], "positive"), (examples[-1], "negative")):
        bundle = features_to_bundle(ex["features"], b["features"], ex["id"], b["fhir_category"])
        (out / f"fhir_bundle_{name}.json").write_text(json.dumps(bundle, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="diabetes", choices=sorted(REGISTRY))
    ap.add_argument("--csv", help="Path to an open tabular dataset (CSV)")
    ap.add_argument("--target", help="Outcome column in the CSV")
    ap.add_argument("--positive", default=1, help="Value of the target that means 'positive'")
    args = ap.parse_args()

    if args.csv:
        if not args.target:
            ap.error("--target is required with --csv")
        pos = int(args.positive) if str(args.positive).lstrip("-").isdigit() else args.positive
        ds = load_csv(args.csv, args.target, pos)
    else:
        ds = load(args.dataset)
    print(f"Dataset: {ds.spec.name}  (n={len(ds.X):,}, features={ds.X.shape[1]}, "
          f"prevalence={ds.y.mean():.2%})")

    print("Model comparison on the validation set:")
    res = train(ds)

    (ROOT / "artifacts").mkdir(exist_ok=True)
    joblib.dump(res.bundle, ROOT / "artifacts" / "model.joblib")
    predictor = RiskPredictor(res.bundle)

    print("Computing global explanations, fairness audit and plots...")
    importance = plot_reports(res, predictor, ROOT / "reports")
    write_samples(res, ROOT / "samples")

    for r in res.bundle["fairness"]:
        print(f"  {r['attribute']:<7} {r['group']:<9} n={r['n']:>6,}  AUROC={r['auroc']:.3f}  "
              f"sens={r['sensitivity']:.3f}  spec={r['specificity']:.3f}")

    report = {
        "model": res.bundle["model_name"],
        "dataset": res.bundle["dataset"],
        "target_sensitivity": res.bundle["target_sensitivity"],
        "high_risk_cutoff": res.bundle["high_risk"],
        "test_metrics": res.bundle["metrics"],
        "model_comparison": res.bundle["model_comparison"],
        "fairness": res.bundle["fairness"],
        "global_importance": importance,
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(json_safe(report), indent=2))
    print("Saved artifacts/model.joblib, reports/, samples/")


if __name__ == "__main__":
    main()
