"""Dataset registry, loading and feature metadata.

Two open datasets are registered:

* ``diabetes`` (default): CDC Diabetes Health Indicators, UCI ML Repository id 891.
  253,680 adults from the CDC Behavioral Risk Factor Surveillance System (BRFSS
  2015), 21 features. BRFSS is US-government public-domain data (UCI defers to the
  source for licensing). Downloaded once to ``data/raw/``.
* ``breast-cancer``: Breast Cancer Wisconsin (Diagnostic), UCI id 17. 569 patients,
  30 features from digitised fine-needle-aspirate images; ships with scikit-learn.

Any other open tabular dataset can be used through ``load_csv``.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series  # 1 = positive class (the risk being predicted)
    spec: "DatasetSpec"


@dataclass
class DatasetSpec:
    key: str
    name: str
    source: str
    citation: str
    license: str
    positive_label: str
    negative_label: str
    outcome_text: str
    target_sensitivity: float
    recommendations: dict
    fhir_category: str
    features: dict  # column -> metadata (see feat())
    loader: Callable[[], tuple[pd.DataFrame, pd.Series]]
    fairness: dict = field(default_factory=dict)  # column -> {"name": str, "groups": {label: (lo, hi)}}


def feature_key(name: str) -> str:
    """Stable, URL/FHIR-safe code for a feature, e.g. 'mean radius' -> 'mean-radius'."""
    return name.strip().lower().replace(" ", "-").replace("_", "-")


def feat(label, group, kind="numeric", help="", options=None, unit=None, loinc=None) -> dict:
    """kind: 'numeric' (free value), 'binary' (0/1), or 'ordinal' (coded options)."""
    if kind == "binary" and options is None:
        options = {0: "No", 1: "Yes"}
    return {"label": label, "group": group, "kind": kind, "help": help,
            "options": options, "unit": unit, "loinc": loinc}


# --------------------------------------------------------------------------- diabetes

DIABETES_URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
DIABETES_FILE = RAW_DIR / "cdc_diabetes_health_indicators.csv"

AGE_BANDS = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54",
             8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"}
GEN_HEALTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
EDUCATION = {1: "Never attended / kindergarten", 2: "Grades 1-8", 3: "Grades 9-11",
             4: "Grade 12 / GED", 5: "College 1-3 years", 6: "College 4+ years"}
INCOME = {1: "< $10k", 2: "$10-15k", 3: "$15-20k", 4: "$20-25k", 5: "$25-35k",
          6: "$35-50k", 7: "$50-75k", 8: ">= $75k"}

DIABETES_FEATURES = {
    "HighBP": feat("High blood pressure", "Clinical history", "binary",
                   "Ever told by a health professional that blood pressure is high"),
    "HighChol": feat("High cholesterol", "Clinical history", "binary",
                     "Ever told by a health professional that blood cholesterol is high"),
    "CholCheck": feat("Cholesterol check in last 5 years", "Healthcare access", "binary"),
    "BMI": feat("Body-mass index", "Clinical history", "numeric", "Weight (kg) / height (m)^2",
                unit="kg/m2", loinc="39156-5"),
    "Smoker": feat("Smoked >= 100 cigarettes in lifetime", "Lifestyle", "binary"),
    "Stroke": feat("History of stroke", "Clinical history", "binary"),
    "HeartDiseaseorAttack": feat("Coronary heart disease or MI", "Clinical history", "binary"),
    "PhysActivity": feat("Physical activity in past 30 days", "Lifestyle", "binary",
                         "Any exercise outside regular job"),
    "Fruits": feat("Eats fruit daily", "Lifestyle", "binary"),
    "Veggies": feat("Eats vegetables daily", "Lifestyle", "binary"),
    "HvyAlcoholConsump": feat("Heavy alcohol use", "Lifestyle", "binary",
                              "Men > 14 drinks/week, women > 7 drinks/week"),
    "AnyHealthcare": feat("Has any health coverage", "Healthcare access", "binary"),
    "NoDocbcCost": feat("Skipped doctor due to cost (past year)", "Healthcare access", "binary"),
    "GenHlth": feat("Self-rated general health", "Self-reported health", "ordinal", options=GEN_HEALTH),
    "MentHlth": feat("Poor mental-health days (past 30)", "Self-reported health", "numeric", unit="days"),
    "PhysHlth": feat("Poor physical-health days (past 30)", "Self-reported health", "numeric", unit="days"),
    "DiffWalk": feat("Serious difficulty walking or climbing stairs", "Self-reported health", "binary"),
    "Sex": feat("Sex", "Demographics", "ordinal", options={0: "Female", 1: "Male"}),
    "Age": feat("Age group", "Demographics", "ordinal", options=AGE_BANDS),
    "Education": feat("Education", "Demographics", "ordinal", options=EDUCATION),
    "Income": feat("Annual household income", "Demographics", "ordinal", options=INCOME),
}


def download_diabetes(force: bool = False) -> Path:
    if DIABETES_FILE.exists() and not force:
        return DIABETES_FILE
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CDC Diabetes Health Indicators from {DIABETES_URL} ...")
    urllib.request.urlretrieve(DIABETES_URL, DIABETES_FILE)
    return DIABETES_FILE


def _load_diabetes():
    df = pd.read_csv(download_diabetes())
    y = df.pop("Diabetes_binary").astype(int).rename("diabetes")
    X = df[list(DIABETES_FEATURES)].astype(float)
    return X, y


DIABETES = DatasetSpec(
    key="diabetes",
    name="CDC Diabetes Health Indicators (BRFSS 2015)",
    source="https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators",
    citation="Centers for Disease Control and Prevention. CDC Diabetes Health Indicators. "
             "UCI Machine Learning Repository, 2023. https://doi.org/10.24432/C53919",
    license="public-domain CDC BRFSS data",
    positive_label="Diabetes or prediabetes",
    negative_label="No diabetes",
    outcome_text="diabetes or prediabetes",
    # Screening use: catch most cases while keeping the referral load manageable.
    target_sensitivity=0.80,
    recommendations={
        "Low": "Low predicted risk. Reinforce healthy lifestyle; routine screening per "
               "ADA/ICMR guidance (e.g. every 3 years from age 35).",
        "Moderate": "Elevated risk. Order HbA1c or fasting plasma glucose to confirm, and "
                    "offer lifestyle counselling (weight, activity, diet).",
        "High": "High risk. Prioritise confirmatory HbA1c/FPG testing, review blood pressure "
                "and lipids, and consider referral to a diabetes-prevention programme.",
    },
    fhir_category="survey",
    features=DIABETES_FEATURES,
    loader=_load_diabetes,
    fairness={
        "Sex": {"name": "Sex", "groups": {"Female": (0, 0), "Male": (1, 1)}},
        "Age": {"name": "Age", "groups": {"18-44": (1, 5), "45-64": (6, 9), "65+": (10, 13)}},
        "Income": {"name": "Income", "groups": {"< $25k": (1, 4), "$25-75k": (5, 7), ">= $75k": (8, 8)}},
    },
)

# ---------------------------------------------------------------------- breast cancer

_FNA_HELP = {
    "radius": "Mean distance from centre to points on the nuclear perimeter",
    "texture": "Standard deviation of grey-scale values",
    "perimeter": "Nuclear perimeter",
    "area": "Nuclear area",
    "smoothness": "Local variation in radius lengths",
    "compactness": "perimeter^2 / area - 1.0",
    "concavity": "Severity of concave portions of the contour",
    "concave points": "Number of concave portions of the contour",
    "symmetry": "Nuclear symmetry",
    "fractal dimension": "'Coastline approximation' - 1",
}


def _fna_meta(name: str) -> dict:
    for prefix, group in (("mean ", "Mean"), ("worst ", "Worst (largest)")):
        if name.startswith(prefix):
            m = name[len(prefix):]
            return feat(name, group, help=_FNA_HELP.get(m, ""))
    m = name.removesuffix(" error")
    return feat(name, "Standard error", help=_FNA_HELP.get(m, ""))


def _load_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    raw = load_breast_cancer(as_frame=True)
    # scikit-learn encodes 0 = malignant; flip so 1 = malignant (the risk).
    return raw.data.astype(float), (1 - raw.target).rename("malignant")


def _breast_cancer_features():
    from sklearn.datasets import load_breast_cancer
    return {c: _fna_meta(c) for c in load_breast_cancer().feature_names}


BREAST_CANCER = DatasetSpec(
    key="breast-cancer",
    name="Breast Cancer Wisconsin (Diagnostic)",
    source="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
    citation="Wolberg, Mangasarian, Street & Street. Breast Cancer Wisconsin (Diagnostic). "
             "UCI Machine Learning Repository, 1995. https://doi.org/10.24432/C5DW2B",
    license="CC BY 4.0",
    positive_label="Malignant",
    negative_label="Benign",
    outcome_text="malignant lesion",
    target_sensitivity=0.95,
    recommendations={
        "Low": "Low predicted risk. Routine imaging follow-up per institutional guidelines; "
               "correlate with clinical and radiological findings.",
        "Moderate": "Indeterminate risk. Consider core-needle biopsy or multidisciplinary review "
                    "before a management decision.",
        "High": "High predicted risk. Prioritise tissue diagnosis (core-needle biopsy) and "
                "early specialist referral.",
    },
    fhir_category="imaging",
    features={},  # filled lazily below
    loader=_load_breast_cancer,
)

REGISTRY = {s.key: s for s in (DIABETES, BREAST_CANCER)}


def load(key: str = "diabetes") -> Dataset:
    spec = REGISTRY[key]
    if not spec.features:
        spec.features = _breast_cancer_features()
    X, y = spec.loader()
    return Dataset(X, y, spec)


def load_csv(path: str, target: str, positive_value=1, name: str | None = None) -> Dataset:
    """Load any open tabular dataset with a binary outcome column (all numeric features)."""
    df = pd.read_csv(path)
    y = (df.pop(target) == positive_value).astype(int).rename(target)
    X = df.select_dtypes("number").astype(float)
    spec = DatasetSpec(
        key="custom", name=name or Path(path).name, source=str(path), citation=str(path),
        license="see source", positive_label="Positive", negative_label="Negative",
        outcome_text="positive outcome", target_sensitivity=0.85,
        recommendations={"Low": "Low predicted risk.", "Moderate": "Elevated risk; consider further assessment.",
                         "High": "High risk; prioritise clinical review."},
        fhir_category="exam", features={c: feat(c, "Features") for c in X.columns}, loader=lambda: (X, y),
    )
    return Dataset(X, y, spec)
