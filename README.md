# MedAI-nEXUS CDSS: Diabetes Risk Screening

Give it a few routine health details (blood pressure, cholesterol, BMI, age, a handful of lifestyle answers) and it tells you how likely that person is to have diabetes or prediabetes. The percentage comes with reasons attached. We also checked whether it treats different groups fairly, and since it speaks HL7 FHIR R4, hospital software can send it data and read what comes back.

Built for **MedAI-nEXUS 2026** (Track B: AI-Driven Clinical Decision Support), SCOPE, VIT Vellore.

**Demo video:** [Google Drive](https://drive.google.com/file/d/1kr9XcrdjecHLGDs4pERtti0Mm-rfw0sp/view?usp=sharing)

> A student research prototype. It is not a medical device, so please don't use it to diagnose anyone.

## Why

A diabetes blood test costs next to nothing. Getting the right people to take one is another matter, particularly when a clinic has fifty patients waiting and two doctors. So we rank patients by risk, and whoever looks most likely to need the test gets it first.

## What it does

- **Predict.** Under the hood it's gradient boosting, with isotonic calibration added so the output behaves like an honest probability: out of 100 people scored at 30%, about 30 should actually turn out positive.
- **Classify.** Low sits below 14.1% and High starts at 39.0%; anything between is Moderate. Both lines were tuned on the validation set, and each band carries its own suggested next step.
- **Explain.** Why did this patient score 54%? Shapley values answer that in percentage points per feature, and the pieces sum to the final figure.
- **Audit.** We don't stop at one overall score. AUROC, sensitivity and specificity are broken out by sex, by age band and by income band.
- **Connect.** Feed it a FHIR R4 `Bundle` of `Observation`s (it finds BMI through LOINC code 39156-5) and you get a FHIR `RiskAssessment` back.

## Results

Scored a single time, on 38,052 patients the model never saw during training:

| AUROC | Sensitivity | Specificity | NPV | Brier |
|---|---|---|---|---|
| 0.828 | 79.5% | 71.0% | 95.5% | 0.097 |

Three models went head to head on validation. Gradient boosting took it with 0.833 AUROC, though random forest (0.830) and logistic regression (0.825) were close behind. On fairness, women and men get roughly equal treatment. Young adults don't. With a single cut-off for everyone, the model spots just 45.5% of cases among 18 to 44-year-olds (diabetes is uncommon at that age, so their scores rarely cross the line), and we plan to fix that with separate cut-offs per age group.

## Data

We trained on the [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) dataset, number 891 on UCI. It's drawn from the US BRFSS phone survey of 2015. That gives 253,680 adults with 21 features each; 13.9% are positive and nothing is missing. You don't need to fetch it yourself, because `train.py` grabs it the first time it runs.

## Getting started

Python 3.10 or later is required.

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Expect training to finish in roughly half a minute. Afterwards the model is in `artifacts/`, charts and metrics are in `reports/`, and `samples/` holds example patients and FHIR files. With the app up, point your browser at http://127.0.0.1:5000.

Tests:

```bash
python -m unittest discover tests
```

Want a different dataset? Pass `--dataset breast-cancer`, or bring your own CSV with `--csv your.csv --target column` (any yes/no outcome column will do).

## Using the app

There are three ways to get a patient in. You can pick one of the held-out test cases from the dropdown, which also shows whether they really had diabetes; upload a FHIR Bundle from `samples/`; or just type values into the form. Hit **Assess risk**. Out comes a gauge, the band, a recommendation, a "Why?" chart and the FHIR output. Try nudging general health from Poor to Excellent and running it again, the number drops quite a bit. Scroll further for the model card and fairness table.

### API

| Endpoint | Purpose |
|---|---|
| `GET /api/meta` | Model info, feature list, metrics, fairness table, example patients |
| `POST /api/predict` | Feature values as JSON in; risk, band, explanation and FHIR `RiskAssessment` out |
| `POST /api/fhir` | FHIR R4 Bundle in; same response as `/api/predict` |

## Project layout

```
medai_cdss/
  data.py      dataset loading and feature descriptions
  model.py     training, model comparison, calibration, cut-offs, fairness audit, predictor
  explain.py   Monte-Carlo Shapley values and global importance
  fhir.py      FHIR R4 Bundle input and RiskAssessment output
app.py         Flask web app and REST API
train.py       trains the model, writes charts, metrics and sample patients
templates/     the web dashboard
tests/         end-to-end tests
```

## Limitations

Everything here was learned from self-reported answers, collected once, in the US. That matters. Indians often develop diabetes at a lower BMI than Americans do, so before this goes near a real clinic it would have to be recalibrated and validated on Indian cohorts such as ICMR-INDIAB. There are no lab values in the data either. Adding HbA1c or fasting glucose, wherever a clinic has them, ought to lift accuracy.

## Team

Parth Khandelwal (23BDS0043), Yash Poddar (23BDS0195), Krishna Gupta (23BDS0014)
School of Computer Science and Engineering (SCOPE), VIT Vellore
