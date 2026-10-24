# MedAI-nEXUS CDSS: Diabetes Risk Screening

A clinical decision support tool that estimates how likely someone is to have diabetes or prediabetes, using details a clinic already has on file: blood pressure, cholesterol, BMI, age, lifestyle and so on. It explains every prediction, checks itself for fairness across groups, and reads and writes HL7 FHIR R4 so it can talk to hospital systems.

Built for **MedAI-nEXUS 2026** (Track B: AI-Driven Clinical Decision Support), SCOPE, VIT Vellore.

**Demo video:** [Google Drive](https://drive.google.com/file/d/1kr9XcrdjecHLGDs4pERtti0Mm-rfw0sp/view?usp=sharing)

> Research prototype for learning and demonstration. Not a medical device, and not meant for diagnosing real patients.

## Why

Blood tests for diabetes are cheap. Deciding who should get one first is the hard part, especially in a busy clinic. This tool ranks patients by risk so the people most likely to need a test get one sooner.

## What it does

- **Predict.** A gradient-boosting model with isotonic calibration gives a risk percentage you can take at face value (a predicted 30% means roughly 30 in 100 such people really have the condition).
- **Classify.** Two cut-offs, picked on the validation set, sort patients into Low (< 14.1%), Moderate, or High (≥ 39.0%), each with a suggested next step.
- **Explain.** Shapley values show how many percentage points each feature added to or took away from a patient's risk. They add up to the final number.
- **Audit.** Sensitivity, specificity and AUROC are reported separately by sex, age band and income band.
- **Connect.** Takes a FHIR R4 `Bundle` of `Observation`s as input (BMI matched by LOINC 39156-5) and returns a FHIR `RiskAssessment`.

## Results

Measured once on 38,052 held-out test patients:

| AUROC | Sensitivity | Specificity | NPV | Brier |
|---|---|---|---|---|
| 0.828 | 79.5% | 71.0% | 95.5% | 0.097 |

Gradient boosting (0.833 validation AUROC) narrowly beat a random forest (0.830) and logistic regression (0.825). The fairness audit found the model performs about the same for women and men, but with one shared cut-off it catches only 45.5% of cases in adults aged 18–44, where diabetes is rare. Age-specific cut-offs are the planned fix.

## Data

[CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) (UCI #891), from the US BRFSS 2015 survey: 253,680 adults, 21 features, 13.9% positive, no missing values. `train.py` downloads it automatically on the first run.

## Getting started

You need Python 3.10 or newer.

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Training takes around 30 seconds. It saves the model to `artifacts/`, charts and metrics to `reports/`, and example patients plus FHIR sample files to `samples/`. Once the app is running, open http://127.0.0.1:5000.

To run the tests:

```bash
python -m unittest discover tests
```

`train.py` also accepts `--dataset breast-cancer`, or `--csv your.csv --target column` to train on another tabular dataset with a yes/no outcome.

## Using the app

Load a held-out test patient from the dropdown (its true outcome is shown), upload a FHIR Bundle from `samples/`, or fill in the form yourself. Press **Assess risk** to see the gauge, risk band, recommendation, the "Why?" chart and the FHIR output. Change an input, say general health, and run it again to watch the risk move. The model card and the fairness table are further down the page.

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

The data is self-reported, cross-sectional and from the US. People in India tend to develop diabetes at a lower BMI, so the model would need recalibrating and validating on Indian cohorts (ICMR-INDIAB, for instance) before any real use. No lab values are included either; adding HbA1c or fasting glucose where available should help.

## Team

Parth Khandelwal (23BDS0043), Yash Poddar (23BDS0195), Krishna Gupta (23BDS0014)
School of Computer Science and Engineering (SCOPE), VIT Vellore
