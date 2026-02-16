# NHS-TT-Dynamic-Prediction-Models

This repository contains code for dynamic prediction models for NHS Talking Therapies outcomes using baseline predictors and early session-by-session symptom trajectories.

It complements the baseline-only models from a previous paper:

- **Baseline models (Paper 1):** `https://github.com/nourkanso/NHS-TT-Baseline-Prediction-Models-`
- **Dynamic models (Paper 2, this repository):** `https://github.com/nourkanso/NHS-TT-Dynamic-Prediction-Models-`

The two repositories are designed to be citable together:
- Paper 1 provides baseline-only prediction models and full stability analyses.
- Paper 2 extends this framework by incorporating early session-by-session change features and focuses on calibration curves across timepoints.

## Study design and data source

This is a retrospective observational cohort study using electronic health records from South London and Maudsley (SLaM) NHS Foundation Trust, which covers Lambeth, Southwark, Lewisham, and Croydon.

Data are extracted using the Clinical Record Interactive Search (CRIS) platform for adult patients who received a course of psychological therapy between **January 1, 2018** and **August 27, 2024**.

This repository does not include CRIS data. Users must follow all information governance and approvals required for CRIS-derived datasets.

## Interventions and sample selection

NHS Talking Therapies services provide evidence-based interventions organised in a stepped care model. This paper focuses on **high-intensity interventions** because their longer duration enables session-by-session longitudinal modelling.

To align with standard clinical practice at SLaM (where progress is typically reviewed after six sessions), and to reflect the NHS Talking Therapies manual limits on treatment length, the modelling dataset is expected to apply:

- adult patients (18 years or older)
- high-intensity interventions only
- **5–21 sessions**
- exclusions described in Supplementary Figure 1 (for example, incomplete baseline outcome measures where required)

All cohort filtering and exclusions are assumed to be applied before running the modelling scripts.

## Outcomes

At each session, patients complete:

- Depression symptoms: PHQ-9 (0–27)
- Anxiety symptoms: GAD-7 (0–21)
- Functional impairment: WSAS (0-40)

Post-treatment outcomes (six total) are derived for PHQ-9 and GAD-7:

- Reliable improvement
- Recovery
- Reliable recovery

These outcomes align with routine NHS Talking Therapies service metrics.

## Predictors

### Baseline predictors
Baseline predictors include demographic, socioeconomic, and clinical variables (for example: age, gender, religion, sexual orientation, ethnicity, English language proficiency, deprivation index, benefits, Statutory Sick Pay, employment status, long-term condition, disability, psychotropic medication prescriptions, previous referrals, primary diagnosis, and screening items for phobias), plus baseline PHQ-9, GAD-7, and WSAS.

Predictors included in modelling should meet the study completeness threshold (at least 70% completeness). This is assumed to be handled during dataset preparation.

### Session-by-session predictors (sessions 2–5)

This repository incorporates PHQ-9, GAD-7, and WSAS information from sessions 2–5 to capture early symptom change. These are aggregated to reflect different aspects of early response:

- **Latest score (session 2 onwards):** current severity at the most recent session.
- **Linear slope (session 2 onwards):** average rate of change across early sessions.
- **Variance (session 3 onwards):** variability across early sessions.
- **Non-linear slopes (session 4 onwards):** captures curvature / non-linear change patterns.

For each outcome, models are fitted at five timepoints:
- baseline (timepoint 1)
- sessions 2, 3, 4, and 5 (timepoints 2–5)

At each timepoint, multiple model specifications are produced:
- baseline only
- one aggregation family at a time (latest, slope, variance, non-linear)
- a combined “all” model including all available aggregations for that timepoint

The combined “all” model is the main model reported in the paper.

## Modelling and evaluation

### Preprocessing and missing data
- Continuous predictors are scaled and centred.
- Missing predictors are imputed using K-nearest neighbours imputation (K = 5), implemented with FastKNNImputer for speed.

Important: K-nearest neighbours imputation requires numeric features. The modelling dataset should therefore use numeric and dummy-coded predictors.

### Model development
Elastic net logistic regression is used for all models.
Hyperparameters are tuned using grid search with 5-fold cross-validation:
- mixing parameter (l1_ratio): 0.1, 0.3, 0.5, 0.7, 0.9
- regularisation strength (C): 0.01, 0.1, 1.0, 10.0

### Internal validation
Internal validation uses nested bootstrap resampling (200 bootstrap samples). At each bootstrap iteration, all modelling steps are repeated (preprocessing, imputation, hyperparameter tuning, and model fitting) to avoid information leakage.

Discrimination and calibration are evaluated:
- AUC is optimism-corrected using Harrell’s bootstrap optimism correction.
- AUC uncertainty is estimated using a bias-based interval derived from optimism percentiles.
- Calibration is assessed using logit-based calibration intercept and slope (logistic recalibration).
- Calibration curves use LOWESS smoothing, with an optimism-corrected LOWESS curve computed by subtracting mean LOWESS optimism across bootstrap iterations from the apparent LOWESS curve.

### Outputs
For each outcome × timepoint × aggregation specification, the scripts produce:
- a calibration curve figure (apparent + optimism-corrected)
- bootstrap results table
- metrics summary table
- the fitted final model object

## Running the models

1. Prepare an analysis-ready dataset:
- `data/analysis_table_dynamic.parquet`

2. Edit `scripts/run_dynamic_models.py`:
- set baseline predictors
- set PHQ-9, GAD-7, WSAS session column names (sessions 1–5)
- set outcome column names

3. Run:
```bash
python scripts/run_dynamic_models.py
