# Training Pipeline

End-to-end ML training pipeline for the hospital readmission prediction model. Runs as a one-off job (not a long-running service) on a schedule or when new data arrives.

---

## What needs to be completed

### 1. Preprocessing — `src/preprocess.py`
The most important file to get right — errors here cause training-serving skew.

- Load raw CSV from `/data/raw/diabetic_data.csv`
- Handle missing values encoded as `?` in the dataset
- Encode age ranges (`[50-60)`) to midpoint numbers
- Binarise the readmission target: `<30` → 1, everything else → 0
- Fit a `OneHotEncoder` on categorical columns (race, gender, admission_type_id, etc.)
- Fit a `StandardScaler` on numerical columns (time_in_hospital, num_medications, etc.)
- Save the processed feature table to PostgreSQL (`patient_features` table) for Feast
- Save the fitted encoder and scaler — they must be logged to MLflow so the inference API loads the same ones

### 2. Training — `src/train.py` (replace stub)
- Load processed features from PostgreSQL
- Split into train / validation / test
- Apply SMOTE on the training set only to handle class imbalance
- Train XGBoost using parameters from `configs/train_config.yaml`
- Log all parameters, metrics, and artifacts to MLflow
- Call `src/evaluate.py` and `fairness/audit.py`

### 3. Evaluation — `src/evaluate.py`
Compute on the held-out test set:
- AUC-ROC
- F1 score
- Precision and recall
- Calibration curve

Log all metrics to MLflow.

### 4. Fairness audit integration — called from `src/train.py`
Call `fairness/audit.py` passing the trained model and test set. Write the results to the `fairness_audit` table in PostgreSQL. A model is only promoted if fairness metrics pass the thresholds in `fairness/thresholds.yaml`.

### 5. Reference data — `src/train.py`
After training, write the training data distribution (mean, std, percentiles per feature) to the `reference_data` table. The monitoring service reads this to detect drift.

### 6. Model registration — `src/register_model.py`
- Register the trained model in the MLflow Model Registry under the name in `configs/train_config.yaml`
- Compare new AUC-ROC against the current Production model
- Promote to Production only if the new model is better AND fairness thresholds pass

### 7. Add full dependencies to `requirements.txt`
The stub uses minimal dependencies. Add:
```
shap==0.45.0
fairlearn==0.10.0
aif360==0.6.1
imbalanced-learn==0.12.2
feast[redis,postgres]==0.40.0
optuna==3.6.1
```

---

## Files to create

```
src/
├── __init__.py         ← empty, already exists
├── train.py            ← replace stub with real implementation
├── preprocess.py       ← create
├── evaluate.py         ← create
└── register_model.py   ← create

configs/
└── train_config.yaml   ← already exists, update as needed
```

## How to run locally

```bash
# Infrastructure must be running first
docker-compose up -d postgres redis mlflow

# Run training
docker-compose --profile training run --rm training

# Check results in MLflow UI
open http://localhost:5000
```

## Important constraint
The encoder and scaler fitted in `preprocess.py` must be logged to MLflow as artifacts alongside the model. The inference API loads all three at startup. If the preprocessing step in the API doesn't match what happened at training time, predictions will be silently wrong.

## Contract with other components

| Direction | Component | What is exchanged |
|-----------|-----------|-------------------|
| Reads | PostgreSQL | Raw feature table, patient_features |
| Writes | PostgreSQL | reference_data, fairness_audit |
| Writes | MLflow | Model, encoder, scaler, metrics, SHAP plots, fairness reports |
| Writes | PostgreSQL (via Feast) | patient_features table (Feast offline store) |
| Triggers | feature_store | Run feast materialize after training to update Redis |
