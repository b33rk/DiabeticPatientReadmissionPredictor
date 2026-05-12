"""
Training pipeline — stub implementation.
See README.md for full list of what needs to be completed.

TODO list:
  [ ] Create src/preprocess.py  — clean raw CSV, fit encoder+scaler, write to PostgreSQL
  [ ] Create src/evaluate.py    — AUC, F1, precision, recall, calibration on test set
  [ ] Create src/register_model.py — MLflow model registration + Production promotion
  [ ] Replace connectivity check below with real training logic
  [ ] Call fairness/audit.py after evaluation — model only promoted if fairness passes
  [ ] Write training distribution to reference_data table (for Evidently)
  [ ] Log encoder and scaler to MLflow as artifacts alongside the model
  [ ] Add full ML dependencies to requirements.txt (shap, fairlearn, aif360, feast...)

NOTE: preprocess.py must be written first — all other steps depend on its output.
The feature names it produces must match feature_store/feature_repo/feature_views.py
and services/api/app/schemas.py exactly.
"""
import os
import sys
import psycopg2
import redis
import mlflow

print("=== Training stub — connectivity check ===")

# TODO: replace everything below with real training logic once preprocess.py exists

try:
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        dbname=os.getenv("POSTGRES_DB", "readmission"),
        user=os.getenv("POSTGRES_USER", "readmission"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme"),
    )
    conn.close()
    print("[OK] PostgreSQL connected")
except Exception as e:
    print(f"[FAIL] PostgreSQL: {e}")
    sys.exit(1)

try:
    redis.Redis(host=os.getenv("REDIS_HOST", "redis")).ping()
    print("[OK] Redis connected")
except Exception as e:
    print(f"[FAIL] Redis: {e}")
    sys.exit(1)

try:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.tracking.MlflowClient().search_experiments()
    print("[OK] MLflow connected")
except Exception as e:
    print(f"[FAIL] MLflow: {e}")
    sys.exit(1)

with mlflow.start_run(run_name="stub-connectivity-check"):
    mlflow.log_param("run_type", "stub")
    mlflow.log_metric("auc_roc", 0.0)
    print("[OK] MLflow test run logged")

print("\nAll connections OK.")
print("=== Done ===")
