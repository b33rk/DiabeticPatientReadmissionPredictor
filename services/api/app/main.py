"""
Inference API — stub implementation.
See README.md for full list of what needs to be completed.

TODO list:
  [ ] Create app/schemas.py     — Pydantic request/response models
  [ ] Create app/db.py          — PostgreSQL connection helper
  [ ] Create app/security.py    — Fernet decryption + JWT auth
  [ ] Create app/predict.py     — Feast lookup -> preprocessing -> model -> SHAP
  [ ] Create app/metrics.py     — Prometheus metrics (request count, latency)
  [ ] Load model/encoder/scaler from MLflow at startup (FastAPI lifespan)
  [ ] Replace /predict stub with real logic
  [ ] Write to prediction_log after every prediction
  [ ] Write to review_queue when confidence < threshold
  [ ] Add ML deps to requirements.txt (mlflow, xgboost, shap, feast...)
"""
import os
import psycopg2
import redis
from fastapi import FastAPI

app = FastAPI(title="Readmission API", version="0.0.1-stub")


def _pg_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        dbname=os.getenv("POSTGRES_DB", "readmission"),
        user=os.getenv("POSTGRES_USER", "readmission"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme"),
    )


def _redis_conn():
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
    )


@app.get("/health")
def health():
    status = {"api": "ok", "postgres": "unknown", "redis": "unknown"}
    try:
        conn = _pg_conn()
        conn.cursor().execute("SELECT 1")
        conn.close()
        status["postgres"] = "ok"
    except Exception as e:
        status["postgres"] = f"error: {e}"
    try:
        _redis_conn().ping()
        status["redis"] = "ok"
    except Exception as e:
        status["redis"] = f"error: {e}"
    status["status"] = "ok" if all(v == "ok" for v in status.values()) else "degraded"
    return status


@app.get("/health/tables")
def health_tables():
    expected = ["prediction_log", "review_queue", "fairness_audit", "reference_data"]
    results = {}
    try:
        conn = _pg_conn()
        cur = conn.cursor()
        for table in expected:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name=%s)",
                (table,),
            )
            results[table] = "exists" if cur.fetchone()[0] else "MISSING"
        conn.close()
    except Exception as e:
        return {"error": str(e)}
    return results


# TODO: add Fernet decryption (app/security.py) before this runs
# TODO: add JWT auth dependency
# TODO: call data-validation service before running model
# TODO: load model at startup, not per-request
# TODO: run SHAP TreeExplainer after model inference
# TODO: write result to prediction_log in PostgreSQL
# TODO: write to review_queue if confidence < threshold
@app.post("/predict")
def predict_stub(payload: dict):
    return {
        "prediction": 0,
        "probability": 0.42,
        "confidence": 0.80,
        "shap_values": {},
        "model_version": "stub",
        "note": "stub — no model loaded yet",
    }
