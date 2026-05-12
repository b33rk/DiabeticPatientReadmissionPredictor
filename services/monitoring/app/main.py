"""
Monitoring service — stub implementation.
See README.md for full list of what needs to be completed.

TODO list:
  [ ] Create app/reference.py   — load training distribution from reference_data table
  [ ] Create app/production.py  — load recent predictions from prediction_log table
  [ ] Create app/drift.py       — run Evidently drift computation, write HTML reports
  [ ] Add background task to main.py that runs drift check every hour
  [ ] Replace /metrics stub with real drift scores from latest report
  [ ] Add evidently, pandas to requirements.txt
"""
import os
import psycopg2
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI(title="Monitoring Service", version="0.0.1-stub")


@app.get("/health")
def health():
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            dbname=os.getenv("POSTGRES_DB", "readmission"),
            user=os.getenv("POSTGRES_USER", "readmission"),
            password=os.getenv("POSTGRES_PASSWORD", "changeme"),
        )
        conn.close()
        return {"status": "ok", "postgres": "ok"}
    except Exception as e:
        return {"status": "degraded", "postgres": str(e)}


# TODO: replace with real drift metrics from app/drift.py
# TODO: add per-feature drift scores
# TODO: add drifted_feature_count metric
# TODO: add last_drift_check_timestamp metric
@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return (
        "# HELP readmission_drift_score Stub drift score\n"
        "# TYPE readmission_drift_score gauge\n"
        "readmission_drift_score 0.0\n"
    )
