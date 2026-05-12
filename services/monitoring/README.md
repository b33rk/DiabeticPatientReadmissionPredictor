# Monitoring Service

Evidently-based service that detects data drift and model drift by comparing recent production predictions against the training data distribution.

---

## What needs to be completed

### 1. Reference data loader — `app/reference.py`
Read the training data distribution from the `reference_data` table in PostgreSQL. This table is populated by the training pipeline after each training run (mean, std, percentiles per feature).

### 2. Production data loader — `app/production.py`
Read recent prediction inputs from the `prediction_log` table. The lookback window (default: 7 days) should be configurable via an environment variable.

### 3. Drift computation — `app/drift.py`
Use Evidently to compare the production window against the reference distribution. At minimum compute:
- Data drift per feature (statistical test: KS test for numerical, chi-squared for categorical)
- Prediction drift (change in readmission rate)
- Dataset drift summary (overall drift detected: yes/no)

### 4. Report generation — inside `app/drift.py`
Write Evidently HTML reports to `/app/reports/`. Name each file with a timestamp so old reports are not overwritten. These reports are served at `/monitoring` via Nginx.

### 5. Background task — `app/main.py`
Run drift computation on a schedule (every 1 hour) as a FastAPI background task. It should run automatically — no manual trigger needed.

### 6. Prometheus metrics — `app/main.py`
Update the `/metrics` endpoint with real values from the latest drift report:
- Drift score per feature
- Number of features flagged as drifted
- Timestamp of last drift check

---

## Files to create

```
app/
├── __init__.py       ← empty, already exists
├── main.py           ← update the stub
├── reference.py      ← create
├── production.py     ← create
└── drift.py          ← create

reports/              ← generated at runtime, gitignored
```

## Dependencies to add to `requirements.txt`

```
evidently==0.4.24
pandas==2.2.0
numpy==1.26.4
```

## How to run locally

```bash
# Needs prediction_log and reference_data to have data first
# Run at least one training cycle and one prediction before testing drift

docker-compose up -d postgres
docker-compose up monitoring

curl http://localhost:8002/health
curl http://localhost:8002/metrics
```

## Contract with other components

| Direction | Component | What is exchanged |
|-----------|-----------|-------------------|
| Reads | PostgreSQL | prediction_log (current data), reference_data (training distribution) |
| Writes | /app/reports/ | HTML drift reports (served by Nginx at /monitoring) |
| Scraped by | Prometheus | GET /metrics every 15s |
