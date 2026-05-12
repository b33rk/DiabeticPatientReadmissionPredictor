# Inference API

FastAPI service that receives patient data, runs the readmission prediction model, and returns a risk score with a SHAP explanation.

---

## What needs to be completed

### 1. Request schema — `app/schemas.py`
Define the Pydantic models for the prediction request and response.

The request should accept all features from the diabetic dataset (age, race, gender, time_in_hospital, num_medications, etc.). The response should include the prediction, probability, confidence score, and SHAP values per feature.

Coordinate with the training developer — the field names in the schema must match exactly what `preprocess.py` outputs.

### 2. Database connection — `app/db.py`
Create a PostgreSQL connection helper using the environment variables already provided (POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB). The API needs to write to `prediction_log` and `review_queue` on every request.

### 3. Security — `app/security.py`
Two things to implement:
- **Fernet encryption** — incoming request payloads are encrypted. Use the `ENCRYPTION_KEY` env var to decrypt them before processing.
- **JWT authentication** — protect the `/predict` endpoint. Use the `SECRET_KEY` env var.

### 4. Prediction logic — `app/predict.py`
The core of the service. On each request:
- Try to fetch pre-processed features from Redis using the patient ID (Feast online store)
- If not found in Redis, apply the saved encoder and scaler loaded from MLflow
- Run the XGBoost model
- Run SHAP `TreeExplainer` to get per-feature contributions
- Return the result

### 5. Model loading — inside `app/main.py`
At startup (using FastAPI lifespan), load from MLflow:
- The model (`MODEL_NAME` / `MODEL_STAGE` env vars are already set)
- The fitted encoder
- The fitted scaler

These should be loaded once at startup, not on every request.

### 6. Prediction logging — inside `app/main.py`
After every prediction, write to PostgreSQL:
- Full result to `prediction_log`
- If confidence is below threshold, also write to `review_queue`

### 7. Prometheus metrics — `app/metrics.py`
Expose a `/metrics` endpoint that Prometheus scrapes. At minimum track:
- Request count
- Request latency
- Prediction count by outcome (readmitted / not readmitted)

---

## Files to create

```
app/
├── __init__.py       ← empty, already exists
├── main.py           ← update the stub
├── schemas.py        ← create
├── predict.py        ← create
├── security.py       ← create
├── db.py             ← create
└── metrics.py        ← create
```

## Dependencies to add to `requirements.txt`

```
mlflow==2.13.0
xgboost==2.0.3
scikit-learn==1.4.2
shap==0.45.0
feast[redis,postgres]==0.40.0
pandas==2.2.0
numpy==1.26.4
python-jose==3.3.0
cryptography==42.0.0
prometheus-client==0.20.0
```

## How to run locally

```bash
# From the project root
docker-compose up -d postgres redis mlflow
docker-compose up api

# Test the health endpoint
curl http://localhost/api/health
curl http://localhost/api/health/tables
```

## Contract with other components

| Direction | Component | What is exchanged |
|-----------|-----------|-------------------|
| Calls | data-validation | POST /validate before every prediction |
| Reads | Redis | Pre-processed feature vectors by patient_id |
| Reads | MLflow | Production model, encoder, scaler at startup |
| Writes | PostgreSQL | prediction_log, review_queue |
| Scraped by | Prometheus | GET /metrics every 15s |
