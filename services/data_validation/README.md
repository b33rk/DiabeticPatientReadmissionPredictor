# Data Validation Service

Great Expectations service that validates incoming patient data before it reaches the model. Rejects malformed, out-of-range, or missing data with a clear error rather than letting it produce a silent wrong prediction.

---

## What needs to be completed

### 1. Expectation suite — `app/expectations/diabetic_suite.json`
Define the validation rules for the diabetic dataset. At minimum:
- All required columns are present and non-null
- `age` is numeric and within a valid range (or a valid age range string)
- `time_in_hospital` is an integer between 1 and 14
- `num_medications` is between 1 and 81
- `gender` is one of: Male, Female, Unknown/Invalid
- `race` is one of: Caucasian, AfricanAmerican, Hispanic, Asian, Other
- `discharge_disposition_id` is a valid integer

Coordinate with the training developer — validate against the exact field names and ranges present in the raw dataset.

### 2. Validator — `app/validator.py`
Load the expectation suite and run it against incoming data. Return a structured result: which fields passed, which failed, and why.

### 3. Real validate endpoint — `app/main.py`
Replace the stub `/validate` endpoint with the real one that:
- Calls the validator
- Returns 200 + `{"valid": true}` if all rules pass
- Returns 422 + a list of failed rules if any rule fails

The inference API calls this endpoint synchronously before every prediction — keep it fast.

### 4. Add `great-expectations` to `requirements.txt`
It was intentionally left out of the stub to keep initial build times short.

---

## Files to create

```
app/
├── __init__.py           ← empty, already exists
├── main.py               ← update the stub
├── validator.py          ← create
└── expectations/
    └── diabetic_suite.json  ← create
```

## Dependencies to add to `requirements.txt`

```
great-expectations==0.18.14
pandas==2.2.0
```

## How to run locally

```bash
docker-compose up -d postgres
docker-compose up data-validation

# Test with valid data
curl -X POST http://localhost:8003/validate \
  -H "Content-Type: application/json" \
  -d '{"age": 55, "gender": "Male", "time_in_hospital": 3}'

# Test with invalid data (should return 422)
curl -X POST http://localhost:8003/validate \
  -H "Content-Type: application/json" \
  -d '{"age": -5, "gender": "Unknown"}'
```

## Contract with other components

| Direction | Component | What is exchanged |
|-----------|-----------|-------------------|
| Called by | Inference API | POST /validate on every prediction request |
| Reads | /app/expectations/ | Expectation suite JSON files (volume-mounted from data/validation/) |
