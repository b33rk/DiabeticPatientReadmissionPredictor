"""
Data validation service — stub implementation.
See README.md for full list of what needs to be completed.

TODO list:
  [ ] Create app/validator.py               — load GE expectation suite and validate data
  [ ] Create app/expectations/diabetic_suite.json — define rules for diabetic dataset
  [ ] Replace /validate stub with real GE validation
  [ ] Return structured errors per failing field (not just valid: true/false)
  [ ] Add great-expectations, pandas to requirements.txt
"""
from fastapi import FastAPI

app = FastAPI(title="Data Validation", version="0.0.1-stub")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: load Great Expectations suite from /app/expectations/diabetic_suite.json
# TODO: validate payload against the suite
# TODO: return 422 with field-level errors if any rule fails
# TODO: coordinate field names with training/src/preprocess.py
@app.post("/validate")
def validate(payload: dict):
    return {"valid": True, "errors": [], "note": "stub — no rules enforced yet"}
