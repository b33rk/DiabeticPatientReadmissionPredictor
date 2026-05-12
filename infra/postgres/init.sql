-- Runs once when the postgres container is first created.
-- Creates the tables needed beyond what MLflow auto-creates.

-- Prediction log: every inference result is persisted here for monitoring
CREATE TABLE IF NOT EXISTS prediction_log (
    id              SERIAL PRIMARY KEY,
    patient_id      TEXT NOT NULL,
    input_hash      TEXT NOT NULL,           -- SHA256 of encrypted input for dedup
    prediction      SMALLINT NOT NULL,       -- 0 = no readmission, 1 = readmission
    probability     FLOAT NOT NULL,
    confidence      FLOAT NOT NULL,
    shap_values     JSONB,                   -- per-feature SHAP contributions
    model_version   TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Reference data: store training distribution for Evidently drift comparison
CREATE TABLE IF NOT EXISTS reference_data (
    id          SERIAL PRIMARY KEY,
    feature_name TEXT NOT NULL,
    mean        FLOAT,
    std         FLOAT,
    p25         FLOAT,
    p50         FLOAT,
    p75         FLOAT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Fairness audit results
CREATE TABLE IF NOT EXISTS fairness_audit (
    id              SERIAL PRIMARY KEY,
    audit_run_id    TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    sensitive_attr  TEXT NOT NULL,  -- 'race', 'age_group', 'gender'
    group_value     TEXT NOT NULL,
    metric_name     TEXT NOT NULL,  -- 'demographic_parity_diff', 'equalized_odds_diff'
    metric_value    FLOAT NOT NULL,
    threshold       FLOAT NOT NULL,
    passed          BOOLEAN NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Low-confidence queue: predictions below threshold sent here for human review
CREATE TABLE IF NOT EXISTS review_queue (
    id              SERIAL PRIMARY KEY,
    prediction_log_id INT REFERENCES prediction_log(id),
    status          TEXT DEFAULT 'pending',  -- pending | reviewed | escalated
    reviewer_id     TEXT,
    reviewer_note   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at     TIMESTAMPTZ
);
