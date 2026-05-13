import os
import sys
import yaml
import joblib
import psycopg2
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score
)
from mlflow.models.signature import infer_signature

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from fairness.audit import run_fairness_audit


def populate_reference_data(df_train, num_cols):
    """Writes baseline statistics for Evidently Monitoring (T1.8)"""

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        dbname=os.getenv("POSTGRES_DB", "readmission"),
        user=os.getenv("POSTGRES_USER", "readmission"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme"),
    )

    try:
        cur = conn.cursor()

        cur.execute("TRUNCATE TABLE reference_data;")

        for col in num_cols:
            if col not in df_train.columns:
                print(f"Skipping missing column: {col}")
                continue

            series = df_train[col].dropna()

            cur.execute("""
                INSERT INTO reference_data
                (feature_name, mean, std, p25, p50, p75)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                col,
                float(series.mean()),
                float(series.std()),
                float(series.quantile(0.25)),
                float(series.quantile(0.50)),
                float(series.quantile(0.75))
            ))

        conn.commit()

    finally:
        cur.close()
        conn.close()


def main():
    with open("configs/train_config.yaml") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )

    mlflow.set_experiment(config['mlflow']['experiment_name'])

    print("Loading processed data and preprocessor...")

    df = pd.read_parquet(config['data']['processed_path'])
    preprocessor = joblib.load("/data/processed/preprocessor.pkl")
    
    required_cols = ['readmit_30_days', 'patient_nbr', 'age_group']
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df.drop(columns=['readmit_30_days', 'patient_nbr'])
    y = df['readmit_30_days']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_seed'],
        stratify=y
    )

    numeric_cols = [
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_outpatient',
        'number_emergency',
        'number_inpatient',
        'number_diagnoses'
    ]

    print("Writing distribution stats to reference_data table...")
    populate_reference_data(X_train, numeric_cols)

    print("Training base XGBoost model...")

    X_train_trans = preprocessor.transform(X_train)

    base_model = XGBClassifier(
        n_estimators=config['xgboost']['n_estimators'],
        max_depth=config['xgboost']['max_depth'],
        learning_rate=config['xgboost']['learning_rate'],
        scale_pos_weight=3.0,
        eval_metric='logloss',
        random_state=42
    )

    base_model.fit(X_train_trans, y_train)

    print("Mitigating bias using Intersectional Sensitive Groups...")
    
    X_train_sensitive = X_train['race'].astype(str) + "_" + X_train['age_group'].astype(str)
    X_test_sensitive = X_test['race'].astype(str) + "_" + X_test['age_group'].astype(str)

    mitigated_model = ThresholdOptimizer(
        estimator=base_model,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method='predict_proba'
    )

    mitigated_model.fit(
        X_train_trans,
        y_train,
        sensitive_features=X_train_sensitive
    )

    with mlflow.start_run(run_name="mitigated_xgboost_run") as run:
        print("Evaluating mitigated model...")

        X_test_trans = preprocessor.transform(X_test)
        X_test_trans = np.asarray(X_test_trans)
        y_pred = mitigated_model.predict(
            X_test_trans,
            sensitive_features=X_test_sensitive
        )

        try:
            y_prob = mitigated_model._pmf_predict(
                X_test_trans,
                sensitive_features=X_test_sensitive
            )[:, 1]

        except Exception:
            print("Falling back to base estimator probabilities...")
            y_prob = base_model.predict_proba(X_test_trans)[:, 1]

        metrics = {
            "auc": roc_auc_score(y_test, y_prob),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "balanced_acc": balanced_accuracy_score(y_test, y_pred)
        }

        mlflow.log_metrics(metrics)

        print(f"Metrics: {metrics}")

        print("Running Fairness Audit...")

        fairness_cols = [
            c for c in ['race', 'age_group', 'gender']
            if c in X_test.columns
        ]

        fairness_passed = run_fairness_audit(
            y_test,
            y_pred,
            X_test[fairness_cols],
            run.info.run_id,
            config['model']['name']
        )

        mlflow.log_param("fairness_passed", fairness_passed)

        signature = infer_signature(
            np.asarray(X_test_trans[:10]),
            y_pred[:10]
        )

        mlflow.sklearn.log_model(
            sk_model=mitigated_model,
            artifact_path="model",
            signature=signature
        )

        mlflow.log_artifact(
            "/data/processed/preprocessor.pkl",
            "preprocessing"
        )

        if fairness_passed and metrics['auc'] > 0.62:
            print("--- VALIDATION PASSED ---")
            print("Registering model to MLflow Model Registry...")

            model_uri = f"runs:/{run.info.run_id}/model"

            reg_version = mlflow.register_model(
                model_uri,
                config['model']['name']
            )

            from mlflow.tracking import MlflowClient

            client = MlflowClient()

            try:
                client.transition_model_version_stage(
                    name=config['model']['name'],
                    version=reg_version.version,
                    stage="Production",
                    archive_existing_versions=True
                )

                print(
                    f"Model version {reg_version.version} "
                    f"promoted to PRODUCTION."
                )

            except Exception as e:
                print(f"Stage transition failed: {e}")

        else:
            print("--- VALIDATION FAILED ---")
            print(
                "Model failed Fairness or "
                "Performance requirements. Not registered."
            )


if __name__ == "__main__":
    main()