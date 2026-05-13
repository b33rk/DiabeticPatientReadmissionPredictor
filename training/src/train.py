import os
import sys
import yaml
import joblib
import psycopg2
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
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

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


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
        X, y, test_size=config['data']['test_size'],
        random_state=config['data']['random_seed'], stratify=y
    )

    num_cols = [c for c in X.columns if X[c].dtype in [np.float64, np.int64]]
    populate_reference_data(X_train, num_cols)

    print("Training base XGBoost model...")

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    best_params = config['xgboost']
    if config['optuna']['enabled']:
        print(f"Starting Optuna search ({config['optuna']['n_trials']} trials)...")
        X_t, X_v, y_t, y_v = train_test_split(X_train_trans, y_train, test_size=0.2, stratify=y_train)
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, X_t, y_t, X_v, y_v), n_trials=config['optuna']['n_trials'])
        best_params = study.best_params
        print(f"Best Tuning Params: {best_params}")

    print("Training optimized base model...")
    base_model = XGBClassifier(**best_params)
    base_model.fit(X_train_trans, y_train)

    print("Applying Intersectional ThresholdOptimizer (Race + Age)...")
    X_train_sens = X_train['race'].astype(str) + "_" + X_train['age_group'].astype(str)
    X_test_sens = X_test['race'].astype(str) + "_" + X_test['age_group'].astype(str)

    mitigated_model = ThresholdOptimizer(
        estimator=base_model,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method='predict_proba'
    )
    mitigated_model.fit(X_train_trans, y_train, sensitive_features=X_train_sens)

    with mlflow.start_run(run_name="final_optimized_fair_run") as run:
        X_test_trans_arr = np.asarray(X_test_trans)
        y_pred = mitigated_model.predict(X_test_trans_arr, sensitive_features=X_test_sens)
        y_prob = mitigated_model._pmf_predict(X_test_trans_arr, sensitive_features=X_test_sens)[:, 1]
        
        metrics = {
            "auc": roc_auc_score(y_test, y_prob),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "balanced_acc": balanced_accuracy_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)
        mlflow.log_params(best_params)
        print(f"Final Metrics: {metrics}")
        
        print("Running Multi-Attribute Fairness Audit...")
        fairness_passed = run_fairness_audit(
            y_test, y_pred, X_test[['race', 'age_group', 'gender']], 
            run.info.run_id, config['model']['name']
        )
        mlflow.log_param("fairness_passed", fairness_passed)
        
        signature = infer_signature(X_test_trans_arr[:10], y_pred[:10])
        mlflow.sklearn.log_model(mitigated_model, "model", signature=signature)
        mlflow.log_artifact("/data/processed/preprocessor.pkl", "preprocessing")
        
        if fairness_passed and metrics['auc'] > 0.60:
            print("--- VALIDATION PASSED: Registering Production Model ---")
            model_uri = f"runs:/{run.info.run_id}/model"
            reg = mlflow.register_model(model_uri, config['model']['name'])
            
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            client.transition_model_version_stage(
                name=config['model']['name'], version=reg.version,
                stage="Production", archive_existing_versions=True
            )
        else:
            print("--- MODEL REJECTED: Failed Fairness or Performance gate ---")


if __name__ == "__main__":
    main()