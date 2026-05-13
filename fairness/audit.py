import os
import yaml
import psycopg2
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def run_fairness_audit(y_true, y_pred, sensitive_df, run_id, model_version):
    with open('fairness/threshold.yaml') as f:
        thresholds = yaml.safe_load(f)
        
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        dbname=os.getenv("POSTGRES_DB", "readmission"),
        user=os.getenv("POSTGRES_USER", "readmission"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme"),
    )
    cur = conn.cursor()
    all_passed = True
    
    for attr in thresholds['sensitive_attributes']:
        if attr not in sensitive_df.columns: continue
        
        dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_df[attr])
        dp_thresh = thresholds['demographic_parity_diff']
        dp_pass = bool(dp_diff <= dp_thresh)
        
        eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_df[attr])
        eo_thresh = thresholds['equalized_odds_diff']
        eo_pass = bool(eo_diff <= eo_thresh)
        
        all_passed = all_passed and dp_pass and eo_pass
        
        query = """
        INSERT INTO fairness_audit (audit_run_id, model_version, sensitive_attr, group_value, metric_name, metric_value, threshold, passed)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (run_id, model_version, attr, "all", "demographic_parity_diff", float(dp_diff), float(dp_thresh), dp_pass))
        cur.execute(query, (run_id, model_version, attr, "all", "equalized_odds_diff", float(eo_diff), float(eo_thresh), eo_pass))
        
    conn.commit()
    cur.close()
    conn.close()
    
    return all_passed