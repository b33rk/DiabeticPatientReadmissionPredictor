import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pathlib import Path

def clean_data(df):
    df['readmit_30_days'] = (df['readmitted'] == '<30').astype(int)

    df = df[df['gender'] != 'Unknown/Invalid'].copy()
    
    age_map = {
        '[0-10)': '<30', '[10-20)': '<30', '[20-30)': '<30',
        '[30-40)': '30-60', '[40-50)': '30-60', '[50-60)': '30-60',
        '[60-70)': '>60', '[70-80)': '>60', '[80-90)': '>60', '[90-100)': '>60'
    }
    df['age_group'] = df['age'].map(age_map)
    
    df['race'] = df['race'].replace('?', 'Unknown')
    df['race'] = df['race'].replace(['Asian', 'Hispanic', 'Other'], 'Other')

    df['discharge_disposition'] = df['discharge_disposition_id'].apply(lambda x: 'Home' if x == 1 else 'Other')
    df['admission_source'] = df['admission_source_id'].apply(lambda x: 'Emergency' if x == 7 else ('Referral' if x in [1, 2, 3] else 'Other'))
    df['medical_specialty'] = df['medical_specialty'].replace('?', 'Missing')
    
    def map_diag(code):
        if str(code) == '?': return 'Missing'
        if 'V' in str(code) or 'E' in str(code): return 'Other'
        try:
            icd = float(code)
            if 390 <= icd <= 459 or icd == 785: return 'Circulatory'
            if 460 <= icd <= 519 or icd == 786: return 'Respiratory'
            if 520 <= icd <= 579 or icd == 787: return 'Digestive'
            if np.floor(icd) == 250: return 'Diabetes'
            if 800 <= icd <= 999: return 'Injury'
            if 710 <= icd <= 739: return 'Musculoskeletal'
            if 580 <= icd <= 629 or icd == 788: return 'Genitourinary'
            if 140 <= icd <= 239: return 'Neoplasms'
            return 'Other'
        except:
            return 'Other'
            
    df['primary_diagnosis'] = df['diag_1'].apply(map_diag)
    
    cols =[
        'patient_nbr', 'race', 'gender', 'age_group',
        'discharge_disposition', 'admission_source', 'medical_specialty',
        'primary_diagnosis', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
        'readmit_30_days'
    ]
    return df[cols]

def main():
    raw_path = Path("/data/raw/diabetic_data.csv")
    processed_path = Path("/data/processed/features.parquet")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Reading and cleaning raw data...")
    df = pd.read_csv(raw_path)
    df_clean = clean_data(df)
    
    numeric_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
        'num_medications', 'number_outpatient', 'number_emergency', 
        'number_inpatient', 'number_diagnoses'
    ]
    categorical_features = [
        'race', 'gender', 'age_group', 'discharge_disposition', 
        'admission_source', 'medical_specialty', 'primary_diagnosis',
        'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'
    ]
    
    print("Fitting ColumnTransformer (Encoder + Scaler)...")
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])
    preprocessor.fit(df_clean)
    joblib.dump(preprocessor, "/data/processed/preprocessor.pkl")
    
    print("Saving cleaned feature table to Parquet (for Feast)...")
    df_clean.to_parquet(processed_path, index=False)
    
    print("Writing to PostgreSQL database...")
    user = os.getenv("POSTGRES_USER", "readmission")
    pwd = os.getenv("POSTGRES_PASSWORD", "changeme")
    host = os.getenv("POSTGRES_HOST", "postgres")
    db = os.getenv("POSTGRES_DB", "readmission")
    
    engine = create_engine(f"postgresql://{user}:{pwd}@{host}:5432/{db}")
    df_clean.to_sql("preprocessed_data", engine, if_exists="replace", index=False)
    
    print("=== Preprocessing Complete ===")

if __name__ == "__main__":
    main()