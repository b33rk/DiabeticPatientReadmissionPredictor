from pydantic import BaseModel, Field
from typing import Literal

class PredictionInput(BaseModel):
    # Demographics
    race: Literal['Caucasian', 'AfricanAmerican', 'Other', 'Unknown']
    gender: Literal['Female', 'Male']
    age_group: Literal['<30', '30-60', '>60']
    
    # Admission/Discharge
    discharge_disposition: Literal['Home', 'Other']
    admission_source: Literal['Emergency', 'Referral', 'Other']
    medical_specialty: str
    primary_diagnosis: Literal['Circulatory', 'Respiratory', 'Diabetes', 'Genitourinary', 'Other', 'Missing']
    
    # Clinical (Numerical)
    time_in_hospital: int = Field(..., ge=1, le=14)
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_diagnoses: int
    
    # Engineered (Numerical)
    total_visits_last_year: int
    severity_index: int
    service_density: float
    
    # Others
    max_glu_serum: str
    A1Cresult: str
    change: str
    diabetesMed: str

class PredictionResponse(BaseModel):
    prediction: int  # 0 or 1
    probability: float
    confidence: float
    shap_values: dict
    model_version: str