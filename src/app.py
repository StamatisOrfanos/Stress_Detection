import os
from typing import Optional, Dict, Any
from pandas import DataFrame, isna
from fastapi import FastAPI
import joblib
from pandas import DataFrame
from numpy import argmax, mean
from joblib import load
from os import getenv, path
from dotenv import load_dotenv
from src.deployment_utils import init_model_weights
from src.healthcare_stress_rules import compute_stress_healthcare, healthcare_compute_confidence
from src.input import DataFrameInput, EducationComputeInput, HealthcareStressInput
from src.education_stress_rules import education_compute_confidence, education_compute_stress
from src.academic_physio_stress import compute_stress_academic_physio
from src.input import AcademicPhysioStressInput

# Step 1: Download model weights at runtime if needed
load_dotenv() 
MODEL_URL = getenv('URL')
init_model_weights(MODEL_URL) # type: ignore

# Step 2: Load model
MODEL_PATH = path.join('model_weights', 'model.pkl')
if not path.exists(MODEL_PATH):
    raise FileNotFoundError(f'Model not found at {MODEL_PATH}')
model = load(MODEL_PATH)

# Step 3: Start API
server = FastAPI(title='Stress Detector API')



# # ------------------------------------------- Health Check Endpoint ----------------------------------------
@server.get('/health')
def health_check():
    return {'status': 'ok', 'message': 'Model loaded successfully.'}


# ------------------------------------------- Generic Prediction Endpoint ----------------------------------------
@server.post('/predict')
def predict(input_data: DataFrameInput):
    try:
        df = DataFrame(data=input_data.dataframe_split['data'], columns=input_data.dataframe_split['columns'] )
        if not hasattr(model, "predict_proba"):
            return {"error": "Model does not support probability predictions."}

        # Predict class and probability
        proba = model.predict_proba(df)  # shape: (n_samples, n_classes)
        preds = argmax(proba, axis=1)    # convert probabilities to class labels

        # Stress class = 1 (assuming binary: 0 = no stress, 1 = stress)
        stress_probs = proba[:, 1]
        mean_stress_prob = float(mean(stress_probs))
        soft_voted_label = int(round(mean_stress_prob))  # threshold at 0.5

        # Questionnaire fusion
        questionnaire = input_data.questionnaire # type: ignore
        if questionnaire is not None:
            if soft_voted_label == 1 and questionnaire >= 1:
                stress_index = 2
            elif soft_voted_label == 1 or questionnaire >= 1:
                stress_index = 1
            else:
                stress_index = 0
        else:
            stress_index = soft_voted_label

        return {
            'model_predictions': preds.tolist(),
            'model_stress_probabilities': stress_probs.tolist(),
            'mean_stress_probability': round(mean_stress_prob, 3),
            'soft_voted_label': soft_voted_label,
            'questionnaire': questionnaire,
            'final_stress_index': stress_index
        }

    except Exception as e:
        return {'error': str(e)}


# ------------------------------------------- Education Stress Compute Endpoint ----------------------------------------
@server.post("/stress/compute/education")
def education_stress_compute(payload: EducationComputeInput):
    # 1. Build DataFrame
    df = DataFrame(
        payload.dataframe_split["data"],
        columns=payload.dataframe_split["columns"]
    )

    # 2. Validate only the subset needed for physiological stress
    required_cols = [
        "hr_base",
        "hrv_base",
        "hr_session",
        "hrv_session",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Empty dataframe provided")

    row = df.iloc[0]

    # 3. Safe extraction helper
    def get_optional_float(value):
        return None if isna(value) else float(value)

    # 4. Use the same physiological stress computation as academics
    stress = compute_stress_academic_physio(
        hr_base=float(row["hr_base"]),
        hrv_base=float(row["hrv_base"]),
        hr_session=float(row["hr_session"]),
        hrv_session=get_optional_float(row["hrv_session"]),
    )

    # 5. Match academic endpoint response style
    hrv_present = not isna(row["hrv_session"])

    return {
        "stress": stress,
        "confidence": 0.7 if hrv_present else 0.4,
        "needs_review": not hrv_present,
    }

# ------------------------------------------- Education Academic Staff Stress Compute Endpoint ----------------------------------------
@server.post("/stress/compute/academic")
def stress_compute_academic_physio(payload: AcademicPhysioStressInput):

    stress = compute_stress_academic_physio(
        hr_base=payload.hr_base,
        hrv_base=payload.hrv_base,
        hr_session=payload.hr_session,
        hrv_session=payload.hrv_session,
    )

    return {
        "stress": stress,
        "confidence": 0.7 if payload.hrv_session is not None else 0.4,
        "needs_review": payload.hrv_session is None,
    }

# ------------------------------------------- Healthcare Stress Compute Endpoint ----------------------------------------
@server.post("/stress/compute/healthcare")
def stress_compute_healthcare(payload: HealthcareStressInput):

    df = DataFrame(
        payload.dataframe_split["data"],
        columns=payload.dataframe_split["columns"],
    )

    required_cols = [
        "hr_base",
        "hrv_base",
        "hr_shift",
        "hrv_shift",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Empty dataframe provided")

    row = df.iloc[0]

    def get_optional_float(value):
        return None if isna(value) else float(value)

    stress = compute_stress_academic_physio(
        hr_base=float(row["hr_base"]),
        hrv_base=float(row["hrv_base"]),
        hr_session=float(row["hr_shift"]),
        hrv_session=get_optional_float(row["hrv_shift"]),
    )

    hrv_present = not isna(row["hrv_shift"])

    return {
        "stress": stress,
        "confidence": 0.7 if hrv_present else 0.4,
        "needs_review": not hrv_present,
    }
