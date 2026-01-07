from os import path
from typing import Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from pandas import DataFrame
from numpy import argmax, mean
from joblib import load
from os import getenv, path
from deployment_utils import init_model_weights
from dotenv import load_dotenv
from src.input import StressComputeInput
from education_stress_rules import education_compute_stress, compute_confidence

# Step 1: Download model weights at runtime if needed
load_dotenv()  # Load environment variables from .env fil
MODEL_URL = getenv('URL')  # Make sure this is passed in via docker-compose
init_model_weights(MODEL_URL) # type: ignore

# Step 2: Load model
MODEL_PATH = path.join('model_weights', 'model.pkl')
if not path.exists(MODEL_PATH):
    raise FileNotFoundError(f'Model not found at {MODEL_PATH}')
model = load(MODEL_PATH)

# Step 3: Start API
server = FastAPI(title='Stress Detector API')

class DataFrameInput(BaseModel):
    dataframe_split: dict
    questionnaire: Optional[int] = None
    

@server.get('/health')
def health_check():
    return {'status': 'ok', 'message': 'Model loaded successfully.'}


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
def education_stress_compute(payload: StressComputeInput):

    df = DataFrame(data=payload.dataframe_split["data"], columns=payload.dataframe_split["columns"],)
    model_prob = None
    model_used = False

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        model_prob = float(mean(proba[:, 1]))
        model_used = True

    # Core computation
    result = education_compute_stress(
        hr_base=payload.hr_base,
        hrv_base=payload.hrv_base,
        steps_base=payload.steps_base,
        hr_session=payload.hr_session,
        hrv_session=payload.hrv_session,
        steps_session=payload.steps_session,
        sleep_last_24h=payload.sleep_last_24h,
        model_stress_prob=model_prob,
        pre_sr=payload.pre_sr,
        post_sr=payload.post_sr,
        weekly_sr=payload.weekly_sr,
        deadlines_72h=payload.deadlines_72h,
        exam_hours_until=payload.exam_hours_until,
        back_to_back_sessions=payload.back_to_back_sessions,
        credit_overload=payload.credit_overload,
        work_hours_week=payload.work_hours_week,
        commute_minutes_day=payload.commute_minutes_day,
    )

    # Confidence
    confidence = compute_confidence(
        hrv_session=payload.hrv_session,
        hr_missing_ratio=0.0,
        model_used=model_used,
        post_sr_present=payload.post_sr is not None,
        sleep_present=payload.sleep_last_24h is not None,
    )

    # Review flag
    needs_review = confidence < 0.6

    return {
        "stress": result,
        "confidence": round(confidence, 2),
        "needs_review": needs_review,
    }
