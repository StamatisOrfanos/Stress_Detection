from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from pandas import DataFrame
from numpy import argmax, mean
from joblib import load
from os import getenv, path
from deployment_utils import init_model_weights
from dotenv import load_dotenv

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
    dataframe_split: dict # {columns: [...], data: [...]}
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


