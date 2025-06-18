from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from deployment_utils import init_model_weights

# Step 1: Download model weights at runtime if needed
MODEL_URL = os.getenv('URL')  # Make sure this is passed in via docker-compose
init_model_weights(MODEL_URL) # type: ignore

# Step 2: Load model
MODEL_PATH = os.path.join('model_weights', 'model.pkl')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f'Model not found at {MODEL_PATH}')
model = joblib.load(MODEL_PATH)

# Step 3: Start API
server = FastAPI(title='Stress Detector API')

class DataFrameInput(BaseModel):
    dataframe_split: dict  # {columns: [...], data: [...]}

@server.get('/health')
def health_check():
    return {'status': 'ok', 'message': 'Model loaded successfully.'}

@server.post('/predict')
def predict(input_data: DataFrameInput):
    try:
        df = pd.DataFrame(
            data=input_data.dataframe_split['data'],
            columns=input_data.dataframe_split['columns']
        )
        preds = model.predict(df)
        return {'predictions': preds.tolist()}
    except Exception as e:
        return {'error': str(e)}
