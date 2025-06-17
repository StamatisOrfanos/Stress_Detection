from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Load model from unzipped path
MODEL_PATH = os.path.join("model_weights", "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

server = FastAPI(title="Stress Detector API")

# Input schema
class DataFrameInput(BaseModel):
    dataframe_split: dict  # {columns: [...], data: [...]}


# Health check route
@server.get("/health")
def health_check():
    return {"status": "ok", "message": "Model loaded successfully."}


# Prediction route
@server.post("/predict")
def predict(input_data: DataFrameInput):
    try:
        df = pd.DataFrame(
            data=input_data.dataframe_split["data"],
            columns=input_data.dataframe_split["columns"]
        )
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}
