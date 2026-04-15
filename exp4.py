#   EXPERIMENT 4 — API Key Auth

import os
import numpy as np
import joblib

from fastapi import FastAPI, Request, HTTPException, status, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from datetime import datetime

API_KEY = os.getenv("API_KEY", "mysecretkey123")
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key"
    )

model = joblib.load("random_forest_most_optimized.pkl")

app = FastAPI(title="Heart Disease Prediction API — Exp 4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalch: float
    exang: int
    oldpeak: float

@app.get("/")
def root():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: PredictRequest, api_key: str = Security(get_api_key)):
    input_data = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalch, data.exang, data.oldpeak
    ]])
    prediction = model.predict(input_data)[0]
    return {
        "prediction": int(prediction),
        "result": "Disease Detected" if prediction == 1 else "No Disease",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }