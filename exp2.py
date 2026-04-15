#   EXPERIMENT 2 — FastAPI Backend for Model Inference

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load(r"C:\Users\shuvt\Downloads\MLops\MLops\random_forest_most_optimized.pkl")

app = FastAPI(title="Heart Disease Prediction API")

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

class PredictResponse(BaseModel):
    prediction: int
    result: str

@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API is running!"}

@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    input_data = np.array([[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalch,
        data.exang,
        data.oldpeak
    ]])

    prediction = model.predict(input_data)[0]
    result = "Disease Detected" if prediction == 1 else "No Disease"

    return PredictResponse(prediction=int(prediction), result=result)