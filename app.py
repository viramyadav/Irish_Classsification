from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

app = FastAPI(title="Iris Classification API", version="1.0.0")

class PredictRequest(BaseModel):
    sepal_length: float = Field(gt=0, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, description="Sepal width in cm")
    petal_length: float = Field(gt=0, description="Petal length in cm")
    petal_width: float = Field(gt=0, description="Petal width in cm")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris classification API"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/predict")
def predict(request: PredictRequest):
    if not MODEL_PATH.exists():
        return {
            "error": "Trained model not found. Run `python train.py` first."
        }

    model = joblib.load(MODEL_PATH)
    data = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width,
    ]])
    prediction = model.predict(data)[0]
    return {"prediction": str(prediction)}

