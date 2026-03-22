from fastapi import FastAPI
import os
import joblib
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="ML Model API")

MODEL_PATH = "model/model.pkl"

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Make sure training ran successfully.")

model = joblib.load(MODEL_PATH)


# Input schema
class InputData(BaseModel):
    age: float
    experience: float
    education_level: float
    company_size: float
    role_level: float
    location_tier: float
    skills_score: float


@app.get("/")
def home():
    return {"message": "ML Model API is running successfully"}


@app.post("/predict")
def predict(data: InputData):
    try:
        input_array = np.array([[
            data.age,
            data.experience,
            data.education_level,
            data.company_size,
            data.role_level,
            data.location_tier,
            data.skills_score
        ]])

        prediction = model.predict(input_array)

        return {
            "input": data.dict(),
            "predicted_salary": float(prediction[0])
        }

    except Exception as e:
        return {"error": str(e)}