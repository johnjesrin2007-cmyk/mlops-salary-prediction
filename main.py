from fastapi import FastAPI
import os
import joblib
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="ML Model API")

# 🔥 Get absolute base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔥 Correct model path
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

# 🔥 Load model safely
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")
else:
    print(f"❌ Model not found at: {MODEL_PATH}")


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

    # 🔥 Handle missing model gracefully
    if model is None:
        return {"error": "Model not loaded. Training might have failed."}

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