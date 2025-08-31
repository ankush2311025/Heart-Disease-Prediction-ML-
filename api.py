from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Define input format
class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API"}

@app.post("/predict")
def predict(data: HeartData):
    input_data = np.array([
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]).reshape(1, -1)

    prediction = model.predict(input_data)
    result = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
    return {"prediction": result}
