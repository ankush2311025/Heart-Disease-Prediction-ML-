# 🫀 Heart Disease Prediction Project

This project is a **Machine Learning-based Heart Disease Prediction System**.  
It has two parts:  

1. **Streamlit App** – A simple web interface to interact with the ML model.  
2. **FastAPI Backend** – An API that exposes prediction functionality for integration with other apps (e.g., Node.js backend, React frontend).  

---

## 🚀 Features
- Predicts likelihood of heart disease based on input features.  
- Exposes REST API for easy integration.  
- User-friendly Streamlit UI for direct usage.  
- Deployable on **Render** or **Streamlit Cloud**.  

---

## 🔗 Live Links
- **Streamlit App (UI):** [👉 Open Streamlit App](https://heartdiseaseprediction101.streamlit.app/)  
- **FastAPI Backend (API):** [👉 Open FastAPI API](https://heart-disease-prediction-ml-vzpw.onrender.com/)  

---

## 🛠️ Tech Stack
- **Python** (scikit-learn, pandas, numpy, pickle)  
- **Streamlit** (Frontend UI)  
- **FastAPI** (Backend API)  
- **Uvicorn** (Server)  

---

## 📂 Project Structure
Heart-Disease/
│── heart_disease.py # ML model training & saving as model.pkl
│── model.pkl # Saved ML model
│── api.py # FastAPI backend
│── app.py # Streamlit frontend
│── requirements.txt # Dependencies


2️⃣ API Endpoints
GET /
Health check – confirms API is running.

json
{"message": "Heart Disease Prediction API is running!"}
POST /predict
Send patient data to get prediction.
Example Request (JSON body):

json
{
  "age": 45,
  "sex": 1,
  "cp": 2,
  "trestbps": 130,
  "chol": 250,
  "fbs": 0,
  "restecg": 1,
  "thalach": 170,
  "exang": 0,
  "oldpeak": 1.2,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
Example Response:
{
  "prediction": "No Heart Disease"
}

🌍 Deployment
Streamlit: Deploy directly via Streamlit Cloud.

FastAPI: Deploy using Render.

