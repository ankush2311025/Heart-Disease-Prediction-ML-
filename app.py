import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# Load your trained model (you may need to pickle it first)
# For now, assume you're training inside the script
heart_data = pd.read_csv("heart.csv")
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

model = LogisticRegression(max_iter=1000)
model.fit(X, Y)

st.title("❤️ Heart Disease Prediction App")

# User input fields
age = st.number_input("Age", 20, 100, 30)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.number_input("Resting ECG (0-2)", 0, 2, 0)
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.number_input("Slope (0-2)", 0, 2, 1)
ca = st.number_input("Number of Major Vessels (0-4)", 0, 4, 0)
thal = st.number_input("Thal (0-3)", 0, 3, 2)

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ The person has Heart Disease")
    else:
        st.success("✅ The person does NOT have Heart Disease")
