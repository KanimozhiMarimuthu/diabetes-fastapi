import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("D:/diabetesprediction/diabetes.csv")

# Preprocess
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit App
st.set_page_config(page_title="Diabetes Prediction System")
st.title("🩺 Diabetes Prediction System")

import joblib
joblib.dump(model, 'model.pkl')

















# User Inputs
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose Level", 0, 200, 120)
blood_pressure = st.slider("Blood Pressure", 0, 140, 70)
skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 900, 80)
bmi = st.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.slider("Age", 10, 100, 30)

# Prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)
    st.success(f"Prediction: {'🟥 Diabetic' if result[0]==1 else '🟩 Not Diabetic'}")

import os
print(os.getcwd())

