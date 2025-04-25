import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("rf_crs.pkl")
scaler = joblib.load("crop_scaler.pkl")

st.title("🌿 Crop Recommendation System")
st.write("Enter your farm conditions to get the best crop suggestion.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
P = st.number_input("Phosphorous (P)", min_value=5, max_value=145, value=50)
K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=50)
temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=45.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=60.0)
ph = st.number_input("pH", min_value=3.5, max_value=10.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0, value=100.0)

# Predict
if st.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_crop = str(prediction[0])
    st.success(f"✅ Best Crop to be grown: **{predicted_crop.capitalize()}**")

