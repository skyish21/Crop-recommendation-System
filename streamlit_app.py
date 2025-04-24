import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('rf_crs.pkl')

# Label mapping
label_dict = {
    'apple': 0, 'banana': 1, 'blackgram': 2, 'chickpea': 3, 'coconut': 4, 'coffee': 5,
    'cotton': 6, 'grapes': 7, 'jute': 8, 'kidneybeans': 9, 'lentil': 10, 'maize': 11,
    'mango': 12, 'mothbeans': 13, 'mungbean': 14, 'muskmelon': 15, 'orange': 16,
    'papaya': 17, 'pigeonpeas': 18, 'pomegranate': 19, 'rice': 20, 'watermelon': 21
}

# Invert the dictionary for prediction output
inv_label_dict = {v: k for k, v in label_dict.items()}

# App title
st.title("🌾 Crop Recommendation System")

st.markdown("Provide the soil and weather details:")

# Input sliders/fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=90)
P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=42)
K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=43)
temperature = st.number_input("Temperature (°C)", value=20.87)
humidity = st.number_input("Humidity (%)", value=82.00)
ph = st.number_input("pH Level", value=6.5)
rainfall = st.number_input("Rainfall (mm)", value=202.93)

# Prediction
if st.button("Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Scale the input data for prediction using the scaler
    scaled_data = scaler.transform(input_data)

    # Predict the crop using the trained model
    prediction = model.predict(scaled_data)[0]
    
    # Map the prediction back to the crop name
    crop_name = inv_label_dict.get(prediction, f"Unknown crop with label {prediction}")
    
    st.success(f"🌱 Best Crop to be grown: **{crop_name.capitalize()}**")
