import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the saved scaler and model
try:
    scaler = joblib.load('crop_scaler.pkl')
    model = joblib.load('rf_crs.pkl')
    st.sidebar.success("Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Label mapping
label_dict = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut', 5: 'coffee',
    6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize',
    12: 'mango', 13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon', 16: 'orange',
    17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}

# App title
st.title("🌾 Crop Recommendation System")

# Input fields
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42)
    K = st.number_input("Potassium (K)", min_value=0, max_value=300, value=43)
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.87)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.00)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=202.93)

# Prediction
if st.button("Recommend Crop", type="primary"):
    # Create input data
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Show raw input
    st.write("Input values:", input_data)
    
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Show scaled input
    st.write("Scaled input:", scaled_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0]
    st.write("Raw prediction (number):", prediction)
    
    # Convert prediction to crop name
    crop_name = label_dict.get(prediction, f"Unknown crop with label {prediction}")
    
    # Display result
    st.success(f"Best Crop to be grown: {crop_name.capitalize()}")
    
    # Show probabilities if available
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(scaled_data)[0]
        top_3 = np.argsort(probs)[-3:][::-1]
        
        st.write("Top predictions:")
        for i in top_3:
            st.write(f"{label_dict.get(i, i).capitalize()}: {probs[i]*100:.2f}%")