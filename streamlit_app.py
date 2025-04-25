import streamlit as st
import numpy as np
import joblib

# Load the saved scaler and model
scaler = joblib.load('crop_scaler.pkl')
model = joblib.load('svm_crs.pkl')

# The correct mapping based on model.classes_
# This maps the model's internal indices to the actual crop names
crop_mapping = {
    20: 'rice', 11: 'maize', 3: 'chickpea', 9: 'kidneybeans', 
    18: 'pigeonpeas', 13: 'mothbeans', 14: 'mungbean', 2: 'blackgram', 
    10: 'lentil', 19: 'pomegranate', 1: 'banana', 12: 'mango', 
    7: 'grapes', 21: 'watermelon', 15: 'muskmelon', 0: 'apple', 
    16: 'orange', 17: 'papaya', 4: 'coconut', 6: 'cotton', 
    8: 'jute', 5: 'coffee'
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
    
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Make prediction using the model's internal class index
    prediction_idx = model.predict(scaled_data)[0]
    
    # Get the actual crop label (number) from the model's classes_ attribute
    crop_label = model.classes_[prediction_idx]
    
    # Map the crop label to crop name using our mapping
    crop_name = crop_mapping.get(crop_label, f"Unknown crop with label {crop_label}")
    
    # Display result
    st.success(f"🌱 Best Crop to be grown: **{crop_name.capitalize()}**")
    
    # Show top recommendations if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(scaled_data)[0]
        
        # Create list of (crop_name, probability) tuples
        crop_probs = []
        for idx, prob in enumerate(probabilities):
            if idx < len(model.classes_):
                crop_label = model.classes_[idx]
                crop = crop_mapping.get(crop_label, f"Unknown ({crop_label})")
                crop_probs.append((crop, prob))
        
        # Sort by probability (descending)
        crop_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 3
        st.subheader("Top Recommendations:")
        for i, (crop, prob) in enumerate(crop_probs[:3]):
            st.write(f"{i+1}. {crop.capitalize()} - {prob*100:.2f}% confidence")
