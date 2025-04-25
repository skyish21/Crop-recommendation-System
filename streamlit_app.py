import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the model and scaler
try:
    scaler = joblib.load('crop_scaler.pkl')
    model = joblib.load('rf_crs.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# App title
st.title("🌾 Crop Recommendation System")

# Create inputs
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42)
K = st.number_input("Potassium (K)", min_value=0, max_value=300, value=43)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.87)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.00)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=202.93)

# Recommendation button
if st.button("Recommend Crop"):
    # Create input array
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Scale input
    scaled_data = scaler.transform(input_data)
    
    # Get raw prediction
    raw_prediction = model.predict(scaled_data)[0]
    
    # Get the actual class labels from the model
    if hasattr(model, 'classes_'):
        class_labels = model.classes_
        st.write("Model's internal class labels:", class_labels)
        
        # Map the prediction using model's own classes
        prediction_index = np.where(class_labels == raw_prediction)[0][0]
        
        # Define crop names in order they appear in your data
        crop_names = [
            'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
            'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
            'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange',
            'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
        ]
        
        # Use the model's own mapping
        if prediction_index < len(crop_names):
            crop_name = crop_names[prediction_index]
        else:
            crop_name = f"Unknown crop (index {prediction_index})"
    else:
        # Fallback if model doesn't have classes_
        st.write("Raw prediction value:", raw_prediction)
        
        # Try direct indexing based on numeric prediction
        crop_names = [
            'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
            'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
            'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange',
            'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
        ]
        
        if isinstance(raw_prediction, (int, np.integer)) and 0 <= raw_prediction < len(crop_names):
            crop_name = crop_names[raw_prediction]
        else:
            crop_name = "Unknown prediction format"
    
    # Display result
    st.success(f"Best Crop: {crop_name.capitalize()}")
    
    # Show probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(scaled_data)[0]
        st.write("Raw probability values:", probabilities)
        
        # Create a dataframe to display probabilities
        if hasattr(model, 'classes_'):
            prob_df = pd.DataFrame({
                'Crop Index': model.classes_,
                'Probability': probabilities
            })
        else:
            prob_df = pd.DataFrame({
                'Index': range(len(probabilities)),
                'Probability': probabilities
            })
        
        st.write("Probability breakdown:")
        st.dataframe(prob_df.sort_values('Probability', ascending=False))