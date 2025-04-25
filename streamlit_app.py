import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the saved scaler and model separately
scaler = joblib.load('crop_scaler.pkl')
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
st.markdown("Provide the soil and weather details to get the best crop recommendation.")

# Create two columns for input parameters
col1, col2 = st.columns(2)

with col1:
    # Input fields - first column
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42)
    K = st.number_input("Potassium (K)", min_value=0, max_value=300, value=43)
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.87)

with col2:
    # Input fields - second column
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.00)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=202.93)

# Add some space
st.write("")

# Prediction
if st.button("Recommend Crop", type="primary"):
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Create input data
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Update progress
    progress_bar.progress(30)
    
    # Scale the input data for prediction
    scaled_data = scaler.transform(input_data)
    
    # Update progress
    progress_bar.progress(70)
    
    # Get prediction and probabilities
    prediction = model.predict(scaled_data)[0]
    crop_name = inv_label_dict.get(prediction, f"Unknown crop with label {prediction}")
    
    # If the model supports probability prediction, show top 3 recommendations
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(scaled_data)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_crops = []
        
        for idx in top_indices:
            if idx < len(model.classes_):
                crop_idx = model.classes_[idx]
                crop = inv_label_dict.get(crop_idx, f"Unknown ({crop_idx})")
                confidence = probabilities[idx] * 100
                top_crops.append((crop.capitalize(), confidence))
    
    # Complete progress
    progress_bar.progress(100)
    
    # Display results
    st.success(f"🌱 Best Crop to be grown: **{crop_name.capitalize()}**")
    
    # Display top recommendations if available
    if hasattr(model, 'predict_proba'):
        st.subheader("Top Recommendations:")
        for i, (crop, confidence) in enumerate(top_crops):
            st.write(f"{i+1}. {crop} - {confidence:.2f}% confidence")
    
    # Add crop-specific information
    st.subheader(f"About {crop_name.capitalize()}:")
    
    # Add some information about the predicted crop
    crop_info = {
        'apple': "Apples grow best in cool climates with moderate humidity and require well-drained soil.",
        'banana': "Bananas thrive in tropical conditions with high humidity and temperatures.",
        'blackgram': "Blackgram is a drought-resistant crop that grows well in warm climates.",
        'chickpea': "Chickpeas prefer cool, dry growing conditions and well-drained soils.",
        'coconut': "Coconuts need tropical climate with high humidity and plenty of sunlight.",
        'coffee': "Coffee requires subtropical climate with moderate rainfall and shade.",
        'cotton': "Cotton grows best in warm climates with moderate rainfall.",
        'grapes': "Grapes need sunny, warm days and cool nights with well-drained soil.",
        'jute': "Jute requires high temperature, high humidity and heavy rainfall.",
        'kidneybeans': "Kidney beans prefer warm temperatures and moderate rainfall.",
        'lentil': "Lentils grow best in cool, dry conditions with well-drained soil.",
        'maize': "Maize needs warm soil and sunny conditions for optimal growth.",
        'mango': "Mangoes require tropical climate with distinct wet and dry seasons.",
        'mothbeans': "Mothbeans are drought-tolerant and grow well in arid regions.",
        'mungbean': "Mungbeans thrive in warm, humid conditions with moderate rainfall.",
        'muskmelon': "Muskmelons need warm temperatures, plenty of sunlight, and moderate water.",
        'orange': "Oranges grow best in subtropical climate with mild winters.",
        'papaya': "Papayas need tropical conditions with no frost and plenty of sunlight.",
        'pigeonpeas': "Pigeonpeas are drought-resistant and grow well in semi-arid conditions.",
        'pomegranate': "Pomegranates prefer hot, dry summers and cool winters.",
        'rice': "Rice requires warm temperatures and abundant water supply.",
        'watermelon': "Watermelons need warm soil, hot days, cool nights, and steady moisture."
    }
    
    st.write(crop_info.get(crop_name, "Information not available for this crop."))
    
    # Display soil requirements
    st.subheader("Soil and Climate Requirements:")
    st.write(f"• Nitrogen (N): {N} - {'Optimal' if 70 <= N <= 110 else 'May need adjustment'}")
    st.write(f"• Phosphorus (P): {P} - {'Optimal' if 30 <= P <= 60 else 'May need adjustment'}")
    st.write(f"• Potassium (K): {K} - {'Optimal' if 30 <= K <= 60 else 'May need adjustment'}")
    st.write(f"• Temperature: {temperature}°C - {'Optimal' if 18 <= temperature <= 30 else 'May need adjustment'}")
    st.write(f"• Humidity: {humidity}% - {'Optimal' if 50 <= humidity <= 85 else 'May need adjustment'}")
    st.write(f"• pH Level: {ph} - {'Optimal' if 5.5 <= ph <= 7.5 else 'May need adjustment'}")
    st.write(f"• Rainfall: {rainfall}mm - {'Optimal' if 80 <= rainfall <= 250 else 'May need adjustment'}")