# streamlit_app.py
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load(open('rf_crs.pkl', 'rb'))

# Label mapping
label_dict = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut', 5: 'coffee',
    6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize',
    12: 'mango', 13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon', 16: 'orange',
    17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}

# App title
st.title("🌾 Crop Recommendation System")
st.markdown("Provide the soil and weather details:")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=90)
P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=42)
K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=43)
temperature = st.number_input("Temperature (°C)", value=20.87)
humidity = st.number_input("Humidity (%)", value=82.00)
ph = st.number_input("pH Level", value=6.5)
rainfall = st.number_input("Rainfall (mm)", value=202.93)

# Prepare input
input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

# Predict and display result
if st.button("Recommend Crop"):
    prediction = model.predict(input_data)[0]
    crop_name = label_dict.get(prediction, f"Unknown crop with label {prediction}")
    st.success(f"🌱 Best Crop to be grown: **{crop_name.capitalize()}**")

    # Show feature importance chart
    if hasattr(model, 'feature_importances_'):
        st.subheader("🔍 Feature Importance")
        importances = model.feature_importances_
        features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        fig, ax = plt.subplots()
        ax.barh(features, importances, color='green')
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
