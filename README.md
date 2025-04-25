# 🌱 Crop Recommendation System

This project predicts the optimal crop to be grown based on various environmental and soil parameters using multiple machine learning models. The system can suggest the most suitable crop depending on the current conditions of the soil and environment.

## 📁 Dataset Overview

The dataset used in this project contains various features that represent environmental and soil conditions. The goal is to predict the best crop to grow based on these conditions.

### **Features:**
- **N**: Nitrogen content in the soil (measured in kg/ha)
- **P**: Phosphorus content in the soil (measured in kg/ha)
- **K**: Potassium content in the soil (measured in kg/ha)
- **temperature**: Ambient temperature (°C)
- **humidity**: Relative humidity (%) of the air
- **ph**: pH value of the soil
- **rainfall**: Annual rainfall (mm)

### **Label:**
- **label**: The recommended crop based on the input features. The label consists of various crop names such as rice, maize, wheat, etc. This is the target variable that the machine learning models aim to predict.

## 🧠 Algorithms & Evaluation

The following machine learning algorithms were implemented and evaluated to predict the optimal crop:

| Algorithm                  | Accuracy | F1 Score | Precision | Recall |
|----------------------------|----------|----------|-----------|--------|
| **Naive Bayes (nb)**        | 0.9945   | 1.00     | 1.00      | 1.00   |
| **Decision Tree (dt)**      | 0.9836   | 0.98     | 0.99      | 0.98   |
| **Support Vector Machine (svm)** | 0.9891 | 0.99 | 0.99     | 0.99   |
| **Random Forest (rf)**      | 1.00     | 1.00     | 1.00      | 1.00   |

These models were evaluated using metrics such as accuracy, precision, recall, and F1-score to assess their performance. The **Random Forest model** achieved the highest performance, with perfect accuracy and evaluation scores across all classes.

### **Key Metrics:**
- **Accuracy**: The percentage of correct predictions made by the model.
- **Precision**: The proportion of positive predictions that are actually correct.
- **Recall**: The proportion of actual positives that were correctly identified by the model.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.

## 🚀 Deployment

### **Platform:** Streamlit

A simple web-based interface for users to input the relevant soil and environmental parameters and get crop predictions in real-time.

### **Frontend:**
- A user-friendly UI where users can enter soil/environment parameters such as nitrogen content, temperature, humidity, etc.
- The UI is designed to provide instant feedback on the recommended crop based on the inputs.

### **Backend:**
- The backend uses machine learning models built with Scikit-learn to perform the prediction based on the input data.

### **Hosting:**
- The app can be hosted locally or on a cloud platform like **Streamlit Sharing**, allowing users to access the prediction tool from anywhere.

### **Edge Deployment:**
- Currently, edge deployment is not implemented, but future work may include deploying the model on edge devices (e.g., Raspberry Pi) for real-time recommendations in the field.

## 📦 Requirements

To run the project and its dependencies, you can install them using the following command:

```bash
pip install -r requirements.txt
```

## 🚧 Future Improvements
**Model Enhancement**: Experiment with other advanced models like XGBoost or Neural Networks for even better performance.

**Edge Deployment**: Implement deployment on edge devices (e.g., Raspberry Pi) for real-time predictions in agricultural fields.

**Expanded Dataset**: Use more diverse environmental and soil data to further improve crop prediction accuracy.

**Real-time Weather Integration**: Incorporate real-time weather forecasts into the crop recommendation process.
