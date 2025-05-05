import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import pickle

# Add the directory containing this file to Python path to import neighboring modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import predict_diseases
from utils import (
    validate_clinical_measurements,
    calculate_risk_score,
    get_measurement_trend,
    generate_health_recommendations
)

# Add debugging function to check threshold values
def debug_patient_values(patient_data):
    """Helpful debug function to check if patient data would trigger disease conditions"""
    debug_info = {}
    # Check flu thresholds
    debug_info['temp_risk'] = patient_data['Body_Temperature'] > 38.0
    debug_info['wbc_risk'] = patient_data['WBC_Count'] > 11000
    debug_info['flu_risk'] = debug_info['temp_risk'] or debug_info['wbc_risk']
    
    # Add other disease threshold checks as needed
    return debug_info

st.set_page_config(page_title="Multi-Disease Risk Predictor", layout="wide")

# Set custom theme and styling
st.markdown("""
<style>
    .recommendation-item {
        background-color: #1E3045;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 4px solid #4CAF50;
    }
    .stExpander {
        border: none !important;
        box-shadow: none !important;
    }
    .metric-card {
        background-color: #f1f8ff;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-weight: bold;
        color: #2C3E50;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 18px;
        color: #1E3045;
    }
</style>
""", unsafe_allow_html=True)

# Model performance metrics (would typically come from model evaluation)
MODEL_METRICS = {
    "cardiovascular": {
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.87,
        "f1_score": 0.88,
        "auc": 0.94
    },
    "cardiovascular-heart-attack": {
        "accuracy": 0.91,
        "precision": 0.88,
        "recall": 0.85,
        "f1_score": 0.87,
        "auc": 0.93
    },
    "cardiovascular-hypertension": {
        "accuracy": 0.93,
        "precision": 0.90,
        "recall": 0.89,
        "f1_score": 0.90,
        "auc": 0.95
    },
    "infectious": {
        "accuracy": 0.89,
        "precision": 0.85,
        "recall": 0.82,
        "f1_score": 0.83,
        "auc": 0.91
    },
    "infectious-flu": {
        "accuracy": 0.88,
        "precision": 0.84,
        "recall": 0.83,
        "f1_score": 0.84,
        "auc": 0.90
    },
    "infectious-hepatitis": {
        "accuracy": 0.94,
        "precision": 0.92,
        "recall": 0.89,
        "f1_score": 0.91,
        "auc": 0.96
    },
    "infectious-influenza": {
        "accuracy": 0.91,
        "precision": 0.88,
        "recall": 0.86,
        "f1_score": 0.87,
        "auc": 0.92
    },
    "metabolic": {
        "accuracy": 0.90,
        "precision": 0.87,
        "recall": 0.84,
        "f1_score": 0.86,
        "auc": 0.92
    },
    "metabolic-diabetes": {
        "accuracy": 0.91,
        "precision": 0.89,
        "recall": 0.86,
        "f1_score": 0.88,
        "auc": 0.94
    },
    "metabolic-obesity": {
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.91,
        "f1_score": 0.92,
        "auc": 0.97
    },
    "oncology-breast-cancer": {
        "accuracy": 0.93,
        "precision": 0.91,
        "recall": 0.88,
        "f1_score": 0.90,
        "auc": 0.95
    },
    "oncology-colon-cancer": {
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.87,
        "f1_score": 0.88,
        "auc": 0.94
    },
    "oncology-lung-cancer": {
        "accuracy": 0.91,
        "precision": 0.88,
        "recall": 0.86,
        "f1_score": 0.87,
        "auc": 0.93
    },
    "respiratory": {
        "accuracy": 0.91,
        "precision": 0.88,
        "recall": 0.86,
        "f1_score": 0.87,
        "auc": 0.93
    },
    "respiratory-asthma": {
        "accuracy": 0.93,
        "precision": 0.91,
        "recall": 0.89,
        "f1_score": 0.90,
        "auc": 0.95
    },
    "respiratory-copd": {
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.87,
        "f1_score": 0.88,
        "auc": 0.94
    }
}

def main():
    st.title("Multi-Disease Risk Prediction System")
    st.write("Enter patient measurements to predict disease risks")

    # Create three columns for input organization
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Basic Information")
        age = st.number_input("Age", 0, 120, 50)
        gender = st.selectbox("Gender", ["M", "F"])
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
        family_history = st.checkbox("Family History of Disease")
        
    with col2:
        st.subheader("Cardiovascular Measurements")
        systolic = st.number_input("Systolic BP", 80, 200, 120)
        diastolic = st.number_input("Diastolic BP", 40, 130, 80)
        heart_rate = st.number_input("Heart Rate", 40, 200, 75)
        cholesterol = st.number_input("Total Cholesterol", 100, 400, 200)
        
    with col3:
        st.subheader("Additional Measurements")
        glucose = st.number_input("Blood Glucose", 50, 500, 100)
        oxygen = st.number_input("Oxygen Saturation", 70, 100, 98)
        smoking_status = st.selectbox("Smoking Status", ["Non-smoker", "Former smoker", "Current smoker"])
        body_temp = st.number_input("Body Temperature (°C)", 35.0, 42.0, 37.0)

    # Advanced measurements in expandable section
    with st.expander("Advanced Measurements"):
        col4, col5 = st.columns(2)
        
        with col4:
            wbc = st.number_input("WBC Count", 1000, 50000, 7500)
            crp = st.number_input("C-Reactive Protein", 0.0, 50.0, 1.0)
            liver_enzymes = st.number_input("Liver Enzymes (ALT)", 5, 200, 30)
            
        with col5:
            fev1 = st.number_input("FEV1", 0.0, 6.0, 3.0)
            respiratory_rate = st.number_input("Respiratory Rate", 8, 40, 16)
            waist_circumference = st.number_input("Waist Circumference (cm)", 50, 200, 90)

    # Create patient data dictionary
    patient_data = {
        "AGE": age,
        "GENDER_M": 1 if gender == "M" else 0,
        "BMI": bmi,
        "Systolic_BP": systolic,
        "Diastolic_BP": diastolic,
        "Heart_Rate": heart_rate,
        "Blood_Glucose": glucose,
        "Oxygen_Saturation": oxygen,
        "Cholesterol_Total": cholesterol,
        "WBC_Count": wbc,
        "CRP": crp,
        "FEV1": fev1,
        "Respiratory_Rate": respiratory_rate,
        "Waist_Circumference": waist_circumference,
        "Family_History": 1 if family_history else 0,
        "Smoking_Status": 0 if smoking_status == "Non-smoker" else (1 if smoking_status == "Former smoker" else 2),
        "Body_Temperature": body_temp,
        "Liver_Enzymes": liver_enzymes
    }

    # Ensure all disease categories are displayed in tabs
    if st.button("Predict Disease Risks"):
        # Get predictions
        predictions = predict_diseases(patient_data)

        # Validate measurements
        warnings = validate_clinical_measurements(patient_data)
        if warnings:
            st.warning("Medical Attention Advised:")
            for warning in warnings.values():
                st.write(f"⚠️ {warning}")

        # Display predictions by category
        st.header("Disease Risk Predictions")

        # Create tabs for each disease category
        if predictions:
            tabs = st.tabs([category.title() for category in predictions.keys()])

            # Display predictions for each category in tabs
            for i, (category, category_diseases) in enumerate(predictions.items()):
                with tabs[i]:
                    st.subheader(f"{category.title()} Disease Risks")

                    # Display each disease in its own card
                    for disease_name, disease_data in category_diseases.items():
                        st.markdown(f"### {disease_name.replace('_', ' ')}")
                        st.markdown(f"**Risk Probability:** {disease_data['probability']:.1f}%")
                        st.markdown("**Key Features:**")
                        for feature in disease_data['features_used']:
                            st.markdown(f"- {feature}")
                        st.markdown("**Recommended Actions:**")
                        for action in disease_data['actions']:
                            st.markdown(f"- {action}")

if __name__ == "__main__":
    main()

