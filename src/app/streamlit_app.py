import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from model import predict_diseases
from utils import (
    validate_clinical_measurements,
    calculate_risk_score,
    get_measurement_trend,
    generate_health_recommendations
)

st.set_page_config(page_title="Multi-Disease Risk Predictor", layout="wide")

def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    fig.update_layout(height=200)
    return fig

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
        
    with col2:
        st.subheader("Cardiovascular Measurements")
        systolic = st.number_input("Systolic BP", 80, 200, 120)
        diastolic = st.number_input("Diastolic BP", 40, 130, 80)
        heart_rate = st.number_input("Heart Rate", 40, 200, 75)
        
    with col3:
        st.subheader("Additional Measurements")
        glucose = st.number_input("Blood Glucose", 50, 500, 100)
        oxygen = st.number_input("Oxygen Saturation", 70, 100, 98)
        cholesterol = st.number_input("Total Cholesterol", 100, 400, 200)

    # Advanced measurements in expandable section
    with st.expander("Advanced Measurements"):
        col4, col5 = st.columns(2)
        
        with col4:
            wbc = st.number_input("WBC Count", 1000, 50000, 7500)
            crp = st.number_input("C-Reactive Protein", 0.0, 50.0, 1.0)
            
        with col5:
            fev1 = st.number_input("FEV1", 0.0, 6.0, 3.0)
            respiratory_rate = st.number_input("Respiratory Rate", 8, 40, 16)

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
        "Respiratory_Rate": respiratory_rate
    }

    if st.button("Predict Disease Risks"):
        # Get predictions
        predictions = predict_diseases(patient_data)
        
        # Validate measurements
        warnings = validate_clinical_measurements(patient_data)
        if warnings:
            st.warning("Medical Attention Advised:")
            for warning in warnings.values():
                st.write(f"⚠️ {warning}")
        
        # Display predictions
        st.header("Disease Risk Predictions")
        
        # Create columns for each disease category
        cols = st.columns(len(predictions))
        
        for idx, (category, pred) in enumerate(predictions.items()):
            with cols[idx]:
                st.subheader(category.title())
                
                # Create and display gauge chart
                fig = create_gauge_chart(
                    pred['probability'],
                    f"{category.title()} Risk"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display risk score
                risk_score = calculate_risk_score(patient_data, category)
                st.metric(
                    "Risk Score", 
                    f"{risk_score:.1f}/100"
                )
        
        # Generate and display recommendations
        st.header("Health Recommendations")
        recommendations = generate_health_recommendations(patient_data, predictions)
        for rec in recommendations:
            st.write(f"✔️ {rec}")
            
        # Display important features
        st.header("Key Contributing Factors")
        for category, pred in predictions.items():
            if pred['probability'] > 30:  # Only show for significant risks
                st.subheader(f"{category.title()} Factors:")
                for feature in pred['features_used'][:3]:  # Show top 3 features
                    st.write(f"- {feature.replace('_', ' ').title()}")

if __name__ == "__main__":
    main()

