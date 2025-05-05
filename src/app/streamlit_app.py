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
    .summary-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #3498db;
    }
    .high-risk-card {
        background-color: #fff8f8;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-left: 5px solid #e74c3c;
    }
    .medium-risk-card {
        background-color: #fff8f0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-left: 5px solid #f39c12;
    }
    .action-item {
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 5px;
        background-color: #eaf2f8;
        border-left: 3px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

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
        tabs = st.tabs([category.title() for category in predictions.keys()])
        
        # Display predictions for each category in tabs
        for i, (category, category_diseases) in enumerate(predictions.items()):
            with tabs[i]:
                # Create a container for disease cards
                st.subheader(f"{category.title()} Disease Risks")
                
                # Calculate number of columns based on number of diseases
                num_diseases = len(category_diseases)
                cols = st.columns(min(3, num_diseases))
                
                # Display each disease in its own card
                for j, (disease_name, disease_data) in enumerate(category_diseases.items()):
                    col_idx = j % min(3, num_diseases)
                    with cols[col_idx]:
                        # Create styled card for disease
                        st.markdown("---")
                        st.markdown(f"### {disease_name.replace('_', ' ')}")
                        
                        # Display risk probability as a number with appropriate color
                        risk_prob = disease_data['probability']
                        risk_level = "Low"
                        risk_color = "green"
                        
                        if risk_prob > 70:
                            risk_level = "High"
                            risk_color = "red"
                        elif risk_prob > 30:
                            risk_level = "Medium"
                            risk_color = "orange"
                            
                        # Display risk percentage with colored text
                        st.markdown(f"#### Risk: <span style='color:{risk_color};font-size:24px;'>{risk_prob:.1f}%</span>", unsafe_allow_html=True)
                        st.markdown(f"**Level: {risk_level}**")
                        
                        # Show key contributing factors
                        st.markdown("#### Key Factors:")
                        for feature in disease_data['features_used'][:3]:  # Top 3 features
                            feature_name = feature.replace('_', ' ').title()
                            st.markdown(f"- {feature_name}")
                            
                        # Display recommended actions
                        st.markdown("#### Recommended Actions:")
                        for action in disease_data['actions'][:3]:  # Top 3 actions
                            st.markdown(f"✓ {action}")
        
        # Enhanced Health Summary Section
        st.header("Health Summary")
        
        # Create containers for high/medium/low risk conditions
        high_risk_diseases = []
        medium_risk_diseases = []
        low_risk_diseases = []
        
        # Categorize diseases by risk level
        for category, diseases in predictions.items():
            for disease, data in diseases.items():
                disease_name = disease.replace('_', ' ')
                if data['probability'] > 70:
                    high_risk_diseases.append((disease_name, data['probability'], data['actions']))
                elif data['probability'] > 30:
                    medium_risk_diseases.append((disease_name, data['probability'], data['actions']))
                else:
                    low_risk_diseases.append((disease_name, data['probability'], data['actions']))
        
        # Calculate overall health score
        total_conditions = len(high_risk_diseases) + len(medium_risk_diseases) + len(low_risk_diseases)
        health_score = 100 - (len(high_risk_diseases) * 15 + len(medium_risk_diseases) * 5)
        health_score = max(10, min(100, health_score))  # Ensure score is between 10 and 100
        
        # Create overall health summary card
        st.markdown(f"""
        <div class="summary-card">
            <h3>Overall Health Assessment</h3>
            <p>Based on your measurements and predicted disease risks:</p>
            <ul>
                <li><strong>Health Score:</strong> {health_score}/100</li>
                <li><strong>High Risk Conditions:</strong> {len(high_risk_diseases)}</li>
                <li><strong>Medium Risk Conditions:</strong> {len(medium_risk_diseases)}</li>
                <li><strong>Low Risk Conditions:</strong> {len(low_risk_diseases)}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display prioritized health concerns
        if high_risk_diseases:
            st.markdown("### Urgent Health Concerns")
            st.markdown("These conditions require immediate attention and potential medical consultation:")
            
            for disease_name, prob, actions in high_risk_diseases:
                st.markdown(f"""
                <div class="high-risk-card">
                    <h4>⚠️ {disease_name} - {prob:.1f}% risk</h4>
                    <p><strong>Key Actions:</strong></p>
                    <ul>
                        {"".join([f"<li>{action}</li>" for action in actions[:3]])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        if medium_risk_diseases:
            st.markdown("### Moderate Health Concerns")
            st.markdown("These conditions should be monitored and preventive steps taken:")
            
            # Use columns to display medium risk conditions
            cols = st.columns(2)
            for idx, (disease_name, prob, actions) in enumerate(medium_risk_diseases):
                with cols[idx % 2]:
                    st.markdown(f"""
                    <div class="medium-risk-card">
                        <h4>⚠️ {disease_name} - {prob:.1f}% risk</h4>
                        <p><strong>Key Actions:</strong></p>
                        <ul>
                            {"".join([f"<li>{action}</li>" for action in actions[:2]])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Compile top priority personalized actions
        st.markdown("### Top Priority Health Actions")
        
        # Collect all actions from high and medium risk diseases
        all_actions = []
        for _, _, actions in high_risk_diseases:
            all_actions.extend([(action, 2) for action in actions])  # Weight 2 for high risk
            
        for _, _, actions in medium_risk_diseases:
            all_actions.extend([(action, 1) for action in actions])  # Weight 1 for medium risk
        
        # Count action frequency weighted by risk
        action_weights = {}
        for action, weight in all_actions:
            if action in action_weights:
                action_weights[action] += weight
            else:
                action_weights[action] = weight
        
        # Sort actions by weight and display top ones
        top_actions = sorted(action_weights.items(), key=lambda x: x[1], reverse=True)
        
        if top_actions:
            col1, col2 = st.columns(2)
            
            with col1:
                for action, _ in top_actions[:min(5, len(top_actions))]:
                    st.markdown(f"""
                    <div class="action-item">
                        <strong>✓</strong> {action}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                for action, _ in top_actions[min(5, len(top_actions)):min(10, len(top_actions))]:
                    st.markdown(f"""
                    <div class="action-item">
                        <strong>✓</strong> {action}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No specific health actions identified. Maintain your healthy lifestyle!")
        
        # Personalized advice based on measurements
        st.markdown("### Personalized Health Advice")
        
        # Age-specific advice
        if patient_data["AGE"] > 60:
            st.markdown("""
            <div class="action-item">
                <strong>Age-related:</strong> Consider bone density screening and regular hearing/vision checks.
            </div>
            """, unsafe_allow_html=True)
        
        # Blood pressure advice
        if patient_data["Systolic_BP"] > 130 or patient_data["Diastolic_BP"] > 85:
            st.markdown("""
            <div class="action-item">
                <strong>Blood Pressure:</strong> Monitor your blood pressure regularly and consider reducing sodium intake.
            </div>
            """, unsafe_allow_html=True)
        
        # BMI advice
        if patient_data["BMI"] > 30:
            st.markdown("""
            <div class="action-item">
                <strong>Weight Management:</strong> Consider consulting with a nutritionist for a personalized weight management plan.
            </div>
            """, unsafe_allow_html=True)
        
        # Smoking advice
        if patient_data["Smoking_Status"] == 2:
            st.markdown("""
            <div class="action-item">
                <strong>Smoking:</strong> Quitting smoking is one of the most impactful steps you can take for your health. Consider smoking cessation programs.
            </div>
            """, unsafe_allow_html=True)
        
        # Glucose advice
        if patient_data["Blood_Glucose"] > 100:
            st.markdown("""
            <div class="action-item">
                <strong>Blood Sugar:</strong> Your blood glucose is elevated. Consider reducing simple carbohydrates and increasing physical activity.
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

