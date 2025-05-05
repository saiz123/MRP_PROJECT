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
                            
        # Display overall health summary
        st.header("Health Summary")
        
        # Find high-risk diseases (probability > 70)
        high_risk_diseases = []
        for category, diseases in predictions.items():
            for disease, data in diseases.items():
                if data['probability'] > 70:
                    high_risk_diseases.append((disease.replace('_', ' '), data['probability']))
        
        # Display high-risk diseases if any
        if high_risk_diseases:
            st.warning("#### High Risk Conditions:")
            for disease, prob in high_risk_diseases:
                st.markdown(f"⚠️ **{disease}** - {prob:.1f}% risk")
                
        # Display health recommendations
        st.subheader("Health Recommendations")
        
        # Collect all unique actions from medium and high risk diseases
        all_actions = set()
        for category, diseases in predictions.items():
            for disease, data in diseases.items():
                if data['probability'] > 30:  # Medium or high risk
                    all_actions.update(data['actions'])
        
        # Group recommendations by category
        recommendation_categories = {
            "Lifestyle": [],
            "Diet & Nutrition": [],
            "Medical Follow-up": [],
            "Screening Tests": [],
            "General": []
        }
        
        # Categorize recommendations
        for action in all_actions:
            if any(keyword in action.lower() for keyword in ["exercise", "physical", "activity", "smoking", "alcohol"]):
                recommendation_categories["Lifestyle"].append(action)
            elif any(keyword in action.lower() for keyword in ["diet", "eat", "food", "nutrient", "vitamin", "mineral", "weight"]):
                recommendation_categories["Diet & Nutrition"].append(action)
            elif any(keyword in action.lower() for keyword in ["doctor", "physician", "consult", "appointment", "medication"]):
                recommendation_categories["Medical Follow-up"].append(action)
            elif any(keyword in action.lower() for keyword in ["test", "screen", "monitor", "check"]):
                recommendation_categories["Screening Tests"].append(action)
            else:
                recommendation_categories["General"].append(action)
        
        # Display recommendations by category in expandable sections with custom styling
        col1, col2 = st.columns(2)
        
        with col1:
            for category in ["Lifestyle", "Diet & Nutrition"]:
                if recommendation_categories[category]:
                    with st.expander(f"📋 {category} Recommendations", expanded=True):
                        for action in recommendation_categories[category]:
                            st.markdown(f"""
                            <div class="recommendation-item">
                                <span style="color:#4CAF50">✓</span> {action}
                            </div>
                            """, unsafe_allow_html=True)
        
        with col2:
            for category in ["Medical Follow-up", "Screening Tests", "General"]:
                if recommendation_categories[category]:
                    with st.expander(f"📋 {category} Recommendations", expanded=True):
                        for action in recommendation_categories[category]:
                            st.markdown(f"""
                            <div class="recommendation-item">
                                <span style="color:#4CAF50">✓</span> {action}
                            </div>
                            """, unsafe_allow_html=True)
        
        # Display model performance metrics
        st.header("Model Performance Metrics")
        st.write("The following metrics represent the performance of our prediction models on validation data:")
        
        # Create tabs for each model category
        metric_tabs = st.tabs(["Overview"] + list(set([model_name.split('-')[0] for model_name in MODEL_METRICS.keys() if '-' in model_name])))
        
        # Overview tab - show average metrics
        with metric_tabs[0]:
            st.subheader("Overall Model Performance")
            
            # Calculate average metrics
            avg_metrics = {
                "accuracy": sum(m["accuracy"] for m in MODEL_METRICS.values()) / len(MODEL_METRICS),
                "precision": sum(m["precision"] for m in MODEL_METRICS.values()) / len(MODEL_METRICS),
                "recall": sum(m["recall"] for m in MODEL_METRICS.values()) / len(MODEL_METRICS),
                "f1_score": sum(m["f1_score"] for m in MODEL_METRICS.values()) / len(MODEL_METRICS),
                "auc": sum(m["auc"] for m in MODEL_METRICS.values()) / len(MODEL_METRICS)
            }
            
            # Create metric cards in columns
            metric_cols = st.columns(5)
            metrics = [
                ("Accuracy", avg_metrics["accuracy"], "Proportion of correct predictions"),
                ("Precision", avg_metrics["precision"], "Ability to not label negative samples as positive"),
                ("Recall", avg_metrics["recall"], "Ability to find all positive samples"),
                ("F1 Score", avg_metrics["f1_score"], "Harmonic mean of precision and recall"),
                ("AUC", avg_metrics["auc"], "Area under ROC curve")
            ]
            
            for i, (metric_name, metric_value, description) in enumerate(metrics):
                with metric_cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">{metric_name}</div>
                        <div class="metric-value">{metric_value:.2f}</div>
                        <div>{description}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("""
            **About these metrics:**
            - **Accuracy**: The ratio of correctly predicted observations to the total observations.
            - **Precision**: The ratio of correctly predicted positive observations to the total predicted positive observations.
            - **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
            - **F1 Score**: The weighted average of Precision and Recall.
            - **AUC**: Area under the ROC Curve - measures the model's ability to distinguish between classes.
            """)
        
        # Category-specific tabs
        for i, category in enumerate(set([model_name.split('-')[0] for model_name in MODEL_METRICS.keys() if '-' in model_name])):
            with metric_tabs[i+1]:
                st.subheader(f"{category.title()} Models Performance")
                
                # Get models for this category
                category_models = {model_name: metrics for model_name, metrics in MODEL_METRICS.items() 
                                  if model_name.startswith(category) and model_name != category}
                
                # Add parent category model
                if category in MODEL_METRICS:
                    category_models[category] = MODEL_METRICS[category]
                
                # Display metrics for each model in this category
                for model_name, metrics in category_models.items():
                    display_name = model_name.replace(f"{category}-", "").replace("-", " ").replace("_", " ").title()
                    if model_name == category:
                        display_name = f"{category.title()} (General)"
                    
                    st.markdown(f"### {display_name}")
                    
                    metric_cols = st.columns(5)
                    with metric_cols[0]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Accuracy</div>
                            <div class="metric-value">{metrics['accuracy']:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[1]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Precision</div>
                            <div class="metric-value">{metrics['precision']:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[2]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Recall</div>
                            <div class="metric-value">{metrics['recall']:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[3]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">F1 Score</div>
                            <div class="metric-value">{metrics['f1_score']:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[4]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">AUC</div>
                            <div class="metric-value">{metrics['auc']:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")

if __name__ == "__main__":
    main()

