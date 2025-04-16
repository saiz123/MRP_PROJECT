import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional

def rule_based_prediction(age, bmi, blood_pressure, cholesterol, smoker, diabetes, heart_rate):
    """
    Predict disease based on input parameters using rule-based logic.

    Parameters:
    - age: int
    - bmi: float
    - blood_pressure: float
    - cholesterol: str ("High" or "Normal")
    - smoker: str ("Yes" or "No")
    - diabetes: str ("Yes" or "No")
    - heart_rate: int

    Returns:
    - predicted_disease: str
    - probability: int
    - suggested_action: str
    """

    if cholesterol == "High" and blood_pressure > 130:
        predicted_disease = "Hypertension"
        probability = np.random.randint(75, 90)
        suggested_action = "Reduce salt intake, regular exercise, monitor BP"

    elif age > 45 and smoker == "Yes" and blood_pressure > 130:
        predicted_disease = "Heart Disease"
        probability = np.random.randint(75, 85)
        suggested_action = "Cardio tests, lifestyle changes, consult a cardiologist"

    elif bmi > 30 and diabetes == "Yes":
        predicted_disease = "Diabetes"
        probability = np.random.randint(70, 85)
        suggested_action = "Monitor blood sugar, healthy diet, medication"

    elif blood_pressure > 160 and heart_rate > 100:
        predicted_disease = "Hypertension"
        probability = np.random.randint(80, 90)
        suggested_action = "Consult a doctor for potential BP medications"

    elif cholesterol == "High" and age > 50 and heart_rate < 60:
        predicted_disease = "Heart Disease"
        probability = np.random.randint(70, 85)
        suggested_action = "Cardiac tests, monitor cholesterol, regular checkups"

    elif smoker == "Yes" and age > 40 and heart_rate > 100:
        predicted_disease = "Heart Disease"
        probability = np.random.randint(70, 80)
        suggested_action = "Quit smoking, regular exercise, heart screenings"

    elif diabetes == "Yes" and blood_pressure > 140:
        predicted_disease = "Diabetes"
        probability = np.random.randint(75, 85)
        suggested_action = "Manage blood sugar, healthy eating, medication"

    elif bmi > 35 and age > 50 and cholesterol == "High":
        predicted_disease = "Heart Disease"
        probability = np.random.randint(70, 80)
        suggested_action = "Lifestyle changes, consider medication for cholesterol, heart checkups"

    elif age > 60 and bmi < 18.5:
        predicted_disease = "No Disease"
        probability = 0
        suggested_action = "Maintain a balanced diet and healthy lifestyle"

    elif age < 20 and bmi >= 18.5 and cholesterol == "Normal":
        predicted_disease = "No Disease"
        probability = 0
        suggested_action = "Keep up the good health habits"

    else:
        predicted_disease = "No Disease"
        probability = 0
        suggested_action = "No Immediate Action"

    return predicted_disease, probability, suggested_action

def validate_clinical_measurements(data: Dict[str, float]) -> Dict[str, str]:
    """
    Validate clinical measurements and return warnings for out-of-range values
    """
    warnings = {}
    
    # Blood Pressure
    if data.get('Systolic_BP', 0) > 180:
        warnings['bp'] = "Critically high blood pressure"
    elif data.get('Systolic_BP', 0) > 140:
        warnings['bp'] = "High blood pressure"
        
    # Heart Rate
    if data.get('Heart_Rate', 0) > 120:
        warnings['heart_rate'] = "Elevated heart rate"
    elif data.get('Heart_Rate', 0) < 50:
        warnings['heart_rate'] = "Low heart rate"
        
    # Blood Glucose
    if data.get('Blood_Glucose', 0) > 200:
        warnings['glucose'] = "High blood glucose"
        
    # Temperature
    if data.get('Body_Temperature', 0) > 38:
        warnings['temperature'] = "Fever detected"
        
    # Oxygen Saturation
    if data.get('Oxygen_Saturation', 0) < 95:
        warnings['oxygen'] = "Low oxygen saturation"
        
    return warnings

def calculate_risk_score(measurements: Dict[str, float], category: str) -> float:
    """
    Calculate a risk score (0-100) for a specific disease category based on measurements
    """
    risk_score = 0
    
    if category == "CARDIOVASCULAR":
        # Blood pressure contribution (30%)
        systolic = measurements.get('Systolic_BP', 120)
        diastolic = measurements.get('Diastolic_BP', 80)
        bp_score = min(100, max(0, (systolic - 120) * 1.5 + (diastolic - 80) * 2))
        
        # Cholesterol contribution (30%)
        chol = measurements.get('Cholesterol_Total', 200)
        chol_score = min(100, max(0, (chol - 200) * 0.5))
        
        # Other factors (40%)
        bmi_score = min(100, max(0, (measurements.get('BMI', 25) - 25) * 4))
        age_score = min(100, max(0, (measurements.get('AGE', 50) - 40) * 2))
        
        risk_score = 0.3 * bp_score + 0.3 * chol_score + 0.2 * bmi_score + 0.2 * age_score

    elif category == "METABOLIC":
        # Glucose contribution (40%)
        glucose = measurements.get('Blood_Glucose', 100)
        glucose_score = min(100, max(0, (glucose - 100) * 1.0))
        
        # BMI contribution (30%)
        bmi_score = min(100, max(0, (measurements.get('BMI', 25) - 25) * 4))
        
        # Other factors (30%)
        waist_score = min(100, max(0, (measurements.get('Waist_Circumference', 90) - 90) * 2))
        
        risk_score = 0.4 * glucose_score + 0.3 * bmi_score + 0.3 * waist_score

    elif category == "RESPIRATORY":
        # Oxygen saturation contribution (40%)
        o2_score = min(100, max(0, (100 - measurements.get('Oxygen_Saturation', 98)) * 10))
        
        # Respiratory rate contribution (30%)
        resp_rate = measurements.get('Respiratory_Rate', 16)
        resp_score = min(100, max(0, abs(resp_rate - 16) * 5))
        
        # Other factors (30%)
        fev1_score = min(100, max(0, (3 - measurements.get('FEV1', 3)) * 33.3))
        
        risk_score = 0.4 * o2_score + 0.3 * resp_score + 0.3 * fev1_score

    return min(100, max(0, risk_score))

def get_measurement_trend(history: List[Dict[str, float]], 
                        measurement: str) -> Optional[float]:
    """
    Calculate trend for a specific measurement from historical data
    Returns percentage change over time
    """
    if not history or len(history) < 2:
        return None
        
    values = [h.get(measurement) for h in history if measurement in h]
    if len(values) < 2:
        return None
        
    start_val = values[0]
    end_val = values[-1]
    if start_val == 0:
        return None
        
    return ((end_val - start_val) / start_val) * 100

def generate_health_recommendations(data: Dict[str, float], 
                                 predictions: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Generate personalized health recommendations based on measurements and predictions
    """
    recommendations = []
    
    # Cardiovascular recommendations
    if predictions.get('CARDIOVASCULAR', {}).get('probability', 0) > 30:
        if data.get('Systolic_BP', 0) > 140:
            recommendations.append("Monitor blood pressure daily")
        if data.get('Cholesterol_Total', 0) > 200:
            recommendations.append("Consider cholesterol management strategies")
            
    # Metabolic recommendations
    if predictions.get('METABOLIC', {}).get('probability', 0) > 30:
        if data.get('Blood_Glucose', 0) > 100:
            recommendations.append("Monitor blood glucose regularly")
        if data.get('BMI', 0) > 25:
            recommendations.append("Consider weight management program")
            
    # Respiratory recommendations
    if predictions.get('RESPIRATORY', {}).get('probability', 0) > 30:
        if data.get('Oxygen_Saturation', 0) < 95:
            recommendations.append("Monitor oxygen levels")
        if data.get('FEV1', 0) < 2.5:
            recommendations.append("Consider pulmonary rehabilitation")
            
    # General recommendations
    if data.get('Smoking_Status', 0) > 0:
        recommendations.append("Consider smoking cessation program")
    if len(recommendations) == 0:
        recommendations.append("Maintain current healthy lifestyle")
        
    return recommendations
