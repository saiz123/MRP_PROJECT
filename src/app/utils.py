import numpy as np

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
