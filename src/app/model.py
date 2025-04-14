from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Prepare dummy data for ML model
def train_ml_model(data):
    df = data.copy()

    # Encode categorical features
    le_chol = LabelEncoder()
    le_smoke = LabelEncoder()
    le_diab = LabelEncoder()

    df["Cholesterol"] = le_chol.fit_transform(df["Cholesterol"])  # Normal=0, High=1
    df["Smoker"] = le_smoke.fit_transform(df["Smoker"])            # No=0, Yes=1
    df["Diabetes"] = le_diab.fit_transform(df["Diabetes"])         # No=0, Yes=1

    # Create a binary target based on risk threshold
    df["High Risk"] = (df["Risk Score"] > df["Risk Score"].mean()).astype(int)

    features = ["Age", "BMI", "Blood Pressure", "Heart Rate", "Cholesterol", "Smoker", "Diabetes"]
    X = df[features]
    y = df["High Risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, le_chol, le_smoke, le_diab

# Example usage (you'll call this from your main Streamlit app)
# Make sure dummy_data, age, bmi, blood_pressure, heart_rate, cholesterol, smoker, diabetes are defined in that file

ml_model, le_chol, le_smoke, le_diab = train_ml_model(dummy_data)

chol_encoded = le_chol.transform([cholesterol])[0]
smoker_encoded = le_smoke.transform([smoker])[0]
diabetes_encoded = le_diab.transform([diabetes])[0]

input_features = np.array([[age, bmi, blood_pressure, heart_rate, chol_encoded, smoker_encoded, diabetes_encoded]])

ml_prediction = ml_model.predict(input_features)[0]
ml_prob = ml_model.predict_proba(input_features)[0][1] * 100

ml_predicted_disease = "High Risk" if ml_prediction == 1 else "Low Risk"
ml_probability = round(ml_prob, 1)
ml_suggested_action = "Consult Specialist" if ml_prediction == 1 else "Monitor Regularly"
