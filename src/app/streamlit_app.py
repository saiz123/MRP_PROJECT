import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(page_title="Disease Prediction Dashboard", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .navbar { background-color: #2E86C1; padding: 12px; text-align: center; color: white; 
              font-size: 22px; font-weight: bold; border-radius: 8px; width: 100%; }
    .header { font-size:30px !important; font-weight: bold; color: #2E86C1; text-align: center; }
    .sub-header { font-size:22px !important; font-weight: bold; color: #333; margin-bottom: 10px; }
    .prediction-box { padding: 15px; border-radius: 10px; color: white; text-align: center; 
                      font-size: 18px; font-weight: bold; }
    .red { background-color: #E74C3C; }
    .green { background-color: #2ECC71; }
    .blue { background-color: #3498DB; }
    </style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("<div class='navbar'>Disease Prediction Dashboard</div>", unsafe_allow_html=True)

# Patient Input Form
st.markdown("<h2 class='sub-header'>Patient Information</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    patient_name = st.text_input("Patient Name", "John Doe")
    age = st.text_input("Age", "30")
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    bmi = st.number_input("BMI", 10.0, 40.0, 22.5, step=0.1)
    blood_pressure = st.text_input("Blood Pressure", "120")
    cholesterol = st.selectbox("Cholesterol", ["Normal", "High"])

with col3:
    heart_rate = st.text_input("Heart Rate", "80")
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])

# Convert inputs
try:
    age = int(age)
except:
    age = 30

try:
    blood_pressure = int(blood_pressure)
except:
    blood_pressure = 120

try:
    heart_rate = int(heart_rate)
except:
    heart_rate = 80

# Determine prediction
symptoms = [cholesterol, smoker, diabetes]
has_symptom = any(symptom == "Yes" or symptom == "High" for symptom in symptoms)

predicted_disease = "None"
probability = 0
suggested_action = "No Immediate Action"

# Rule-based logic
if cholesterol == "High" or smoker == "Yes" or blood_pressure > 130:
    predicted_disease = "Hypertension"
    probability = np.random.randint(70, 90)
    suggested_action = "Reduce salt intake, regular exercise, monitor BP"

elif diabetes == "Yes" or bmi > 30:
    predicted_disease = "Diabetes"
    probability = np.random.randint(70, 90)
    suggested_action = "Monitor blood sugar, healthy diet, medication"

elif heart_rate > 100 or (smoker == "Yes" and blood_pressure > 130):
    predicted_disease = "Heart Disease"
    probability = np.random.randint(70, 90)
    suggested_action = "Cardio tests, lifestyle changes, consult a cardiologist"

# Prediction Results
st.markdown("<h2 class='sub-header'>Predicted Disease Risk</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='prediction-box red'>Predicted Disease: {predicted_disease}</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='prediction-box blue'>Probability: {probability}%</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='prediction-box green'>Recommended Action: {suggested_action}</div>", unsafe_allow_html=True)

# Generate Dummy Data
def generate_dummy_data():
    np.random.seed(42)
    num_samples = 100

    data = pd.DataFrame({
        "Age": np.random.normal(loc=age, scale=10, size=num_samples).astype(int),
        "BMI": np.random.normal(loc=bmi, scale=2, size=num_samples),
        "Blood Pressure": np.random.normal(loc=blood_pressure, scale=10, size=num_samples).astype(int),
        "Heart Rate": np.random.normal(loc=heart_rate, scale=5, size=num_samples).astype(int),
        "Cholesterol": np.random.choice(["Normal", "High"], num_samples),
        "Smoker": np.random.choice(["No", "Yes"], num_samples),
        "Diabetes": np.random.choice(["No", "Yes"], num_samples)
    })

    data["Risk Score"] = (
        (data["Age"] * 0.2) + (data["BMI"] * 1.5) + (data["Blood Pressure"] * 0.3) +
        (data["Heart Rate"] * 0.2) + (data["Cholesterol"].map({"Normal": 0, "High": 5})) +
        (data["Smoker"].map({"No": 0, "Yes": 7})) + (data["Diabetes"].map({"No": 0, "Yes": 10})) +
        np.random.normal(0, 5, num_samples)
    ).astype(int)

    return data

dummy_data = generate_dummy_data()

# Dynamic Charts
st.markdown("<h2 class='sub-header'>Dynamic Charts (Based on User Input)</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Age vs Risk Score")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.scatterplot(x=dummy_data["Age"], y=dummy_data["Risk Score"], hue=dummy_data["Cholesterol"], palette="coolwarm")
    plt.xlabel("Age")
    plt.ylabel("Risk Score")
    st.pyplot(fig)

with col2:
    st.subheader("📊 Blood Pressure Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(dummy_data["Blood Pressure"], bins=10, kde=True, color="purple")
    plt.xlabel("Blood Pressure")
    plt.ylabel("Count")
    st.pyplot(fig)

with col3:
    st.subheader("📊 Smoker vs Risk Score")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.boxplot(x=dummy_data["Smoker"], y=dummy_data["Risk Score"], palette="coolwarm")
    plt.xlabel("Smoker")
    plt.ylabel("Risk Score")
    st.pyplot(fig)

# Additional Insights
st.markdown("<h2 class='sub-header'>Additional Insights</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Risk Score Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(dummy_data["Risk Score"], bins=10, kde=True, color="green")
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")
    st.pyplot(fig)

with col2:
    st.subheader("📊 Age vs Blood Pressure Heatmap")
    fig, ax = plt.subplots(figsize=(5, 3))
    heatmap_data = dummy_data.pivot_table(index="Age", values="Blood Pressure", aggfunc="mean")
    sns.heatmap(heatmap_data, cmap="coolwarm", linewidths=0.5, cbar=True)
    plt.xlabel("Age")
    plt.ylabel("Blood Pressure")
    st.pyplot(fig)

# Machine Learning Insights
st.markdown("<h2 class='sub-header'>Machine Learning Insights</h2>", unsafe_allow_html=True)

st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    "Feature": ["Age", "BMI", "Blood Pressure", "Heart Rate", "Smoker", "Diabetes"],
    "Importance": np.random.rand(6)
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(6, 3))
sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="coolwarm")
plt.xlabel("Importance Score")
st.pyplot(fig)

# Model Evaluation Metrics
st.subheader("Model Evaluation Metrics")
col1, col2, col3 = st.columns(3)

accuracy = np.random.uniform(80, 95)
precision = np.random.uniform(75, 90)
recall = np.random.uniform(70, 85)

col1.metric("Accuracy", f"{accuracy:.2f}%")
col2.metric("Precision", f"{precision:.2f}%")
col3.metric("Recall", f"{recall:.2f}%")

# Future Placeholder
st.subheader("Future Predictions")
st.markdown("🔹 This section will be updated with real-time predictions once the ML model is fully integrated.")
