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

# # Navbar
# menu = st.sidebar.radio("Navigation", ["🏠 Home", "👤 Profile"])

# # **🔹 Home Section**
# if menu == "🏠 Home":
st.markdown("<div class='navbar'>Disease Prediction Dashboard</div>", unsafe_allow_html=True)

# **🔹 Patient Input Form**
st.markdown("<h2 class='sub-header'>Patient Information</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    patient_name = st.text_input("Patient Name", "John Doe")
    age = st.slider("Age", 1, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    bmi = st.number_input("BMI", 10.0, 40.0, 22.5, step=0.1)
    blood_pressure = st.slider("Blood Pressure", 90, 180, 120)
    cholesterol = st.selectbox("Cholesterol", ["Normal", "High"])

with col3:
    heart_rate = st.slider("Heart Rate", 50, 150, 80)
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])

# **🔹 Dummy Prediction Logic (To be replaced with ML)**
predicted_disease = np.random.choice(["Hypertension", "Diabetes", "Heart Disease", "None"])
probability = np.random.randint(70, 99)
suggested_action = np.random.choice(["Lifestyle Changes", "Medication", "Regular Checkups", "No Immediate Action"])

# **🔹 Prediction Results**
st.markdown("<h2 class='sub-header'>Predicted Disease Risk</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='prediction-box red'>Predicted Disease: {predicted_disease}</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='prediction-box blue'>Probability: {probability}%</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='prediction-box green'>Recommended Action: {suggested_action}</div>", unsafe_allow_html=True)

# **🔹 Generate Dynamic Data**
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

# **🔹 Dynamic Charts**
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

# **🔹 Additional Insightful Visualizations**
st.markdown("<h2 class='sub-header'>Additional Insights</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Risk Score Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(dummy_data["Risk Score"], bins=10, kde=True, color="green")
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")
    st.pyplot(fig)

# **New Chart 2: Age vs Blood Pressure Heatmap**
with col2:
    st.subheader("📊 Age vs Blood Pressure Heatmap")
    fig, ax = plt.subplots(figsize=(5, 3))
    heatmap_data = dummy_data.pivot_table(index="Age", values="Blood Pressure", aggfunc="mean")
    sns.heatmap(heatmap_data, cmap="coolwarm", linewidths=0.5, cbar=True)
    plt.xlabel("Age")
    plt.ylabel("Blood Pressure")
    st.pyplot(fig)

    
# **🔹 Machine Learning Insights**
st.markdown("<h2 class='sub-header'>Machine Learning Insights</h2>", unsafe_allow_html=True)

# Dummy Feature Importance (To be replaced with actual ML model)
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    "Feature": ["Age", "BMI", "Blood Pressure", "Heart Rate", "Smoker", "Diabetes"],
    "Importance": np.random.rand(6)
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(6, 3))
sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="coolwarm")
plt.xlabel("Importance Score")
st.pyplot(fig)

# Dummy Model Evaluation Metrics (Placeholder for actual model)
st.subheader("Model Evaluation Metrics")
col1, col2, col3 = st.columns(3)

accuracy = np.random.uniform(80, 95)
precision = np.random.uniform(75, 90)
recall = np.random.uniform(70, 85)

col1.metric("Accuracy", f"{accuracy:.2f}%")
col2.metric("Precision", f"{precision:.2f}%")
col3.metric("Recall", f"{recall:.2f}%")

# Future Placeholder: Model Predictions
st.subheader("Future Predictions")
st.markdown("🔹 This section will be updated with real-time predictions once the ML model is fully integrated.")



# # **🔹 Profile Section**
# elif menu == "👤 Profile":
    # st.markdown("<div class='header'>User Profile</div>", unsafe_allow_html=True)

    # # Profile Image & Basic Info
    # col1, col2 = st.columns([1, 4])

    # with col1:
    #     st.image("https://www.vhv.rs/dpng/d/15-155087_dummy-image-of-user-hd-png-download.png", caption="User Photo", use_container_width=True)

    # with col2:
    #     st.markdown("### **John Doe**")
    #     st.markdown("**Age:** 45  &nbsp; | &nbsp;  **Gender:** Male")
    #     st.markdown("**Email:** johndoe@example.com")
    #     st.markdown("**Phone:** +1-234-567-8901")
    #     st.markdown("**Address:** 123 Health St, Wellness City, USA")

    # st.markdown("---")

    # # Medical History
    # st.markdown("<h3 class='sub-header'>🩺 Medical History</h3>", unsafe_allow_html=True)
    # st.markdown("- **Blood Pressure:** Slightly high (130/85)")
    # st.markdown("- **Cholesterol Level:** Normal")
    # st.markdown("- **Smoker:** No")
    # st.markdown("- **Diabetes:** No")
    # st.markdown("- **Past Conditions:** Mild hypertension (Managed)")
    
    # st.markdown("---")

    # # Recent Health Reports
    # st.markdown("<h3 class='sub-header'>📊 Recent Health Reports</h3>", unsafe_allow_html=True)
    # col1, col2, col3 = st.columns(3)

    # col1.metric("BMI", "24.5", "Normal")
    # col2.metric("Blood Sugar", "95 mg/dL", "Healthy")
    # col3.metric("Heart Rate", "78 bpm", "Normal")

    # st.markdown("---")

    # # Lifestyle & Habits
    # st.markdown("<h3 class='sub-header'>🏋️ Lifestyle & Habits</h3>", unsafe_allow_html=True)
    # col1, col2 = st.columns(2)

    # with col1:
    #     st.markdown("**Exercise Routine:** 3-4 times per week")
    #     st.markdown("**Diet Preference:** Balanced (Low Sugar, High Protein)")
    #     st.markdown("**Sleep Duration:** 7-8 hours per night")

    # with col2:
    #     st.markdown("**Water Intake:** ~2L per day")
    #     st.markdown("**Alcohol Consumption:** Occasional")
    #     st.markdown("**Stress Level:** Moderate")

    # st.markdown("---")

    # # Next Health Checkup
    # st.markdown("<h3 class='sub-header'>📅 Next Health Checkup</h3>", unsafe_allow_html=True)
    # st.markdown("**Upcoming Appointment:** May 15, 2025")
    # st.markdown("**Doctor:** Dr. Sarah Thompson (General Physician)")
    # st.markdown("**Clinic:** Wellness Medical Center, USA")

