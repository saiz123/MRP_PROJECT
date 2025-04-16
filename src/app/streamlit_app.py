import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# Load model, features, and metrics
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    metrics = joblib.load("model_metrics.pkl")  # Dictionary: accuracy, precision, recall
    return model, feature_names, metrics

# Load processed data
@st.cache_data
def load_data():
    return pd.read_csv("../synthea_processed.csv")

# Preprocess form input to match training data
def preprocess_input(data, feature_names):
    base = {col: 0 for col in feature_names}
    base.update({
        "AGE": data["AGE"],
        "BMI": data["BMI"],
        "Systolic_BP": data["Systolic_BP"],
        "Diastolic_BP": data["Diastolic_BP"],
        "has_hypertension": 1 if data["Diabetes"] else 0,  # dummy logic
        "GENDER_M": 1 if data["Gender"] == "Male" else 0,
        "Smoking_Status_Smokes tobacco daily (finding)": 1 if data["Smoker"] else 0,
        "Smoking_Status_Never smoked tobacco (finding)": 0 if data["Smoker"] else 1,
        "ETHNICITY_nonhispanic": 1,
        "RACE_white": 1
    })
    return pd.DataFrame([base], columns=feature_names)

# Predict disease
def predict_disease(model, input_df):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df).max() * 100
    return prediction, probability

# Navbar
st.markdown("<div class='navbar'>Disease Prediction Dashboard</div>", unsafe_allow_html=True)

# Patient Input Form
st.markdown("<h2 class='sub-header'>Patient Information</h2>", unsafe_allow_html=True)

# Create columns with styling for a cleaner look
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 0, 120, 30, help="Enter your age.")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender.")
    bmi = st.number_input("BMI", 10.0, 40.0, 22.5, help="Enter your BMI.")

with col2:
    systolic_bp = st.number_input("Systolic Blood Pressure", 60, 200, 120, help="Enter systolic BP.")
    diastolic_bp = st.number_input("Diastolic Blood Pressure", 40, 120, 80, help="Enter diastolic BP.")
    smoker = st.selectbox("Smoker", ["No", "Yes"], help="Select if you smoke.")

with col3:
    cholesterol = st.selectbox("Cholesterol", ["Normal", "High"], help="Select your cholesterol level.")
    diabetes = st.selectbox("High Glucose Level?", ["No", "Yes"], help="Select if you have high glucose levels.")
    
    submit = st.button("Predict Disease", help="Click to predict disease based on your inputs.", key="submit", use_container_width=True)

    # Apply the updated button styles
    st.markdown("""
        <style>
            .css-1emrehy.edgvbvh3 {
                background-color: #4CAF50; /* Vibrant Green */
                color: white;
                font-size: 20px;
                border-radius: 10px;
                padding: 15px;
                transition: background-color 0.3s ease;
                width: 100%;
                text-align: center;
            }
            .css-1emrehy.edgvbvh3:hover {
                background-color: #388E3C; /* Darker Green for hover effect */
            }
        </style>
    """, unsafe_allow_html=True)


# Load model/data
model, feature_names, metrics = load_model()
df = load_data()

# Prediction
if submit:
    user_input = {
        "AGE": age,
        "Gender": gender,
        "BMI": bmi,
        "Systolic_BP": systolic_bp,
        "Diastolic_BP": diastolic_bp,
        "Smoker": smoker == "Yes",
        "Diabetes": diabetes == "Yes"
    }

    input_df = preprocess_input(user_input, feature_names)
    pred, prob = predict_disease(model, input_df)
    pred_label = "No Disease" if pred == 0 else "Disease Detected"
    prob = prob

    st.subheader("Prediction Results")

    # Display metrics in vibrant color boxes (green, orange, blue)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<div class='prediction-box red'><strong>Prediction:</strong> {pred_label}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='prediction-box blue'> <strong>Probability:</strong> {prob:.2f}%</div>", unsafe_allow_html=True)

    with col3:
        action = "Visit Doctor" if prob > 50 else "Monitor"
        st.markdown(f"<div class='prediction-box green'><strong>Action:</strong> {action}</div>", unsafe_allow_html=True)

# Visualizations
st.header("Data Insights")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Blood Pressure Variation with Age")
    # Group the data by age and calculate average systolic and diastolic blood pressure
    bp_by_age = df.groupby("AGE")[["Systolic_BP", "Diastolic_BP"]].mean().reset_index()

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bp_by_age["AGE"], bp_by_age["Systolic_BP"], label="Systolic BP", color="#e74c3c", linewidth=2)
    ax.plot(bp_by_age["AGE"], bp_by_age["Diastolic_BP"], label="Diastolic BP", color="#3498db", linewidth=2)

    # Style the plot
    ax.set_title("Average Blood Pressure by Age", fontsize=14)
    ax.set_xlabel("Age")
    ax.set_ylabel("Blood Pressure (mm Hg)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()

    # Render in Streamlit
    st.pyplot(fig)

with col2:
    st.subheader("Age Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(df["AGE"], bins=30, kde=True, ax=ax, color="#6c5ce7")
    ax.set_title("Age Histogram", fontsize=11)
    ax.set_xlabel("Age")
    ax.set_ylabel("Average Count of Disease")
    st.pyplot(fig)

with col3:
    st.subheader("Smoking Status Distribution")
    smoking_counts = df['Smoking_Status_Never smoked tobacco (finding)'].value_counts()
    labels = ['Never Smoked' if val == 1 else 'Smoker' for val in smoking_counts.index]
    sizes = smoking_counts.values.tolist()

    def format_autopct(pct, allvals):
        absolute = int(round(pct / 100. * sum(allvals)))
        return f"{pct:.1f}%\n({absolute})"

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: format_autopct(pct, sizes),
        startangle=90,
        colors=plt.cm.Pastel1.colors,
        wedgeprops={'edgecolor': 'black'},
        radius=0.7
    )
    ax.axis('equal')
    st.pyplot(fig)

# Model metrics note
st.header("Model Performance")

# Load the saved metrics from the pickle file
metrics = joblib.load("model_metrics.pkl")

# Extract relevant metrics
accuracy = metrics["accuracy"]
classification_report = metrics["classification_report"]

# For class "0" (negative class)
precision_class_0 = classification_report['0']['precision']
recall_class_0 = classification_report['0']['recall']
f1_class_0 = classification_report['0']['f1-score']

# For class "1" (positive class), check if it exists
precision_class_1 = recall_class_1 = f1_class_1 = None
if '1' in classification_report:
    precision_class_1 = classification_report['1']['precision']
    recall_class_1 = classification_report['1']['recall']
    f1_class_1 = classification_report['1']['f1-score']

# Display Machine Learning Insights at the bottom of the page
st.subheader("Machine Learning Insights")

# Create 4 columns for display
col1, col2, col3, col4 = st.columns(4)

# Display the metrics in the columns
with col1:
    st.metric(label="Accuracy", value=f"{accuracy*100:.2f}%")

with col2:
    st.metric(label="Precision", value=f"{precision_class_0*100:.2f}%")

with col3:
    st.metric(label="Recall", value=f"{recall_class_0*100:.2f}%")

with col4:
    st.metric(label="F1 Score", value=f"{f1_class_0:.2f}")

