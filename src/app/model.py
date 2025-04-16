import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

# Load the preprocessed data
data_path = os.path.join("..", "synthea_processed.csv")
df = pd.read_csv(data_path)

# Drop rows where 'disease' is null (if needed)
df = df.dropna(subset=["disease"])

# Define features and label
relevant_features = [
    'AGE', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'has_hypertension'
    # Add more features if needed
]

# Extract features and target variable
X = df[relevant_features]
y = df["disease"]

# Step 1: Split data using stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 2: Check class distribution in training set
class_distribution = y_train.value_counts()
print(f"Class distribution in training data: {class_distribution}")

# Initialize RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# If only one class exists in the training data, use class_weight='balanced'
if class_distribution.shape[0] == 1:
    print("Warning: Only one class present in training data. Using class_weight='balanced'.")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
else:
    # Apply SMOTE to handle class imbalance if both classes are present
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    rf.fit(X_train_resampled, y_train_resampled)

# Step 3: Train model (use the appropriate X_train based on the SMOTE step)
rf.fit(X_train, y_train)

# Step 4: Predictions
y_pred = rf.predict(X_test)

# Classification report (returns a dictionary)
classification_rep = classification_report(y_test, y_pred, output_dict=True)

# Debug: Print the classification report dictionary to check its format
print("Classification Report (as dictionary):", classification_rep)

# Initialize precision and recall variables
precision_class_0 = classification_rep['0']['precision']
recall_class_0 = classification_rep['0']['recall']

# Check if class '1' exists in the classification report (only if both classes are present)
precision_class_1 = recall_class_1 = None
if '1' in classification_rep:
    precision_class_1 = classification_rep['1']['precision']
    recall_class_1 = classification_rep['1']['recall']

# Save the metrics, including precision and recall for both classes
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "classification_report": classification_rep,  # This will be a dictionary
    "precision_class_0": precision_class_0,  # Precision for class "0"
    "recall_class_0": recall_class_0,  # Recall for class "0"
    "precision_class_1": precision_class_1,  # Precision for class "1" (if exists)
    "recall_class_1": recall_class_1,  # Recall for class "1" (if exists)
    "confusion_matrix": confusion_matrix(y_test, y_pred)
}

# Save the model, features, and metrics
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(relevant_features, "feature_names.pkl")
joblib.dump(metrics, "model_metrics.pkl")

print("\nModel and features saved successfully:")
print("- Model: random_forest_model.pkl")
print("- Features: feature_names.pkl")
print("- Metrics: model_metrics.pkl")
