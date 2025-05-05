from src.app.model import predict_diseases

# Sample patient data for testing
sample_patient = {
    'AGE': 45,
    'BMI': 28.5,
    'GENDER_M': 1,  # 1 for male, 0 for female
    'Systolic_BP': 135,
    'Diastolic_BP': 85,
    'Heart_Rate': 78,
    'Cholesterol_Total': 210,
    'Blood_Glucose': 115,
    'Oxygen_Saturation': 97,
    'Body_Temperature': 37.1
}

# Make predictions
print("Running disease risk predictions...")
predictions = predict_diseases(sample_patient)

# Display the results
print("\nRisk Predictions for Sample Patient:")
print("====================================")
for category, diseases in predictions.items():
    print(f"\n{category} Diseases:")
    print("-----------------")
    for disease, details in diseases.items():
        print(f"  {disease}: {details['probability']:.1f}% risk")
        print(f"    Features used: {', '.join(details['features_used'])}")
        print(f"    Top actions: {details['actions'][0]}, {details['actions'][1]}")