import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, Any

class MultiDiseasePredictor:
    def __init__(self):
        self.models = {
            'CARDIOVASCULAR': RandomForestClassifier(n_estimators=100, random_state=42),
            'METABOLIC': RandomForestClassifier(n_estimators=100, random_state=42),
            'RESPIRATORY': RandomForestClassifier(n_estimators=100, random_state=42),
            'INFECTIOUS': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.scalers = {}
        self.feature_names = joblib.load('feature_names.pkl')
        
    def preprocess_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        df = pd.DataFrame([data])
        
        # Fill missing values with defaults
        defaults = {
            'Heart_Rate': 75,
            'Cholesterol_Total': 200,
            'Blood_Glucose': 100,
            'Oxygen_Saturation': 98,
            'WBC_Count': 7500,
            'CRP': 1.0
        }
        df = df.fillna(defaults)
        
        # Scale numerical features
        numerical_features = [
            'AGE', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate',
            'Cholesterol_Total', 'Blood_Glucose', 'Oxygen_Saturation'
        ]
        
        for feature in numerical_features:
            if feature in df.columns:
                df[feature] = df[feature].astype(float)
        
        return df

    def predict_diseases(self, patient_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Predict multiple disease risks for a patient
        Returns dictionary with predictions for each disease category
        """
        df = self.preprocess_data(patient_data)
        predictions = {}
        
        for category, model in self.models.items():
            # Get relevant features for this category
            features = self.feature_names[category]
            X = df[features].values
            
            # Scale features if scaler exists
            if category in self.scalers:
                X = self.scalers[category].transform(X)
            
            # Get prediction probability
            try:
                prob = model.predict_proba(X)[0][1] * 100
            except:
                prob = 0.0
            
            predictions[category] = {
                'probability': prob,
                'features_used': features
            }
        
        return predictions

    def save_models(self):
        """Save trained models and scalers"""
        for category, model in self.models.items():
            joblib.dump(model, f'{category.lower()}_model.pkl')
            if category in self.scalers:
                joblib.dump(self.scalers[category], f'{category.lower()}_scaler.pkl')

    def load_models(self):
        """Load trained models and scalers"""
        for category in self.models.keys():
            try:
                self.models[category] = joblib.load(f'{category.lower()}_model.pkl')
                scaler_path = f'{category.lower()}_scaler.pkl'
                if os.path.exists(scaler_path):
                    self.scalers[category] = joblib.load(scaler_path)
            except FileNotFoundError:
                print(f"Model files for {category} not found. Using default model.")

# Function to be called from the Streamlit app
def predict_diseases(patient_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    predictor = MultiDiseasePredictor()
    predictor.load_models()
    return predictor.predict_diseases(patient_data)
