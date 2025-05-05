import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from typing import Dict, Any, List

class MultiDiseasePredictor:
    def __init__(self):
        # Define specific diseases for each category
        self.disease_mapping = {
            'CARDIOVASCULAR': ['Hypertension', 'Heart_Attack'],
            'METABOLIC': ['Diabetes', 'Obesity'],
            'RESPIRATORY': ['Asthma', 'COPD'],
            'ONCOLOGY': ['Breast_Cancer', 'Lung_Cancer', 'Colon_Cancer'],
            'INFECTIOUS': ['Hepatitis', 'Influenza', 'Flu']
        }
        
        # Define paths
        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.MODELS_DIR = os.path.join(self.ROOT_DIR, 'models')
        self.FINAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_model')
        
        # Create directories if they don't exist
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.FINAL_MODEL_DIR, exist_ok=True)
        
        # Path to the synthea data
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', 'synthea_processed.csv')
        
        # Unified model approach - a single model for all diseases
        self.unified_model = None
        self.unified_scaler = StandardScaler()
        
        # Keep the individual models dictionary for backward compatibility
        self.models = {}
        for category, diseases in self.disease_mapping.items():
            for disease in diseases:
                self.models[f"{category}_{disease}"] = None  # Placeholder
        
        self.scalers = {}
        
        # Define feature mappings from synthea dataset to our disease models
        self.feature_mapping = {
            'CARDIOVASCULAR_Hypertension': {
                'target': 'Essential hypertension (disorder)',
                'features': ['AGE', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'GENDER_M']
            },
            'CARDIOVASCULAR_Heart_Attack': {
                'target': 'Myocardial infarction (disorder)',
                'features': ['AGE', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'GENDER_M']
            },
            'METABOLIC_Diabetes': {
                'target': 'Diabetes mellitus type 2 (disorder)',
                'features': ['AGE', 'BMI', 'GENDER_M']
            },
            'METABOLIC_Obesity': {
                'target': 'Body mass index 30+ - obesity (finding)',
                'features': ['AGE', 'BMI', 'GENDER_M']
            },
            'RESPIRATORY_Asthma': {
                'target': 'Asthma (disorder)',
                'features': ['AGE', 'BMI', 'GENDER_M', 'Smoking_Status_Smokes tobacco daily (finding)']
            },
            'RESPIRATORY_COPD': {
                'target': 'Chronic obstructive bronchitis (disorder)',
                'features': ['AGE', 'BMI', 'GENDER_M', 'Smoking_Status_Smokes tobacco daily (finding)']
            },
            'ONCOLOGY_Breast_Cancer': {
                'target': 'Malignant neoplasm of breast (disorder)',
                'features': ['AGE', 'BMI', 'GENDER_M']
            },
            'ONCOLOGY_Lung_Cancer': {
                'target': 'Small cell carcinoma of lung (disorder)',
                'features': ['AGE', 'BMI', 'GENDER_M', 'Smoking_Status_Smokes tobacco daily (finding)']
            },
            'ONCOLOGY_Colon_Cancer': {
                'target': 'Malignant neoplasm of colon (disorder)',
                'features': ['AGE', 'BMI', 'GENDER_M']
            },
            'INFECTIOUS_Hepatitis': {
                'target': 'Chronic hepatitis C (disorder)',
                'features': ['AGE', 'BMI', 'GENDER_M']
            },
            'INFECTIOUS_Influenza': {
                'target': 'Viral sinusitis (disorder)',
                'features': ['AGE', 'BMI', 'GENDER_M']
            },
            'INFECTIOUS_Flu': {
                'target': 'Acute viral pharyngitis (disorder)',
                'features': ['AGE', 'BMI', 'GENDER_M']
            }
        }
        
        # Define default feature names for each disease
        self.default_feature_names = {
            # Cardiovascular diseases
            'CARDIOVASCULAR_Hypertension': ['AGE', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'GENDER_M'],
            'CARDIOVASCULAR_Heart_Attack': ['AGE', 'BMI', 'Systolic_BP', 'Heart_Rate', 'Cholesterol_Total'],
            
            # Metabolic diseases
            'METABOLIC_Diabetes': ['AGE', 'BMI', 'Blood_Glucose', 'Cholesterol_Total'],
            'METABOLIC_Obesity': ['AGE', 'BMI', 'Waist_Circumference'],
            
            # Respiratory diseases
            'RESPIRATORY_Asthma': ['AGE', 'BMI', 'Oxygen_Saturation', 'Respiratory_Rate', 'FEV1'],
            'RESPIRATORY_COPD': ['AGE', 'Oxygen_Saturation', 'Respiratory_Rate', 'FEV1', 'Smoking_Status'],
            
            # Oncology diseases
            'ONCOLOGY_Breast_Cancer': ['AGE', 'GENDER_M', 'BMI', 'Family_History'],
            'ONCOLOGY_Lung_Cancer': ['AGE', 'Smoking_Status', 'Respiratory_Rate', 'FEV1'],
            'ONCOLOGY_Colon_Cancer': ['AGE', 'BMI', 'Family_History'],
            
            # Infectious diseases
            'INFECTIOUS_Hepatitis': ['AGE', 'Liver_Enzymes'],
            'INFECTIOUS_Influenza': ['AGE', 'Body_Temperature', 'Oxygen_Saturation', 'WBC_Count'],
            'INFECTIOUS_Flu': ['AGE', 'Body_Temperature', 'WBC_Count', 'CRP']
        }
        
        # Define all possible features for the unified model
        self.all_features = self._get_all_unique_features()
        
        # Try to load feature names from file, use defaults if not found
        try:
            # Check in final_model directory first
            feature_names_path = os.path.join(self.FINAL_MODEL_DIR, 'feature_names.pkl')
            if os.path.exists(feature_names_path):
                self.feature_names = joblib.load(feature_names_path)
            # Then try models directory
            elif os.path.exists(os.path.join(self.MODELS_DIR, 'feature_names.pkl')):
                self.feature_names = joblib.load(os.path.join(self.MODELS_DIR, 'feature_names.pkl'))
            # Then try app directory (backward compatibility)
            elif os.path.exists('feature_names.pkl'):
                self.feature_names = joblib.load('feature_names.pkl')
            else:
                self.feature_names = self.default_feature_names
                # Save the default feature names for future use
                self._create_default_files()
        except Exception as e:
            print(f"Error loading feature_names.pkl: {e}")
            self.feature_names = self.default_feature_names
            self._create_default_files()
        
        # Load the unified model if it exists
        self._load_unified_model()
    
    def _get_all_unique_features(self):
        """Get a list of all unique features used across all disease models"""
        all_features = set()
        for features in self.default_feature_names.values():
            all_features.update(features)
        return sorted(list(all_features))
    
    def _create_default_files(self):
        """Generate default model files and feature names if they don't exist"""
        # Create directories if they don't exist
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.FINAL_MODEL_DIR, exist_ok=True)
        
        # Save default feature names to all locations
        joblib.dump(self.default_feature_names, os.path.join(self.FINAL_MODEL_DIR, 'feature_names.pkl'))
        joblib.dump(self.default_feature_names, os.path.join(self.MODELS_DIR, 'feature_names.pkl'))
        joblib.dump(self.default_feature_names, 'feature_names.pkl') # For backward compatibility
        
        # Create a unified model file
        self._create_unified_model()
        
        print("Created unified model and feature names files")
    
    def _load_synthea_data(self):
        """Load and preprocess the Synthea dataset"""
        try:
            # Check if the data file exists
            if not os.path.exists(self.data_path):
                print(f"Synthea data not found at {self.data_path}, using synthetic data instead")
                return None, None
            
            # Load the data
            print(f"Loading Synthea dataset from {self.data_path}")
            data = pd.read_csv(self.data_path)
            
            # Convert boolean columns to integers
            for col in data.columns:
                if data[col].dtype == bool:
                    data[col] = data[col].astype(int)
            
            # Print dataset info
            print(f"Loaded dataset with {data.shape[0]} patients and {data.shape[1]} features")
            return data, None
        except Exception as e:
            print(f"Error loading Synthea data: {e}")
            return None, None
    
    def _create_unified_model(self):
        """Create a unified model using Synthea data if available, otherwise use synthetic data"""
        # Try to load Synthea data
        data, _ = self._load_synthea_data()
        
        if data is not None:
            print("Creating unified model using Synthea data")
            # Train models using real data
            self._train_models_with_real_data(data)
        else:
            print("Creating unified model using synthetic data")
            # Generate synthetic training data as a fallback
            self._train_models_with_synthetic_data()
    
    def _train_models_with_real_data(self, data):
        """Train models using real patient data from Synthea"""
        # Create a dictionary to store target values for each disease
        y_dict = {}
        common_features = ['AGE', 'BMI', 'GENDER_M', 'Systolic_BP', 'Diastolic_BP']
        available_features = [col for col in common_features if col in data.columns]
        
        # Create initial feature dataframe with common features
        print(f"Using common features: {available_features}")
        X_df = data[available_features].copy()
        
        # Create target variables for each disease using the feature mapping
        print("Preparing disease target variables...")
        for disease_key, mapping in self.feature_mapping.items():
            target_col = mapping['target']
            if target_col in data.columns:
                y_dict[disease_key] = data[target_col].values
                print(f"Found target column for {disease_key}: {target_col}, positive cases: {sum(data[target_col])}")
            else:
                # If target column not found, use a rule-based approach to create synthetic targets
                print(f"Target column {target_col} not found for {disease_key}, using rule-based approach")
                if "Hypertension" in disease_key:
                    y_dict[disease_key] = ((data['Systolic_BP'] > 140) | (data['Diastolic_BP'] > 90)).astype(int).values
                elif "Obesity" in disease_key:
                    y_dict[disease_key] = (data['BMI'] > 30).astype(int).values
                elif "Diabetes" in disease_key:
                    if 'Blood_Glucose' in data.columns:
                        y_dict[disease_key] = (data['Blood_Glucose'] > 126).astype(int).values
                    else:
                        y_dict[disease_key] = (data['BMI'] > 30).astype(int).values * 0.3
                else:
                    # Create synthetic target by using age and gender as risk factors
                    age_factor = (data['AGE'] > 50).astype(int) * 0.1
                    gender_factor = data['GENDER_M'].astype(int) * 0.05
                    y_dict[disease_key] = np.random.binomial(1, age_factor + gender_factor, size=data.shape[0])
        
        # Scale the features
        print("Scaling features...")
        unified_scaler = StandardScaler()
        X_scaled = unified_scaler.fit_transform(X_df)
        
        # Create unified model structure
        unified_model = {
            'scaler': unified_scaler,
            'features': X_df.columns.tolist(),
            'disease_models': {}
        }
        
        # Train a model for each disease
        print("Training disease models...")
        for disease_key, y in y_dict.items():
            print(f"Training model for {disease_key}...")
            
            # Split data for training and validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            print(f"  Train accuracy: {train_accuracy:.2f}, Test accuracy: {test_accuracy:.2f}")
            
            # Store the trained model
            unified_model['disease_models'][disease_key] = model
        
        # Save the unified model
        print("Saving unified model...")
        self.unified_model = unified_model
        joblib.dump(unified_model, os.path.join(self.FINAL_MODEL_DIR, 'unified_disease_model.pkl'))
        joblib.dump(unified_model, os.path.join(self.MODELS_DIR, 'unified_disease_model.pkl'))
        joblib.dump(unified_model, 'unified_disease_model.pkl')
    
    def _train_models_with_synthetic_data(self):
        """Create a unified model using synthetic data when real data is unavailable"""
        # Generate synthetic training data
        n_samples = 2000
        print(f"Generating {n_samples} synthetic patient records...")
        
        # Define feature distributions for synthetic data
        feature_distributions = {
            'AGE': (50, 15),  # mean=50, std=15
            'BMI': (25, 5),   # mean=25, std=5
            'Systolic_BP': (120, 15),
            'Diastolic_BP': (80, 10),
            'Heart_Rate': (75, 10),
            'Cholesterol_Total': (200, 40),
            'Blood_Glucose': (100, 25),
            'Oxygen_Saturation': (97, 2),
            'WBC_Count': (7500, 2000),
            'CRP': (1.0, 1.0),
            'Respiratory_Rate': (16, 3),
            'FEV1': (3.0, 0.5),
            'Waist_Circumference': (90, 15),
            'Body_Temperature': (37.0, 0.5),
            'Liver_Enzymes': (30, 10)
        }
        
        # Generate features with appropriate distributions
        X_df = pd.DataFrame(columns=self.all_features)
        
        for feature in self.all_features:
            if feature in feature_distributions:
                mean, std = feature_distributions[feature]
                X_df[feature] = np.random.normal(mean, std, n_samples)
            elif feature == 'GENDER_M':
                X_df[feature] = np.random.choice([0, 1], n_samples)
            elif feature == 'Family_History':
                X_df[feature] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
            elif feature == 'Smoking_Status':
                X_df[feature] = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.2, 0.2])
            else:
                X_df[feature] = np.random.rand(n_samples)
        
        # Generate target variables with dependencies on features
        y_dict = {}
        
        # For each disease, create a prediction rule based on risk factors
        bp_risk = (X_df['Systolic_BP'] > 130).astype(int) + (X_df['Diastolic_BP'] > 85).astype(int)
        age_risk = (X_df['AGE'] > 60).astype(int)
        bmi_risk = (X_df['BMI'] > 30).astype(int)
        
        # Cardiovascular - Hypertension
        y_dict['CARDIOVASCULAR_Hypertension'] = ((bp_risk + age_risk + bmi_risk >= 2) | 
                                               (X_df['Systolic_BP'] > 140) | 
                                               (X_df['Diastolic_BP'] > 90)).astype(int)
        
        # Add more disease rules similar to those in create_unified_model.py
        # Cardiovascular - Heart Attack
        cholesterol_risk = (X_df['Cholesterol_Total'] > 240).astype(int)
        y_dict['CARDIOVASCULAR_Heart_Attack'] = ((age_risk + bp_risk + cholesterol_risk >= 2) | 
                                               (X_df['Systolic_BP'] > 160)).astype(int)
        
        # Metabolic - Diabetes
        glucose_risk = (X_df['Blood_Glucose'] > 126).astype(int) + (X_df['Blood_Glucose'] > 100).astype(int)
        y_dict['METABOLIC_Diabetes'] = ((glucose_risk + bmi_risk + age_risk >= 2) | 
                                      (X_df['Blood_Glucose'] > 126)).astype(int)
        
        # Metabolic - Obesity
        y_dict['METABOLIC_Obesity'] = (X_df['BMI'] > 30).astype(int)
        
        # Respiratory - Asthma
        resp_rate_risk = (X_df['Respiratory_Rate'] > 20).astype(int)
        fev_risk = (X_df['FEV1'] < 2.5).astype(int)
        y_dict['RESPIRATORY_Asthma'] = (resp_rate_risk + fev_risk >= 1).astype(int)
        
        # Respiratory - COPD
        smoking_risk = (X_df['Smoking_Status'] > 0).astype(int)
        y_dict['RESPIRATORY_COPD'] = ((fev_risk + smoking_risk + age_risk >= 2) | 
                                    (X_df['FEV1'] < 2.0)).astype(int)
        
        # Oncology diseases
        family_risk = (X_df['Family_History'] > 0).astype(int)
        gender_risk = (X_df['GENDER_M'] == 0).astype(int)  # Higher risk for females
        y_dict['ONCOLOGY_Breast_Cancer'] = ((gender_risk + family_risk + age_risk >= 2)).astype(int)
        y_dict['ONCOLOGY_Lung_Cancer'] = ((smoking_risk + age_risk + fev_risk >= 2)).astype(int)
        y_dict['ONCOLOGY_Colon_Cancer'] = ((family_risk + age_risk + bmi_risk >= 2)).astype(int)
        
        # Infectious diseases
        liver_risk = (X_df['Liver_Enzymes'] > 40).astype(int)
        y_dict['INFECTIOUS_Hepatitis'] = (liver_risk == 1).astype(int)
        
        temp_risk = (X_df['Body_Temperature'] > 38.0).astype(int)
        wbc_risk = (X_df['WBC_Count'] > 11000).astype(int)
        y_dict['INFECTIOUS_Influenza'] = ((temp_risk + wbc_risk >= 1)).astype(int)
        y_dict['INFECTIOUS_Flu'] = ((temp_risk + wbc_risk >= 1)).astype(int)
        
        # Fit the unified scaler
        unified_scaler = StandardScaler()
        X_scaled = unified_scaler.fit_transform(X_df)
        
        # Create disease-specific probability estimators
        unified_model = {
            'scaler': unified_scaler,
            'features': self.all_features,
            'disease_models': {}
        }
        
        # Train a model for each disease with train/test split
        for disease_key in self.default_feature_names.keys():
            # Generate target based on features if not already defined
            if disease_key not in y_dict:
                # Use a simple random approach
                prob = np.random.uniform(0.1, 0.3, n_samples)
                y_dict[disease_key] = np.random.binomial(1, prob)
            
            print(f"Training model for {disease_key}...")
            
            # Split data for training and validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_dict[disease_key], test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            print(f"  Train accuracy: {train_accuracy:.2f}, Test accuracy: {test_accuracy:.2f}")
            
            # Add to unified model
            unified_model['disease_models'][disease_key] = model
            
            # Save individual models
            model_path = os.path.join(self.MODELS_DIR, f"{disease_key}_model.pkl")
            scaler_path = os.path.join(self.MODELS_DIR, f"{disease_key}_scaler.pkl")
            joblib.dump(model, model_path)
            joblib.dump(unified_scaler, scaler_path)
        
        # Save the unified model to different locations
        self.unified_model = unified_model
        
        # Save to final_model directory
        joblib.dump(unified_model, os.path.join(self.FINAL_MODEL_DIR, 'unified_disease_model.pkl'))
        
        # Save to models directory
        joblib.dump(unified_model, os.path.join(self.MODELS_DIR, 'unified_disease_model.pkl'))
        
        # Save to app directory for backward compatibility
        joblib.dump(unified_model, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unified_disease_model.pkl'))
    
    def _load_unified_model(self):
        """Load the unified model if it exists"""
        try:
            # Try loading from final_model directory first
            unified_model_path = os.path.join(self.FINAL_MODEL_DIR, 'unified_disease_model.pkl')
            if os.path.exists(unified_model_path):
                self.unified_model = joblib.load(unified_model_path)
                print(f"Loaded unified disease model from {unified_model_path}")
            # Then try models directory
            elif os.path.exists(os.path.join(self.MODELS_DIR, 'unified_disease_model.pkl')):
                self.unified_model = joblib.load(os.path.join(self.MODELS_DIR, 'unified_disease_model.pkl'))
                print(f"Loaded unified disease model from {self.MODELS_DIR}")
            # Then try app directory (backward compatibility)
            elif os.path.exists('unified_disease_model.pkl'):
                self.unified_model = joblib.load('unified_disease_model.pkl')
                print("Loaded unified disease model from current directory")
            else:
                print("Unified model not found, will create default")
                self._create_unified_model()
        except Exception as e:
            print(f"Error loading unified model: {e}")
            self._create_unified_model()
    
    def save_models(self):
        """Save the unified model"""
        # Create directories if they don't exist
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.FINAL_MODEL_DIR, exist_ok=True)
        
        # Save to final_model directory
        joblib.dump(self.unified_model, os.path.join(self.FINAL_MODEL_DIR, 'unified_disease_model.pkl'))
        
        # Save to models directory
        joblib.dump(self.unified_model, os.path.join(self.MODELS_DIR, 'unified_disease_model.pkl'))
        
        # Save to app directory for backward compatibility
        joblib.dump(self.unified_model, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unified_disease_model.pkl'))
        
        print("Saved unified disease model to all locations")

    def load_models(self):
        """Load the unified model"""
        try:
            # Try loading from final_model directory first
            unified_model_path = os.path.join(self.FINAL_MODEL_DIR, 'unified_disease_model.pkl')
            if os.path.exists(unified_model_path):
                self.unified_model = joblib.load(unified_model_path)
                print(f"Loaded unified disease model from {unified_model_path}")
            # Then try models directory
            elif os.path.exists(os.path.join(self.MODELS_DIR, 'unified_disease_model.pkl')):
                self.unified_model = joblib.load(os.path.join(self.MODELS_DIR, 'unified_disease_model.pkl'))
                print(f"Loaded unified disease model from {self.MODELS_DIR}")
            # Then try app directory (backward compatibility)
            elif os.path.exists('unified_disease_model.pkl'):
                self.unified_model = joblib.load('unified_disease_model.pkl')
                print("Loaded unified disease model from current directory")
            else:
                print("Unified model not found, creating default")
                self._create_unified_model()
        except Exception as e:
            print(f"Error loading unified model: {e}")
            self._create_unified_model()

    def predict_diseases(self, patient_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Predict multiple diseases for the patient based on their data"""
        try:
            # If we don't have a unified model, try to load again
            if self.unified_model is None:
                self._load_unified_model()
                
                # If still not available, return placeholder predictions
                if self.unified_model is None:
                    return self._get_placeholder_predictions()
            
            # Prepare patient data as a DataFrame
            patient_df = pd.DataFrame([patient_data])
            
            # Ensure all required features are present
            for feature in self.all_features:
                if feature not in patient_df.columns:
                    patient_df[feature] = 0.0
            
            # Select only the features needed for the model
            patient_features = patient_df[self.unified_model['features']]
            
            # Scale the features
            patient_scaled = self.unified_model['scaler'].transform(patient_features)
            
            # Make predictions for each disease
            results = {}
            
            for category, diseases in self.disease_mapping.items():
                category_results = {}
                
                for disease in diseases:
                    disease_key = f"{category}_{disease}"
                    if disease_key in self.unified_model['disease_models']:
                        model = self.unified_model['disease_models'][disease_key]
                        
                        # Use probability rather than binary prediction
                        # Fix here: Get the second class probability (positive class)
                        if hasattr(model, "predict_proba"):
                            try:
                                proba = model.predict_proba(patient_scaled)[0]
                                if len(proba) > 1:
                                    probability = proba[1] * 100
                                else:
                                    probability = proba[0] * 100
                            except Exception as e:
                                print(f"Error in predict_proba for {disease_key}: {e}")
                                probability = model.predict(patient_scaled)[0] * 100
                        else:
                            probability = model.predict(patient_scaled)[0] * 100
                        
                        # Special case for flu/influenza - make sure temperature and WBC thresholds are reasonable
                        if "INFECTIOUS_Flu" in disease_key or "INFECTIOUS_Influenza" in disease_key:
                            # Only high probability if actual clinical indicators are present
                            temp_elevated = patient_data.get('Body_Temperature', 37.0) > 38.0
                            wbc_elevated = patient_data.get('WBC_Count', 7500) > 11000
                            
                            # If normal temperature and WBC but high prediction, adjust probability down
                            if probability > 40 and not (temp_elevated or wbc_elevated):
                                probability = max(20, probability * 0.5)  # Cap at reasonable baseline
                        
                        features_used = self.default_feature_names.get(disease_key, [])
                        actions = self._get_disease_actions(disease_key, probability)
                        
                        category_results[disease] = {
                            'probability': probability,
                            'features_used': features_used,
                            'actions': actions
                        }
                
                # Aggregate probabilities for the category
                if category_results:
                    max_prob = max(disease_data['probability'] for disease_data in category_results.values())
                    avg_prob = sum(disease_data['probability'] for disease_data in category_results.values()) / len(category_results)
                    
                    # Use weighted average leaning toward maximum probability
                    category_prob = (max_prob * 0.7) + (avg_prob * 0.3)
                    
                    results[category] = category_results
            
            return results
            
        except Exception as e:
            print(f"Error in predict_diseases: {e}")
            return self._get_placeholder_predictions()

# Function to be called from the Streamlit app
def predict_diseases(patient_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    try:
        predictor = MultiDiseasePredictor()
        
        # Check if unified model exists in any of the expected locations
        final_model_path = os.path.join(predictor.FINAL_MODEL_DIR, 'unified_disease_model.pkl')
        models_path = os.path.join(predictor.MODELS_DIR, 'unified_disease_model.pkl')
        
        if not (os.path.exists(final_model_path) or os.path.exists(models_path) or os.path.exists('unified_disease_model.pkl')):
            print("Unified model not found in any location. Creating default model...")
            predictor._create_unified_model()
        else:
            predictor.load_models()
        
        return predictor.predict_diseases(patient_data)
        
    except Exception as e:
        print(f"Error in disease prediction: {e}")
        # Return default predictions in case of error
        return {
            'CARDIOVASCULAR': {
                'Hypertension': {'probability': 25.0, 'features_used': ['AGE', 'BMI', 'Systolic_BP'], 
                                'actions': ['Maintain healthy diet', 'Regular exercise', 'Annual blood pressure check']},
                'Heart_Attack': {'probability': 20.0, 'features_used': ['AGE', 'BMI', 'Cholesterol_Total'],
                                'actions': ['Maintain healthy lifestyle', 'Regular cardiovascular checkups']}
            },
            # ...existing code...
        }
