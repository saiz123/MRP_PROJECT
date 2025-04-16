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
            if os.path.exists('feature_names.pkl'):
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
        # Save default feature names
        joblib.dump(self.default_feature_names, 'feature_names.pkl')
        
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
        
        # ...existing synthetic data generation code...
        
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
        
        # Save the unified model
        self.unified_model = unified_model
        joblib.dump(unified_model, 'unified_disease_model.pkl')
    
    def _load_unified_model(self):
        """Load the unified model if it exists"""
        try:
            if os.path.exists('unified_disease_model.pkl'):
                self.unified_model = joblib.load('unified_disease_model.pkl')
                print("Loaded unified disease model")
            else:
                print("Unified model not found, will create default")
                self._create_unified_model()
        except Exception as e:
            print(f"Error loading unified model: {e}")
            self._create_unified_model()
    
    # ...existing code for preprocess_data(), predict_with_unified_model(), etc...
    
    def predict_diseases(self, patient_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Predict risks for specific diseases
        Returns dictionary with predictions for each disease
        """
        df = self.preprocess_data(patient_data)
        
        # Use the unified model for predictions if available
        if isinstance(self.unified_model, dict) and 'disease_models' in self.unified_model:
            return self.predict_with_unified_model(df)
        
        # Otherwise fall back to the old method
        predictions = {}
        
        # Organize predictions by category
        for category, diseases in self.disease_mapping.items():
            category_predictions = {}
            
            for disease in diseases:
                model_key = f"{category}_{disease}"
                # Get relevant features for this disease
                features = self.feature_names[model_key]
                
                # Ensure all required features are present
                missing_features = [f for f in features if f not in df.columns]
                for f in missing_features:
                    df[f] = 0  # Add missing features with default values
                
                X = df[features].values
                
                # Use rule-based prediction as we don't have individual models anymore
                prob = self._get_rule_based_probability(disease, patient_data)
                
                category_predictions[disease] = {
                    'probability': prob,
                    'features_used': features,
                    'actions': self._get_recommended_actions(disease, prob)
                }
            
            predictions[category] = category_predictions
        
        return predictions
    
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
            'CRP': 1.0,
            'Respiratory_Rate': 16,
            'FEV1': 3.0,
            'Waist_Circumference': 90,
            'Family_History': 0,
            'Smoking_Status': 0,
            'Body_Temperature': 37.0,
            'Liver_Enzymes': 30
        }
        df = df.fillna(defaults)
        
        # Scale numerical features
        numerical_features = [
            'AGE', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate',
            'Cholesterol_Total', 'Blood_Glucose', 'Oxygen_Saturation',
            'WBC_Count', 'CRP', 'Respiratory_Rate', 'FEV1',
            'Waist_Circumference', 'Body_Temperature', 'Liver_Enzymes'
        ]
        
        for feature in numerical_features:
            if feature in df.columns:
                df[feature] = df[feature].astype(float)
        
        return df
    
    def predict_with_unified_model(self, df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Make predictions using the unified model"""
        # Get the features needed by the model
        model_features = self.unified_model['features']
        
        # Prepare input data with all required features
        input_df = pd.DataFrame(columns=model_features)
        
        # Fill with data from input df where available
        for feature in model_features:
            if feature in df.columns:
                input_df[feature] = df[feature]
            else:
                # Use 0 as default for missing features
                input_df[feature] = 0
        
        # Scale the input data using the unified scaler
        try:
            X_scaled = self.unified_model['scaler'].transform(input_df)
        except:
            # If scaling fails, use the data as is
            X_scaled = input_df.values
        
        # Make predictions for each disease
        predictions = {}
        
        for category, diseases in self.disease_mapping.items():
            category_predictions = {}
            
            for disease in diseases:
                disease_key = f"{category}_{disease}"
                if disease_key in self.unified_model['disease_models']:
                    # Get prediction from the unified model
                    try:
                        model = self.unified_model['disease_models'][disease_key]
                        prob = model.predict_proba(X_scaled)[0][1] * 100
                    except:
                        # Fall back to rule-based prediction
                        prob = self._get_rule_based_probability(disease, df.iloc[0].to_dict())
                else:
                    # If model missing, use rule-based prediction
                    prob = self._get_rule_based_probability(disease, df.iloc[0].to_dict())
                
                # Get relevant features for this disease from feature_names
                features = self.feature_names[disease_key]
                
                category_predictions[disease] = {
                    'probability': prob,
                    'features_used': features,
                    'actions': self._get_recommended_actions(disease, prob)
                }
            
            predictions[category] = category_predictions
        
        return predictions
    
    def _get_rule_based_probability(self, disease: str, data: Dict[str, Any]) -> float:
        """Provide a rule-based probability when model prediction fails"""
        # Default probability ranges for each disease
        base_prob = 30.0  # Default base probability
        
        if disease == 'Hypertension':
            if data.get('Systolic_BP', 120) > 140 or data.get('Diastolic_BP', 80) > 90:
                return 70.0
            elif data.get('Systolic_BP', 120) > 130 or data.get('Diastolic_BP', 80) > 85:
                return 50.0
        
        # ...existing code...
        
        return base_prob
    
    def _get_recommended_actions(self, disease: str, risk_probability: float) -> List[str]:
        """Return recommended actions based on disease and risk level"""
        # ...existing code...
        actions = []
        
        # Low risk (0-30%)
        if risk_probability <= 30:
            if disease == 'Hypertension':
                actions = ["Maintain healthy diet", "Regular exercise", "Annual blood pressure check"]
            # ...existing code...
            
        # Medium risk (30-70%)
        elif risk_probability <= 70:
            # ...existing code...
            pass
            
        # High risk (70-100%)
        else:
            # ...existing code...
            pass
        
        return actions

    def save_models(self):
        """Save the unified model"""
        joblib.dump(self.unified_model, 'unified_disease_model.pkl')
        print("Saved unified disease model")

    def load_models(self):
        """Load the unified model"""
        try:
            if os.path.exists('unified_disease_model.pkl'):
                self.unified_model = joblib.load('unified_disease_model.pkl')
                print("Loaded unified disease model")
            else:
                print("Unified model not found, creating default")
                self._create_unified_model()
        except Exception as e:
            print(f"Error loading unified model: {e}")
            self._create_unified_model()

# Function to be called from the Streamlit app
def predict_diseases(patient_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    try:
        predictor = MultiDiseasePredictor()
        
        # Check if unified model exists
        if not os.path.exists('unified_disease_model.pkl'):
            print("Unified model not found. Creating default model...")
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
