import pandas as pd

# Load data
patients = pd.read_csv("files/patients.csv")
observations = pd.read_csv("files/observations.csv")
conditions = pd.read_csv("files/conditions.csv")

# STEP 1: Select relevant patient info
patients_df = patients[["Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"]]

# STEP 2: Define relevant observation codes
relevant_obs_codes = {
    # Cardiovascular
    "8480-6": "Systolic_BP",
    "8462-4": "Diastolic_BP",
    "8867-4": "Heart_Rate",
    "2093-3": "Cholesterol_Total",
    "2085-9": "HDL",
    "2089-1": "LDL",

    # Metabolic
    "2339-0": "Blood_Glucose",
    "39156-5": "BMI",
    "4548-4": "HbA1c",
    "8280-0": "Waist_Circumference",

    # Respiratory
    "20150-9": "FEV1",
    "19876-2": "FVC",
    "59408-5": "Oxygen_Saturation",
    "9279-1": "Respiratory_Rate",

    # Screening
    "10839-9": "Breast_Cancer_Screening",
    "30525-0": "Lung_Cancer_Screening",
    "58413-6": "Colon_Cancer_Screening",

    # Infectious Disease
    "8310-5": "Body_Temperature",
    "6690-2": "WBC_Count",
    "1988-5": "CRP",
    
    # Other
    "72166-2": "Smoking_Status",
    "8302-2": "Height",
    "29463-7": "Weight",
    "55284-4": "Blood_Pressure"
}

# Define disease categories and their codes
disease_categories = {
    'CARDIOVASCULAR': ['I10', 'I21', 'I25.2'],  # Hypertension, Heart Attack
    'METABOLIC': ['E11', 'E66'],  # Diabetes, Obesity
    'RESPIRATORY': ['J45', 'J44'],  # Asthma, COPD
    'ONCOLOGY': ['C50', 'C34', 'C18'],  # Breast, Lung, Colon Cancer
    'INFECTIOUS': ['B15', 'J10', 'J11']  # Hepatitis, Influenza, Flu
}

obs_filtered = observations[observations["CODE"].isin(relevant_obs_codes.keys())]
obs_filtered["ObservationName"] = obs_filtered["CODE"].map(relevant_obs_codes)

# Pivot observations
obs_pivot = obs_filtered.pivot_table(
    index="PATIENT",
    columns="ObservationName",
    values="VALUE",
    aggfunc="last"
).reset_index()

obs_pivot.rename(columns={"PATIENT": "Id"}, inplace=True)

# STEP 3: Create disease flags for each category
disease_flags = pd.DataFrame()
disease_flags["Id"] = conditions["PATIENT"].unique()

for category, codes in disease_categories.items():
    disease_flags[category] = conditions[conditions["PATIENT"].isin(disease_flags["Id"])].apply(
        lambda x: 1 if x["CODE"] in codes else 0, axis=1
    ).groupby(conditions["PATIENT"]).max()

# STEP 4: Add individual disease columns
disease_df = conditions[["PATIENT", "DESCRIPTION", "CODE"]].drop_duplicates()
disease_df["value"] = 1

disease_pivot = disease_df.pivot_table(
    index="PATIENT",
    columns="DESCRIPTION",
    values="value",
    aggfunc="max",
    fill_value=0
).reset_index()

disease_pivot.rename(columns={"PATIENT": "Id"}, inplace=True)

# STEP 5: Merge all data
merged_df = patients_df.merge(obs_pivot, on="Id", how="left") \
                      .merge(disease_flags, on="Id", how="left") \
                      .merge(disease_pivot, on="Id", how="left")

# Fill missing values with appropriate defaults
merged_df = merged_df.fillna({
    'CARDIOVASCULAR': 0,
    'METABOLIC': 0,
    'RESPIRATORY': 0,
    'ONCOLOGY': 0,
    'INFECTIOUS': 0
})

# STEP 6: Feature engineering
merged_df["AGE"] = pd.to_datetime("today").year - pd.to_datetime(merged_df["BIRTHDATE"]).dt.year

# Calculate BMI if missing but height and weight are present
mask = merged_df["BMI"].isna() & merged_df["Height"].notna() & merged_df["Weight"].notna()
merged_df.loc[mask, "BMI"] = merged_df.loc[mask, "Weight"] / ((merged_df.loc[mask, "Height"]/100) ** 2)

# Final feature selection
clinical_features = [
    "AGE", "Systolic_BP", "Diastolic_BP", "Heart_Rate", 
    "Cholesterol_Total", "HDL", "LDL", "Blood_Glucose", "BMI", 
    "HbA1c", "Waist_Circumference", "FEV1", "FVC", 
    "Oxygen_Saturation", "Respiratory_Rate", "Body_Temperature", 
    "WBC_Count", "CRP"
]

categorical_features = ["GENDER", "RACE", "ETHNICITY", "Smoking_Status"]

disease_categories = ["CARDIOVASCULAR", "METABOLIC", "RESPIRATORY", "ONCOLOGY", "INFECTIOUS"]

final_df = merged_df[clinical_features + categorical_features + disease_categories]

# One-hot encode categoricals
final_df = pd.get_dummies(final_df, columns=categorical_features, drop_first=True)

# Drop rows with too many missing values (more than 50%)
final_df = final_df.dropna(thresh=len(final_df.columns)//2)

# Fill remaining missing values with median for numeric columns
numeric_cols = final_df.select_dtypes(include=['float64', 'int64']).columns
final_df[numeric_cols] = final_df[numeric_cols].fillna(final_df[numeric_cols].median())

# Save processed data
final_df.to_csv("synthea_processed.csv", index=False)
print("✅ Preprocessing complete. File saved: synthea_processed.csv")
