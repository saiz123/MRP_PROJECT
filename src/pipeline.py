import pandas as pd

# Load data
patients = pd.read_csv("files/patients.csv")
observations = pd.read_csv("files/observations.csv")
conditions = pd.read_csv("files/conditions.csv")

# STEP 1: Select relevant patient info
patients_df = patients[["Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"]]

# STEP 2: Filter for relevant observations
relevant_obs_codes = {
    "8480-6": "Systolic_BP",         # Systolic BP
    "8462-4": "Diastolic_BP",        # Diastolic BP
    "39156-5": "BMI",                # Body Mass Index
    "72166-2": "Smoking_Status"      # Smoking status
}

obs_filtered = observations[observations["CODE"].isin(relevant_obs_codes.keys())]
obs_filtered["ObservationName"] = obs_filtered["CODE"].map(relevant_obs_codes)

# Pivot so each patient has one row
obs_pivot = obs_filtered.pivot_table(
    index="PATIENT",
    columns="ObservationName",
    values="VALUE",
    aggfunc="last"
).reset_index()

obs_pivot.rename(columns={"PATIENT": "Id"}, inplace=True)

# STEP 3: Binary label for hypertension
hypertension_codes = ["I10"]
conditions["has_hypertension"] = conditions["CODE"].isin(hypertension_codes).astype(int)
condition_flags = conditions.groupby("PATIENT")["has_hypertension"].max().reset_index()
condition_flags.rename(columns={"PATIENT": "Id"}, inplace=True)

# STEP 4: Add all diseases as columns
disease_df = conditions[["PATIENT", "DESCRIPTION"]].drop_duplicates()
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
                       .merge(condition_flags, on="Id", how="left") \
                       .merge(disease_pivot, on="Id", how="left")

merged_df["has_hypertension"] = merged_df["has_hypertension"].fillna(0)
merged_df[disease_pivot.columns[1:]] = merged_df[disease_pivot.columns[1:]].fillna(0)

# STEP 6: Feature engineering
merged_df["AGE"] = pd.to_datetime("today").year - pd.to_datetime(merged_df["BIRTHDATE"]).dt.year

# Optional: Set primary disease label
if "Hypertensive disorder, systemic arterial (disorder)" in merged_df.columns:
    merged_df["disease"] = merged_df["Hypertensive disorder, systemic arterial (disorder)"]
else:
    merged_df["disease"] = merged_df["has_hypertension"]

# Final selection
final_df = merged_df[[
    "AGE", "GENDER", "RACE", "ETHNICITY",
    "Systolic_BP", "Diastolic_BP", "BMI", "Smoking_Status",
    "has_hypertension", "disease"
] + list(disease_pivot.columns[1:])]

# One-hot encode categoricals
final_df = pd.get_dummies(final_df, columns=["GENDER", "RACE", "ETHNICITY", "Smoking_Status"], drop_first=True)

# Drop missing values
final_df = final_df.dropna()

# Save processed data
final_df.to_csv("synthea_processed.csv", index=False)
print("✅ Preprocessing complete. File saved: synthea_processed.csv")
