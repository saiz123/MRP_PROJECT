import pandas as pd
import mysql.connector
import os
from datetime import datetime
import numpy as np

# Database configuration
DB_PARAMS = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "password",
    "database": "mrp_new"
}

# CSV File Paths
CSV_FOLDER = "files"

# CSV to MySQL Table Mapping
CSV_TABLE_MAPPING = {
    "patients.csv": "patients",
    "medical_history.csv": "medical_history",
    "prescriptions.csv": "prescriptions",
    "lab_results.csv": "lab_results",
    "treatment_recommendations.csv": "treatment_recommendations",
    "encounters.csv": "encounters"
}

# CSV Column Renaming to Match MySQL Schema
CSV_TABLE_COLUMN_MAPPING = {
    "patients.csv": {
        "Id": "patient_id", "BIRTHDATE": "birthdate", "DEATHDATE": "deathdate",
        "GENDER": "gender", "RACE": "race", "ETHNICITY": "ethnicity",
        "ADDRESS": "address", "CITY": "city", "STATE": "state",
        "ZIP": "zip", "INCOME": "income"
    },
    "encounters.csv": {
        "Id": "encounter_id", "PATIENT": "patient_id", "ORGANIZATION": "organization",
        "PROVIDER": "provider", "PAYER": "payer", "ENCOUNTERCLASS": "encounter_class",
        "CODE": "code", "DESCRIPTION": "description", "BASE_ENCOUNTER_COST": "base_encounter_cost",
        "TOTAL_CLAIM_COST": "total_claim_cost", "PAYER_COVERAGE": "payer_coverage",
        "REASONCODE": "reason_code", "REASONDESCRIPTION": "reason_description",
        "START": "start_date", "STOP": "stop_date"
    },
    "medical_history.csv": {
        "Id": "history_id", "PATIENT": "patient_id", "ENCOUNTER": "encounter_id",
        "CODE": "condition_code", "DESCRIPTION": "condition_description",
        "DATE": "diagnosis_date"
    },
    "lab_results.csv": {
        "Id": "lab_id", "PATIENT": "patient_id", "ENCOUNTER": "encounter_id",
        "CODE": "test_code", "DESCRIPTION": "test_description",
        "VALUE": "result_value", "UNITS": "units", "DATE": "test_date"
    },
    "prescriptions.csv": {
        "Id": "prescription_id", "PATIENT": "patient_id", "ENCOUNTER": "encounter_id",
        "CODE": "medication_code", "DESCRIPTION": "medication_description",
        "START": "start_date", "STOP": "end_date", "BASE_COST": "base_cost",
        "TOTAL_COST": "total_cost"
    },
    "treatment_recommendations.csv": {
        "Id": "recommendation_id", "PATIENT": "patient_id",
        "TREATMENT": "recommended_treatment", "CONFIDENCE_SCORE": "confidence_score",
        "DATE": "generated_date"
    }
}

def fix_date_format(date_str):
    """ Convert date string to MySQL-compatible format or return None if invalid. """
    if not isinstance(date_str, str) or date_str.strip() == "":
        return None  # Skip empty values
    
    try:
        # Handle ISO 8601 format (e.g., '2020-02-04T04:52:54Z')
        if "T" in date_str and "Z" in date_str:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")

        # Handle ISO 8601 without 'Z' (e.g., '2020-02-04T04:52:54')
        if "T" in date_str:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")

        # Handle standard date format 'YYYY-MM-DD'
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d %H:%M:%S")

    except ValueError:
        return None  # Skip invalid values


def clean_data(df):
    """ Clean and format the DataFrame before inserting into MySQL. """
    df = df.replace({np.nan: None})  # Replace NaN with None

    for col in df.columns:
        if "date" in col.lower():
            df[col] = df[col].astype(str).apply(fix_date_format)
        elif df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()  # Ensure it's a string before applying .str
    return df

def get_table_columns(cursor, table_name):
    """ Fetch column names from MySQL table. """
    cursor.execute(f"SHOW COLUMNS FROM {table_name}")
    return {row[0] for row in cursor.fetchall()}

def insert_into_mysql(df, table_name, csv_file, cursor):
    """ Insert cleaned data into MySQL table. """
    if csv_file in CSV_TABLE_COLUMN_MAPPING:
        df = df.rename(columns=CSV_TABLE_COLUMN_MAPPING[csv_file])

    table_columns = get_table_columns(cursor, table_name)
    df = df[[col for col in df.columns if col in table_columns]]

    df = df.replace({np.nan: None})  # Ensure NaN is replaced with None

    cols = ",".join(df.columns)
    values = ",".join(["%s"] * len(df.columns))

    sql = f"INSERT INTO {table_name} ({cols}) VALUES ({values})"

    data = []
    for row in df.itertuples(index=False, name=None):
        data.append(tuple(row))

    try:
        cursor.executemany(sql, data)
        print(f"Inserted {len(df)} rows into {table_name}.")
    except mysql.connector.Error as err:
        print(f"Error inserting data from {csv_file} into {table_name}: {err}")

def truncate_tables(cursor):
    """ Truncate all tables in the correct order. """
    truncate_order = [
        "treatment_recommendations", "prescriptions", "lab_results",
        "medical_history", "encounters", "patients"
    ]

    # Disable foreign key checks to allow truncation
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
    print("Foreign key checks disabled.")

    for table_name in truncate_order:
        cursor.execute(f"TRUNCATE TABLE {table_name}")
        print(f"Truncated table: {table_name}")

    # Re-enable foreign key checks after truncation
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
    print("Foreign key checks enabled.")

def process_csv_files():
    """ Main function to process CSV files and insert data into MySQL. """
    try:
        conn = mysql.connector.connect(**DB_PARAMS)
        cursor = conn.cursor()

        # Truncate all tables before inserting new data
        truncate_tables(cursor)

        for csv_file, table_name in CSV_TABLE_MAPPING.items():
            file_path = os.path.join(CSV_FOLDER, csv_file)
            if os.path.exists(file_path):
                print(f"Processing {csv_file} into {table_name}...")
                df = pd.read_csv(file_path)
                df = clean_data(df)
                insert_into_mysql(df, table_name, csv_file, cursor)

        conn.commit()
        cursor.close()
        conn.close()

        print("Data insertion completed successfully!")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

if __name__ == "__main__":
    process_csv_files()
