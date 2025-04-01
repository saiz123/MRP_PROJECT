CREATE TABLE prescriptions (
    prescription_id VARCHAR(50) PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    encounter_id VARCHAR(50) NOT NULL,
    medication_code VARCHAR(20) NOT NULL,
    medication_description TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NULL,
    base_cost DECIMAL(10,2),
    total_cost DECIMAL(10,2),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
    FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id) ON DELETE CASCADE
);
