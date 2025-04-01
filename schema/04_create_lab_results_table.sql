CREATE TABLE lab_results (
    lab_id VARCHAR(50) PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    encounter_id VARCHAR(50) NOT NULL,
    test_code VARCHAR(20) NOT NULL,
    test_description TEXT NOT NULL,
    result_value DECIMAL(10,2) NOT NULL,
    units VARCHAR(20) NOT NULL,
    test_date DATE NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
    FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id) ON DELETE CASCADE
);
