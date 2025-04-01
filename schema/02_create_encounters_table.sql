CREATE TABLE encounters (
    encounter_id VARCHAR(50) PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    organization VARCHAR(100),
    provider VARCHAR(100),
    payer VARCHAR(100),
    encounter_class VARCHAR(50),
    code VARCHAR(20),
    description TEXT,
    base_encounter_cost DECIMAL(10,2),
    total_claim_cost DECIMAL(10,2),
    payer_coverage DECIMAL(10,2),
    reason_code VARCHAR(20),
    reason_description TEXT,
    start_date DATETIME NOT NULL,
    stop_date DATETIME NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);
