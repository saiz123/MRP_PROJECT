CREATE TABLE encounters (
    id VARCHAR(64) PRIMARY KEY,
    start DATETIME,
    stop DATETIME,
    patient VARCHAR(64),
    organization VARCHAR(100),
    provider VARCHAR(64),
    payer VARCHAR(100),
    encounterclass VARCHAR(50),
    code VARCHAR(20),
    description VARCHAR(100),
    base_encounter_cost DOUBLE,
    total_claim_cost DOUBLE,
    payer_coverage DOUBLE,
    reasoncode VARCHAR(20),
    reasondescription VARCHAR(100),
    FOREIGN KEY (patient) REFERENCES patients(id),
    FOREIGN KEY (provider) REFERENCES providers(id)
);
