CREATE TABLE patients (
    patient_id VARCHAR(50) PRIMARY KEY,
    birthdate DATE NOT NULL,
    deathdate DATE NULL,
    gender VARCHAR(50) NOT NULL,
    race VARCHAR(50),
    ethnicity VARCHAR(50),
    address VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(50),
    zip VARCHAR(20),
    income DECIMAL(10,2)
);
