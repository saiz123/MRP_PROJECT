CREATE TABLE careplans (
    id VARCHAR(64) PRIMARY KEY,
    start DATETIME,
    stop DATETIME,
    patient VARCHAR(64),
    encounter VARCHAR(64),
    code VARCHAR(20),
    description VARCHAR(100),
    reasoncode VARCHAR(20),
    reasondescription VARCHAR(100),
    FOREIGN KEY (patient) REFERENCES patients(id),
    FOREIGN KEY (encounter) REFERENCES encounters(id)
);
