CREATE TABLE conditions (
    start DATETIME,
    stop DATETIME,
    patient VARCHAR(64),
    encounter VARCHAR(64),
    `system` VARCHAR(50),
    code VARCHAR(20),
    description VARCHAR(100),
    FOREIGN KEY (patient) REFERENCES patients(id),
    FOREIGN KEY (encounter) REFERENCES encounters(id)
);
