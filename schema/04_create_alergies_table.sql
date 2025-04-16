CREATE TABLE allergies (
    start DATETIME,
    stop DATETIME,
    patient VARCHAR(64),
    encounter VARCHAR(64),
    code VARCHAR(20),
    `system` VARCHAR(50),
    description VARCHAR(100),
    `type` VARCHAR(50),
    `category` VARCHAR(50),
    reaction1 VARCHAR(50),
    description1 VARCHAR(100),
    severity1 VARCHAR(20),
    reaction2 VARCHAR(50),
    description2 VARCHAR(100),
    severity2 VARCHAR(20),
    FOREIGN KEY (patient) REFERENCES patients(id),
    FOREIGN KEY (encounter) REFERENCES encounters(id)
);
