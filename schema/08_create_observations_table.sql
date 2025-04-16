CREATE TABLE observations (
    date DATETIME,
    patient VARCHAR(64),
    encounter VARCHAR(64),
    category VARCHAR(50),
    code VARCHAR(20),
    description VARCHAR(100),
    value VARCHAR(100),
    units VARCHAR(20),
    type VARCHAR(50),
    observation_group VARCHAR(50), -- Added to group related observations (e.g., 'CARDIOVASCULAR', 'METABOLIC')
    FOREIGN KEY (patient) REFERENCES patients(id),
    FOREIGN KEY (encounter) REFERENCES encounters(id)
);
