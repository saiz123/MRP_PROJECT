CREATE TABLE medications (
    start DATETIME,
    stop DATETIME,
    patient VARCHAR(64),
    payer VARCHAR(100),
    encounter VARCHAR(64),
    code VARCHAR(20),
    description VARCHAR(100),
    base_cost DOUBLE,
    payer_coverage DOUBLE,
    dispenses INT,
    totalcost DOUBLE,
    reasoncode VARCHAR(20),
    reasondescription VARCHAR(100),
    FOREIGN KEY (patient) REFERENCES patients(id),
    FOREIGN KEY (encounter) REFERENCES encounters(id)
);
