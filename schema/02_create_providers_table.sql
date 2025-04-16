CREATE TABLE providers (
    id VARCHAR(64) PRIMARY KEY AUTO_INCREMENT;
    organization VARCHAR(100),
    name VARCHAR(100),
    gender VARCHAR(10),
    speciality VARCHAR(100),
    address VARCHAR(100),
    city VARCHAR(50),
    state VARCHAR(20),
    zip VARCHAR(10),
    lat DECIMAL(10, 7),
    lon DECIMAL(10, 7),
    encounters INT,
    procedures INT
);
