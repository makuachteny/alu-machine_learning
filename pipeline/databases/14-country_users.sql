-- Script creates a table users with attributes id, email, name and country

-- in table users
-- -- insert:
-- id 
-- email
-- name
-- country(enums: 'US', "CO", 'TN') - not null, default 'US'
DROP TABLE IF EXISTS users;
CREATE TABLE IF NOT EXISTS users(
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US',
    UNIQUE (email)
);