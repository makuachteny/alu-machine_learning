-- Import the table dump
-- Ensure to unzip names.sql.zip and use the following command in your MySQL environment:
-- mysql -u your_username -p your_database < path_to_unzipped_names.sql
-- Create an index on the first letter of the name column
CREATE INDEX idx_name_first ON names (SUBSTRING(name, 1, 1));