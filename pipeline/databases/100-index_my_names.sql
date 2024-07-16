-- Drop the existing index if it exists
DROP INDEX IF EXISTS idx_name_first ON names;
-- Create an index on the first letter of the name column
CREATE INDEX idx_name_first ON names (SUBSTRING(name, 1, 1));