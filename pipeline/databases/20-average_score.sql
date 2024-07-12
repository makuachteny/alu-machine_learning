-- Create a stored procedure to compute the average score
-- for a user and update the average_score column in the users table.
DELIMITER // 

CREATE PROCEDURE ComputeAverageScoreForUser (
    IN user_id INT
)
BEGIN
    DECLARE avg_score FLOAT;

-- Calculate average score for the user
SELECT AVG(score) INTO avg_score
FROM corrections
WHERE user_id = user_id;
-- Update the average_score column in the users table
UPDATE users
SET average_score = avg_score
WHERE id = user_id;
END;

//

DELIMITER;