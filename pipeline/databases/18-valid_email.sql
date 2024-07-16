-- Validate email
DELIMITER //

CREATE TRIGGER email_update_trigger BEFORE UPDATE ON users
FOR EACH ROW
BEGIN 
    -- Check if the email has changed
    IF STRCMP(old.email, new.email) <> 0 THEN
    -- Reset the valid_email flag
    SET new.valid_email = 0;
END IF;
END;

//
DELIMITER ;