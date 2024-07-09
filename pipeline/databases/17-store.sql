-- Creating the trigger

DELIMITTER // --Allows the trigger creation with multiple statements

CREATE TRIGGER update_quantity_after_order
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE products
    SET quantity = quantity - NEW.number
    WHERE id = NEW.product_id;
END;
// DELIMITTER ; --Resets the delimiter to the default semicolon