-- Creating the trigger
DELIMITER //

CREATE TRIGGER update_quantity_after_order
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE products
    SET quantity = quantity - NEW.number
    WHERE id = NEW.item_name;
END;

//

DELIMITER ; -- Resets the delimiter to the default semicolon
