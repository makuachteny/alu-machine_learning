-- Import the database dump from hbtn_0c_0 and display the average temperature (Fahrenheit) by city ordered by temperature (descending).

SELECT city, AVG(temperatures) AS avg_temp
FROM temperatures
GROUP BY city
ORDER BY avg_temp DESC;