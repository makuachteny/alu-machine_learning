-- Script displays the max temperature of each state ordered by state name

SELECT MAX(value) AS max_temp FROM temperatures
GROUP BY state;