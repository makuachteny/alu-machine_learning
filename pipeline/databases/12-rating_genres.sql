-- Script lists all genres in the database hbtn_0d_tvshows database by their rating

SELECT tv_genres.name, SUM(rate) AS rating
FROM tv_genres
INNER JOIN tv_show_genres ON tv_genres.id = tv_show_ratings.genre_id
GROUP BY tv_genres.name
ORDER BY rating DESC;
