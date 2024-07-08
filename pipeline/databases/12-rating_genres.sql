-- Script lists all genres in the database hbtn_0d_tvshows database by their rating

SELECT tv_genres.name, SUM(rate) AS rating
FROM tv_genres
<<<<<<< HEAD
LEFT JOIN tv_show_ratings ON tv_genres.id = tv_show_ratings.show_id
=======
LEFT JOIN tv_show_ratings ON tv_genres.id = tv_show_ratings.genre_id
>>>>>>> 1fbff2eba8d293d3538028bafe4aa9e63a042e2c
GROUP BY tv_genres.name
ORDER BY rating DESC;
