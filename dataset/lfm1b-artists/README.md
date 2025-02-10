INTERACTIONs DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file lfm1b-artists.inter comprising the artists that users have listened.
Each record/line in the file has the following fields: user_id, artists_id, timestamp

user_id: the id of the users and its type is token. 
artists_id: the id of the artists and its type is token.
timestamp: the UNIX timestamp of the records, and its type is float.
num_repeat: the number of times that the user has listened this artists, and its type is float.

ARTIST INFORMATION DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file lfm1b-artists.item comprising the attributes of the artists.
Each record/line in the file has the following fields: artists_id, name

artists_id: the id of the artists, and its type is token.
name: the name of each artist, and its type is token_seq.

USER INFORMATION DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file lfm1b-albums.user comprising the attributes of the users.
Each record/line in the file has the following fields: user_id, country, age, gender, playcount, registered_timestamp, novelty_artist_avg_month, novelty_artist_avg_6months, novelty_artist_avg_year, mainstreaminess_avg_month, mainstreaminess_avg_6months, mainstreaminess_avg_year, mainstreaminess_global, cnt_listeningevents, cnt_distinct_tracks, cnt_distinct_artists, cnt_listeningevents_per_week, relative_le_per_weekday1, relative_le_per_weekday2, relative_le_per_weekday3, relative_le_per_weekday4, relative_le_per_weekday5, relative_le_per_weekday6, relative_le_per_weekday7, relative le per hour0, relative le per hour1, relative le per hour2, relative le per hour3, relative le per hour4, relative le per hour5, relative le per hour6, relative le per hour7, relative le per hour8, relative le per hour9, relative le per hour10, relative le per hour11, relative le per hour12, relative le per hour13, relative le per hour14, relative le per hour15, relative le per hour16, relative le per hour17, relative le per hour18, relative le per hour19, relative le per hour20, relative le per hour21, relative le per hour22, relative le per hour23:float

user_id: the id of the users, and its type is token.
country: the country of the users, and its type is token.
age: the age of the users, and its type is float.
gender: the gender of the users, and its type is token.
playcount: the number of playcount, and its type is float.
registered_timestamp: the UNIX timestamp of users registering, and its type is float.
novelty_artist_avg_month: the percentage of new artists listened to, averaged over time windows of 1 month, and its type is float.
novelty_artist_avg_6months: the percentage of new artists listened to, averaged over time windows of 6 months, and its type is float.
novelty_artist_avg_year: the percentage of new artists listened to, averaged over time windows of 1 year, and its type is float.
mainstreaminess_avg_month: overlap between the user's listening history and an aggregate listening history of all users, averaged over time windows of 1 month, and its type is float.
mainstreaminess_avg_6months: overlap between the user's listening history and an aggregate listening history of all users, averaged over time windows of 6 months, and its type is float.
mainstreaminess_avg_year: overlap between the user's listening history and an aggregate listening history of all users, averaged over time windows of 1 year, and its type is float.
mainstreaminess_global: overlap between the user's listening history and an aggregate listening history of all users, averaged over the entire period of the user's activity on Last.fm, and its type is float.
cnt_listeningevents: total number of the user's listening events (playcounts) included in the dataset, and its type is float.
cnt_distinct_tracks: number of unique tracks listened to by the user, and its type is float.
cnt_distinct_artists: number of unique artists listened to by the user, and its type is float.
cnt_listeningevents_per_week: average number of listening events per week, and its type is float.
relative_le_per_weekday[1-7]: fraction of listening events for each weekday (starting on Monday) among all weekly plays, averaged over the user's entire listening history, and its type is float.
relative le per hour[0-23]: fraction of listening events for each hour of the day (starting with the time span 0:00-0:59) among all 24 hours, averaged over the user's entire listening history, and its type is float.
