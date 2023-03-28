# useful queries
1. Select all hosts: 
   ```
   SELECT 
       host,
       COUNT(*) AS c
   FROM podcasts
   WHERE dead = 0
   GROUP BY host
   ORDER BY c DESC;
   ```

2. Select all active podcasts: 
   ```
   SELECT 
       title,
       newestItemPubdate,
       DATE(newestItemPubdate, 'unixepoch', 'localtime')
   FROM podcasts
   WHERE DATE(newestItemPubdate, 'unixepoch', 'localtime') > DATE('2022-12-01');
   ```

3. Serve podindex via datasette: `datasette -h 0.0.0.0 -i /data/crosslingual-data/podindex-data/podcastindex_feeds.db --inspect-file /data/crosslingual-data/podindex-data/counts.json --setting max_returned_rows 10 --setting allow_download off --memory`

4. Select all multilingual podcasters: 
    ```
    SELECT 
    itunesOwnerName, count(*) as num_podcasts
    , group_concat(distinct 
                   CASE 
                       WHEN language LIKE '%-%' THEN LOWER(SUBSTR(language, 1, INSTR(language, '-') - 1))
                       ELSE LOWER(language)
                  END) langs
    , count(distinct CASE 
                       WHEN language LIKE '%-%' THEN LOWER(SUBSTR(language, 1, INSTR(language, '-') - 1))
                       ELSE LOWER(language)
                  END) as num_langs
    , min(originalUrl) as url
    
    FROM podcasts 
    WHERE length(itunesOwnerName) > 1 and length(language) > 1
    GROUP BY itunesOwnerName 
    HAVING num_langs > 1
    ORDER BY num_langs DESC, num_podcasts DESC;
    ```