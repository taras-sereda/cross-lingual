# useful queries
1. Select all hosts: ```select host, count(*) as c from podcasts where dead == 0 group by host order by c desc```

2. Select all active podcasts: ```select title, newestItemPubdate, date(newestItemPubdate, 'unixepoch', 'localtime') from podcasts where date(newestItemPubdate, 'unixepoch', 'localtime') > date('2022-12-01');```