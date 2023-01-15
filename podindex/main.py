import csv
from datetime import datetime
from itertools import groupby

from podindex import SessionLocal, data_root
from podindex.crud import get_podcasts

if __name__ == '__main__':
    output_path = data_root.parent.joinpath('multileng_podcasters2.csv')
    if output_path.exists():
        exit()

    start = datetime.now()
    db = SessionLocal()
    podcasts = get_podcasts(db, limit=-1)
    podcasts = sorted(podcasts, key=lambda x: x.itunesAuthor)
    multileng_podcasters = []
    for author, author_podcasts in groupby(podcasts, key=lambda x: x.itunesAuthor):
        author_podcasts = list(author_podcasts)
        author_languages = set([itm.language.lower().split('-')[0] for itm in author_podcasts])

        # not interested in single language podcasters
        if len(author_languages) == 1:
            continue
        # interested only in podcasters who already do english
        if 'en' not in author_languages:
            continue
        multileng_podcasters.append({
            'itunesAuthor': author,
            'languages': author_languages})

    multileng_podcasters = sorted(multileng_podcasters, key=lambda x: len(x['languages']), reverse=True)
    for group_size, group_items in groupby(multileng_podcasters, key=lambda x: len(x['languages'])):
        print(group_size)
    print(f'Total number of multilingual podcasters {len(multileng_podcasters)}')

    with open(output_path, 'w', newline='') as fd:
        writer = csv.DictWriter(fd, fieldnames=multileng_podcasters[0].keys())
        writer.writeheader()
        writer.writerows(multileng_podcasters)

    end = datetime.now()
    print(start, end)
