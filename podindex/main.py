import csv
from itertools import groupby
from sqlalchemy.orm import Session

from podindex import SessionLocal
from podindex.models import Podcast
from datetime import datetime


def get_podcasts(db: Session, limit=1000) -> list[Podcast]:
    if limit < 0:
        limit = int(1e9)
    podcasts = db.query(Podcast).filter(Podcast.dead == 0).limit(limit).all()
    return podcasts


def print_podcast_by_lang(db: Session):
    podcasts = get_podcasts(db, limit=-1)
    podcasts = sorted(podcasts, key=lambda x: x.itunesAuthor)
    for author, author_podcasts in groupby(podcasts, key=lambda x: x.itunesAuthor):
        author_podcasts = list(author_podcasts)
        author_languages = set([itm.language.lower().split('-')[0] for itm in author_podcasts])

        # not interested in single language podcasters
        if len(author_languages) == 1:
            continue

        author_podcasts = sorted(author_podcasts, key=lambda x: x.language.lower().split('-')[0])
        print(f'Author: {author}, num languages: {len(author_languages)}')
        for lang, lang_podcasts in groupby(author_podcasts, key=lambda x: x.language.lower().split('-')[0]):
            lang_podcasts = list(lang_podcasts)
            print('*' * 10)
            print(f'Lang: {lang}')
            for pod in lang_podcasts:
                print(pod.title)
            print('')
        print('~' * 40)


if __name__ == '__main__':
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
    with open('multileng_podcasters.csv', 'w', newline='') as fd:
        writer = csv.DictWriter(fd, fieldnames=multileng_podcasters[0].keys())
        writer.writeheader()
        writer.writerows(multileng_podcasters)

    end = datetime.now()
    print(start, end)
