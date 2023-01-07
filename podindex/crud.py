import datetime
import time
from itertools import groupby
from sqlalchemy.orm import Session

from .models import Podcast

DATE_THRESHOLD = time.mktime(datetime.date(2022, 12, 1).timetuple())


def get_podcasts(db: Session, limit=1000) -> list[Podcast]:
    if limit < 0:
        limit = int(1e9)
    podcasts = db.query(Podcast).filter(Podcast.dead == 0,
                                        Podcast.episodeCount >= 10,
                                        Podcast.newestItemPubdate > DATE_THRESHOLD).limit(limit).all()
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
