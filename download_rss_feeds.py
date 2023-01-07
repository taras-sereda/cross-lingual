import asyncio

from podindex.crud import get_podcasts
from podindex import SessionLocal
from pathlib import Path
import requests

db = SessionLocal()
podcasts = get_podcasts(db, -1)

data_root = Path('podindex-data')
if not data_root.exists():
    data_root.mkdir(exist_ok=True)


# for p in tqdm(podcasts):
#     try:
#         output_path = data_root.joinpath(f'{p.podcastGuid}.xml')
#         if output_path.exists() and output_path.stat().st_size > 1000:
#             continue
#
#         res = requests.get(p.url, timeout=10)
#         if res.status_code != 200:
#             continue
#
#         with open(output_path, 'w') as f:
#             f.write(res.text)
#     except Exception as e:
#         print(e)


async def get_rss(podcast):
    try:
        output_path = data_root.joinpath(f'{podcast.podcastGuid}.xml')
        if output_path.exists() and output_path.stat().st_size > 1000:
            return

        res = requests.get(podcast.url, timeout=10)
        if res.status_code != 200:
            return

        with open(output_path, 'w') as f:
            f.write(res.text)
    except Exception as e:
        print(e)


async def main():
    tasks_list = [get_rss(podcast) for podcast in podcasts]
    for future in asyncio.as_completed(tasks_list):
        res = await future


if __name__ == '__main__':
    asyncio.run(main())
