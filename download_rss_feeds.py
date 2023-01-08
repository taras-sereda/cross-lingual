import asyncio
import random
import time
from pathlib import Path

import aiohttp
import tqdm

from podindex import SessionLocal
from podindex.crud import get_podcasts

data_root = Path('podindex-data')
if not data_root.exists():
    data_root.mkdir(exist_ok=True)


async def get_rss(sess, podcast):
    output_path = data_root.joinpath(f'{podcast.podcastGuid}.xml')
    try:
        sleep_prob = random.random()
        if sleep_prob > 0.8:
            await asyncio.sleep(random.random()/3)

        async with sess.get(podcast.originalUrl) as resp:
            if resp.status != 200:
                return
            text = await resp.text()
        with open(output_path, 'w') as f:
            f.write(text)
    except (
            Exception,
            aiohttp.ClientConnectorCertificateError,
            aiohttp.ClientConnectionError
    ) as e:
        print(podcast.originalUrl, str(e))


def filter_already_downloaded(podcasts):
    res = []
    for item in podcasts:
        output_path = data_root.joinpath(f'{item.podcastGuid}.xml')
        if output_path.exists() and output_path.stat().st_size > 1000:
            continue
        res.append(item)
    return res


async def main():
    db = SessionLocal()
    podcasts = get_podcasts(db, -1)
    print(f'Total amount of podcasts {len(podcasts)}')
    podcasts = filter_already_downloaded(podcasts)
    print(f'TODO. Total amount of podcasts {len(podcasts)}')
    size = 300
    async with aiohttp.ClientSession() as sess:
        for chunk_idx in range(0, len(podcasts), size):
            tasks_list = [asyncio.create_task(get_rss(sess, podcast)) for podcast in podcasts[chunk_idx: chunk_idx + size]]
            for t in tqdm.tqdm(tasks_list):
                await t


if __name__ == '__main__':
    start = time.time()
    asyncio.run(main())
    print(time.time() - start)
