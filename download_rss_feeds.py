import argparse
import asyncio
import concurrent.futures
import csv
import os
import random
import time

import aiohttp
import requests
import tqdm

from podindex import SessionLocal, podidx_data_root
from podindex.crud import get_podcasts


async def get_rss(sess, podcast):
    output_path = podidx_data_root.joinpath(f'rss/{podcast.podcastGuid}.xml')
    try:
        sleep_prob = random.random()
        if sleep_prob > 0.8:
            await asyncio.sleep(random.random() / 3)

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


def get_rss_sequentially(podcast):
    output_path = podidx_data_root.joinpath(f'rss/{podcast.podcastGuid}.xml')
    try:
        resp = requests.get(podcast.originalUrl)
        if resp.status_code != 200:
            return podcast.podcastGuid, False
        text = resp.text
        with open(output_path, 'w') as f:
            f.write(text)

    except Exception as e:
        return podcast.podcastGuid, False

    return podcast.podcastGuid, True


def filter_already_downloaded_or_broken(podcasts, broke_csv_path=None):
    broken_guids = set()
    if broke_csv_path:
        with open(broke_csv_path, 'r', newline='') as fd:
            csv_reader = csv.reader(fd)
            broken_guids = set([line[0] for line in csv_reader])

    res = []
    for item in podcasts:
        output_path = podidx_data_root.joinpath(f'rss/{item.podcastGuid}.xml')
        if output_path.exists() and output_path.stat().st_size > 1000:
            continue
        if item.podcastGuid in broken_guids:
            continue
        res.append(item)
    return res


async def main():
    db = SessionLocal()
    podcasts = get_podcasts(db, -1)
    print(f'Total amount of podcasts {len(podcasts)}')
    podcasts = filter_already_downloaded_or_broken(podcasts)
    print(f'TODO. Total amount of podcasts {len(podcasts)}')
    size = 300
    async with aiohttp.ClientSession() as sess:
        for chunk_idx in range(0, len(podcasts), size):
            tasks_list = [asyncio.create_task(get_rss(sess, podcast)) for podcast in
                          podcasts[chunk_idx: chunk_idx + size]]
            for t in tqdm.tqdm(tasks_list):
                await t


def main_multiproc():
    db = SessionLocal()
    podcasts = get_podcasts(db, -1)
    print(f'Total amount of podcasts {len(podcasts)}')

    csv_path = podidx_data_root.joinpath('broken_guids.csv')
    podcasts = filter_already_downloaded_or_broken(podcasts, csv_path)
    random.shuffle(podcasts)
    print(f'After filtering, total amount of podcasts {len(podcasts)}')

    fd = open(csv_path, 'a')
    writer = csv.DictWriter(fd, ['GUID'])

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        for podcast in podcasts:
            futures.append(executor.submit(get_rss_sequentially, podcast))
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            res = future.result()
            pod_guid, status = res
            if status is False:
                writer.writerow({'GUID': pod_guid})

    fd.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--implementation', choices=['asyncio', 'multiprocessing'], default='multiprocessing')
    args = parser.parse_args()
    start = time.time()
    if args.implementation == 'asyncio':
        asyncio.run(main())
    else:
        main_multiproc()
    print(time.time() - start)
