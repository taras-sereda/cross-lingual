import feedparser


def get_audio_link(rss_file_path: str) -> str | None:
    data = feedparser.parse(rss_file_path)
    res = None
    if len(data.entries) == 0:
        return res
    # consider most recent entry. and look for audio/mpeg data
    for link in data.entries[0].links:
        if link["type"] == "audio/mpeg":
            res = link["href"]
    return res
