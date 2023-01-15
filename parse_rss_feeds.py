from podindex import SessionLocal, data_root
from podindex.crud import get_podcasts
from utils import email_re
import xml.etree.ElementTree as ET


def ns_find_data(data):
    email, author = None, None
    try:
        ns = {'itunes': "http://www.itunes.com/dtds/podcast-1.0.dtd"}
        root = ET.fromstring(data)
        channel = root.find('channel')

        itunes_owner = channel.find('itunes:owner', ns)
        email = itunes_owner.find('itunes:email', ns)
        author = channel.find('itunes:author', ns)
        if email is not None:
            email = email.text
        else:
            email_match = email_re.search(data)
            if email_match:
                email = email_match.group(0)

        if author is not None:
            author = author.text
    except Exception as e:
        print(e)

    return email, author


if __name__ == '__main__':

    db = SessionLocal()
    podcasts = get_podcasts(db, 2000)
    pods_num_without_email = 0
    pods_num = 0
    for pod in podcasts:
        feed_path = data_root.joinpath(f'rss/{pod.podcastGuid}.xml')
        if not feed_path.exists():
            continue
        with open(feed_path, 'r') as fd:
            data = fd.read()

        email, author = ns_find_data(data)
        if not email:
            pods_num_without_email += 1
            continue
        pods_num += 1
        print(pod.podcastGuid, author, email)

    print(f'total number of pods: {pods_num}')
    print(f'total number of pods without email: {pods_num_without_email}')
