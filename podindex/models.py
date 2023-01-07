from sqlalchemy import Column, Integer, Text

from . import Base


class Podcast(Base):
    __tablename__ = 'podcasts'

    id = Column(Integer, primary_key=True)
    url = Column(Text, nullable=False, unique=True)
    title = Column(Text, nullable=False)
    lastUpdate = Column(Integer)
    link = Column(Text, nullable=False)
    lastHttpStatus = Column(Integer)
    dead = Column(Integer)
    contentType = Column(Text, nullable=False)
    itunesId = Column(Integer)
    originalUrl = Column(Text, nullable=False)
    itunesAuthor = Column(Text, nullable=False)
    itunesOwnerName = Column(Text, nullable=False)
    explicit = Column(Integer)
    imageUrl = Column(Text, nullable=False)
    itunesType = Column(Text, nullable=False)
    generator = Column(Text, nullable=False)
    newestItemPubdate = Column(Integer)
    language = Column(Text, nullable=False)
    oldestItemPubdate = Column(Integer)
    episodeCount = Column(Integer)
    popularityScore = Column(Integer)
    priority = Column(Integer)
    createdOn = Column(Integer)
    updateFrequency = Column(Integer)
    chash = Column(Text, nullable=False)
    host = Column(Text, nullable=False)
    newestEnclosureUrl = Column(Text, nullable=False)
    podcastGuid = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    category1 = Column(Text, nullable=False)
    category2 = Column(Text, nullable=False)
    category3 = Column(Text, nullable=False)
    category4 = Column(Text, nullable=False)
    category5 = Column(Text, nullable=False)
    category6 = Column(Text, nullable=False)
    category7 = Column(Text, nullable=False)
    category8 = Column(Text, nullable=False)
    category9 = Column(Text, nullable=False)
    category10 = Column(Text, nullable=False)
    newestEnclosureDuration = Column(Integer)
