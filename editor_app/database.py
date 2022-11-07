from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine

from . import cfg

SQLALCHEMY_DATABASE_URL = f'sqlite:///./{cfg.db.name}'
# engine = create_engine('sqlite:///:memory:', echo=True, future=True)
# engine = create_engine("sqlite://")
# above two are equivalent.

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={'check_same_thread': False}, echo=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
