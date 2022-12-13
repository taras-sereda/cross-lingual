from pathlib import Path
from omegaconf import OmegaConf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

podidx_cfg = OmegaConf.load(Path(__file__).parent.parent.joinpath('config.yaml')).podcastindex

db_absolute_path = Path(__file__).parent.parent.joinpath(podidx_cfg.db.name)
engine = create_engine(f"sqlite:///{db_absolute_path}", connect_args={'check_same_thread': False}, future=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
