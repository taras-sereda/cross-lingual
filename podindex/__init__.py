from pathlib import Path
from omegaconf import OmegaConf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

podidx_cfg = OmegaConf.load(Path(__file__).parent.parent.joinpath('config.yaml')).podcastindex

podidx_data_root = Path(__file__).parent.parent.joinpath('podindex-data')
if not podidx_data_root.exists():
    podidx_data_root.mkdir(exist_ok=True)

db_absolute_path = podidx_data_root.joinpath(podidx_cfg.db.name)
engine = create_engine(f"sqlite:///{db_absolute_path}", connect_args={'check_same_thread': False}, future=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
