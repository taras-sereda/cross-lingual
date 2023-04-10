import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

prj_root = Path(__file__).parent
cfg = OmegaConf.load(prj_root.joinpath('config.yaml'))
cfg.general.sysname = os.uname().sysname

if not cfg.db.data_root:
    if cfg.general.sysname == 'Darwin':
        user = os.environ.get('USER')
        cfg.db.data_root = Path(f"/Users/{user}/crosslingual-data/user_data")
    else:
        cfg.db.data_root = Path("/data/crosslingual-data/user_data")

data_root = cfg.db.data_root
data_root.mkdir(exist_ok=True, parents=True)

if torch.cuda.is_available():
    cfg.tts.preset = 'standard'
else:
    cfg.tts.preset = 'ultra_fast'

cfg.assets.default_video_path = prj_root.joinpath(cfg.assets.default_video_path)