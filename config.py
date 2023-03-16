from pathlib import Path

import torch
from omegaconf import OmegaConf

prj_root = Path(__file__).parent
cfg = OmegaConf.load(prj_root.joinpath('config.yaml'))
data_root = prj_root.joinpath(cfg.db.data_root)
data_root.mkdir(exist_ok=True, parents=True)

if torch.cuda.is_available():
    preset = 'standard'
else:
    preset = 'ultra_fast'

cfg.tts.preset = preset