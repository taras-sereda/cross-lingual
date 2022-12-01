import os

import torch
from omegaconf import OmegaConf

cfg = OmegaConf.load('config.yaml')

if torch.cuda.is_available():
    preset = 'standard'
else:
    preset = 'ultra_fast'

cfg.tts.preset = preset


example_text = """
Everything was perfectly swell.

There were no prisons, no slums, no insane asylums, no cripples, no
poverty, no wars.

All diseases were conquered. So was old age.

Death, barring accidents, was an adventure for volunteers.

The population of the United States was stabilized at forty-million
souls.

One bright morning in the Chicago Lying-in Hospital, a man named Edward
K. Wehling, Jr., waited for his wife to give birth. He was the only man
waiting. Not many people were born a day any more.

Wehling was fifty-six, a mere stripling in a population whose average
age was one hundred and twenty-nine.

X-rays had revealed that his wife was going to have triplets. The
children would be his first.

"""

example_voice_sample_path = os.path.join(os.path.dirname(__file__), '../data/VLND2ptAOio.clip.24000.wav')
