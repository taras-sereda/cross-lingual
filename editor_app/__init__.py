from pathlib import Path

import torch
from omegaconf import OmegaConf

prj_root = Path(__file__).parent.parent
cfg = OmegaConf.load(prj_root.joinpath('config.yaml'))
data_root = prj_root.joinpath(cfg.db.data_root)

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

example_voice_sample_path = Path(__file__).parent.parent.joinpath('data/VLND2ptAOio.clip.24000.wav')

# For some reason gradio, rewrites target all the time to _blank, no matter what target I specify.
# So all links are still opening in a new tab unfortunately.
html_menu = """
<ul>
	<li><a target="_blank" rel="noopener noreferrer" href="/transcribe">transcribe</a></li>
	<li><a target="_blank" rel="noopener noreferrer" href="/translate">translate</a></li>
	<li><a target="_blank" rel="noopener noreferrer" href="/submit">submit</a></li>
	<li><a target="_blank" rel="noopener noreferrer" href="/editor/">edit</a></li>
</ul>
"""
