import whisper
import numpy as np
import torch
from torchaudio.transforms import Resample

from . import cfg
from .models import Utterance
import soundfile as sf

stt_model = whisper.load_model(cfg.stt.model_size)


def transcribe_utterance(utterance: Utterance, language=None):

    # waveform, sample_rate = sf.read(utterance.get_audio_path())
    # if waveform.ndim == 1:
    #     waveform = waveform[np.newaxis, :]
    # if waveform.ndim == 2 and np.argmin(waveform.shape) == 1:
    #     waveform = waveform.transpose()
    # waveform = torch.from_numpy(waveform).to(torch.float32)
    #
    # stt_waveform = Resample(orig_freq=sample_rate, new_freq=cfg.stt.sample_rate)(waveform)
    # stt_waveform = stt_waveform.squeeze(0)
    # stt_waveform = stt_waveform / 32768.0
    seg_res = stt_model.transcribe(str(utterance.get_audio_path()), language=language)
    text = seg_res['text']
    lang = seg_res['language']
    return text, lang
