import whisper

from . import cfg

stt_model = whisper.load_model(cfg.stt.model_size)
