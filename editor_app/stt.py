import whisper
import numpy as np
import torch
from torchaudio.transforms import Resample

from . import cfg
from .models import Utterance
import soundfile as sf
from utils import compute_string_similarity
from datetime import datetime
from . import schemas, crud

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


def compute_and_store_score(db, utterance) -> float:
    stt_text, lang = transcribe_utterance(utterance)
    levenstein_score = compute_string_similarity(utterance.text, stt_text)
    score = round(levenstein_score, 3)
    utter_stt = schemas.UtteranceSTTCreate(orig_utterance_id=utterance.id,
                                           text=stt_text,
                                           levenstein_similarity=score,
                                           date=datetime.now())
    crud.create_utterance_stt(db, utter_stt)
    return score


def get_or_compute_score(db, utterance) -> float:
    key_func = lambda x: x.date
    stt_utterances = sorted(utterance.utterance_stt, key=key_func)
    if len(stt_utterances) > 0:
        score = stt_utterances[-1].levenstein_similarity
    else:
        score = compute_and_store_score(db, utterance)
    return score


def calculate_project_score(db, project) -> (float, list):
    scores = []
    for utterance in project.utterances:
        scores.append(get_or_compute_score(db, utterance))
    return sum(scores) / len(scores), scores
