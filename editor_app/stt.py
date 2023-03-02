from datetime import datetime

import whisper

from utils import compute_string_similarity
from . import cfg
from . import schemas, crud
from .models import Utterance

stt_model = whisper.load_model(cfg.stt.model_size)


def transcribe_utterance(utterance: Utterance, language=None):
    seg_res = stt_model.transcribe(str(utterance.get_audio_path()), language=language)
    text = seg_res['text']
    lang = seg_res['language']
    return text, lang


def compute_and_store_score(db, utterance, lang=None) -> float:
    stt_text, lang = transcribe_utterance(utterance, language=lang)
    score = compute_string_similarity(utterance.text, stt_text)
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
