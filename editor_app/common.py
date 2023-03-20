from collections import defaultdict

import gradio as gr
import pandas as pd
import soundfile as sf
from sqlalchemy.orm import Session
from tortoise.utils.audio import load_audio

from config import cfg
from editor_app import crud
from editor_app.database import SessionLocal
from editor_app.models import User
from utils import get_user_from_request, raw_speaker_re


def get_cross_projects(request: gr.Request, limit=10) -> pd.DataFrame:
    user_email = get_user_from_request(request)
    db: Session = SessionLocal()
    user = crud.get_user_by_email(db, user_email)
    db.close()
    data = defaultdict(list)
    for proj in user.crosslingual_projects:
        for translation in proj.translations:
            data['title'] += [proj.title]
            data['lang'] += [translation.lang]
            data['date_completed'] += [translation.date_completed]
    data = pd.DataFrame(data=data).head(limit)
    if len(data) > 0:
        data = data.sort_values(by=['date_completed'], ascending=False, na_position='first')
    return data


def get_transcripts(request: gr.Request, limit=10, without_translations=True) -> pd.DataFrame:
    user_email = get_user_from_request(request)
    db: Session = SessionLocal()
    user = crud.get_user_by_email(db, user_email)
    db.close()
    data = defaultdict(list)
    for proj in user.crosslingual_projects:
        assert len(proj.transcript) in {0, 1}, "Single CrossLingual project may have one transcription at most"
        if len(proj.transcript) == 0:
            print(proj.title)
            continue
        transcript = proj.transcript[0]
        if without_translations and len(proj.translations) > 0:
            continue
        data['title'] += [proj.title]
        data['lang'] += [transcript.lang]
    data = pd.DataFrame(data=data).head(limit)
    # if len(data) > 0:
    #     data = data.sort_values(by=['date_completed'], ascending=False, na_position='first').head(limit)
    return data

def add_speaker(audio_tuple, speaker_name, user_email):
    raise NotImplementedError
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)

    if not raw_speaker_re.fullmatch(speaker_name):
        raise Exception(f"Invalid speaker name")

    if speaker_name in [spkr.name for spkr in user.speakers]:
        raise Exception(f"Speaker {speaker_name} already exists!")

    # write data to db
    speaker = crud.create_speaker(db, speaker_name, user.id)

    # save data on disk
    for idx, audio_temp_file in enumerate(audio_tuple):
        wav_path = speaker.get_speaker_data_root().joinpath(f'{idx:03}.wav')
        data = load_audio(audio_temp_file.name, sampling_rate=cfg.tts.spkr_emb_sample_rate)
        sf.write(wav_path, data.squeeze(0), cfg.tts.spkr_emb_sample_rate)
    db.close()


def get_speakers(user_email, cross_project_name, limit=10) -> pd.DataFrame:
    db: Session = SessionLocal()
    user = crud.get_user_by_email(db, user_email)
    cross_project = crud.get_cross_project_by_title(db, cross_project_name, user.id)
    db.close()
    data = defaultdict(list)
    for spkr in cross_project.speakers:
        data['name'] += [spkr.name]
    return pd.DataFrame(data=data).head(limit)