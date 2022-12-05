import json
from collections import defaultdict
from datetime import datetime

import gradio as gr
import soundfile as sf
import numpy as np
from sqlalchemy.orm import Session

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voices
from tortoise.utils.text import split_and_recombine_text

from utils import compute_string_similarity, split_on_raw_utterances
from datatypes import RawUtterance
from . import schemas, crud, cfg
from .database import SessionLocal
from .models import User, Project, Utterance, Speaker
from .stt import stt_model


tts_model = TextToSpeech()


def add_speaker(audio_tuple, speaker_name, user_email):
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    if not user:
        raise Exception(f"User {user_email} not found. Provide valid email")

    if speaker_name in [spkr.name for spkr in user.speakers]:
        raise Exception(f"Speaker {speaker_name} already exists!")

    # write data to db
    speaker = crud.create_speaker(db, speaker_name, user.id)

    # save data on disk
    wav_path = speaker.get_speaker_data_root().joinpath('1.wav')
    sample_rate, audio = audio_tuple
    sf.write(wav_path, audio, sample_rate)
    db.close()


def get_speakers(user_email):
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    db.close()
    return [spkr.name for spkr in user.speakers]


def get_projects(user_email):
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    db.close()
    return [spkr.title for spkr in user.projects]


def read(title, raw_text, user_email):
    if len(title) == 0:
        raise Exception(f"Project title {title} can't be empty.")

    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    if not user:
        raise Exception(f"User {user_email} not found. Provide valid email")
    if crud.get_project_by_title(db, title, user.id):
        raise Exception(f"Project {title} already exists! Try to load it")

    speakers_to_features = dict()
    data: list[RawUtterance] = []
    for utter in split_on_raw_utterances(raw_text):

        db_speaker: Speaker = crud.get_speaker_by_name(db, utter.speaker, user.id)
        if not db_speaker:
            raise Exception(f"No such speaker {utter.speaker} {db_speaker}")

        # load voice samples and conditioning latents.
        if db_speaker.name not in speakers_to_features:
            spkr_data_root = db_speaker.get_speaker_data_root().parent
            voice_samples, conditioning_latents = load_voices([db_speaker.name], [spkr_data_root])
            speakers_to_features[db_speaker.name] = {
                'id': db_speaker.id,
                'voice_samples': voice_samples,
                'conditioning_latents': conditioning_latents}

        for text in split_and_recombine_text(utter.text):
            data.append(RawUtterance(utter.timecode, db_speaker.name, text))

    project_data = schemas.ProjectCreate(title=title, text=raw_text, date_created=datetime.now())
    project = crud.create_project(db, project_data, user.id)

    for idx, utter in enumerate(data):
        gen_start = datetime.now()
        gen = tts_model.tts_with_preset(utter.text,
                                        voice_samples=speakers_to_features[utter.speaker]['voice_samples'],
                                        conditioning_latents=speakers_to_features[utter.speaker]['conditioning_latents'],
                                        preset=cfg.tts.preset, k=cfg.tts.candidates, use_deterministic_seed=cfg.tts.seed,
                                        num_autoregressive_samples=cfg.tts.num_autoregressive_samples)
        gen = gen.cpu().numpy().squeeze()

        utterance_data = schemas.UtteranceCreate(text=utter.text, utterance_idx=idx,
                                                 date_started=gen_start, timecode=utter.timecode)
        utterance: Utterance = crud.create_utterance(db, utterance_data, project.id, speakers_to_features[utter.speaker]['id'])
        sf.write(utterance.get_audio_path(), gen, cfg.tts.sample_rate)
        crud.update_any_db_row(db, utterance, date_completed=datetime.now())

    crud.update_any_db_row(db, project, date_completed=datetime.now())
    db.close()


def playground_read(text, speaker_name, user_email):
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    new_speaker: Speaker = crud.get_speaker_by_name(db, speaker_name, user.id)
    if not new_speaker:
        raise Exception(f"Speaker {speaker_name} doesn't exists. Add it first")
    voice_samples, conditioning_latents = load_voices([speaker_name], [new_speaker.get_speaker_data_root().parent])
    gen = tts_model.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                    preset=cfg.tts.preset, k=cfg.tts.candidates, use_deterministic_seed=cfg.tts.seed,
                                    num_autoregressive_samples=cfg.tts.num_autoregressive_samples)
    gen = gen.cpu().numpy().squeeze()
    db.close()

    stt_res = stt_model.transcribe(gen)
    stt_text = stt_res['text']
    similarity = compute_string_similarity(text, stt_text)
    return stt_text, similarity, (cfg.tts.sample_rate, gen), new_speaker.name


def load(project_name: str, user_email: str, from_idx: int):
    db = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    if not user:
        raise Exception(f"User {user_email} not found. Provide valid email")

    project: Project = crud.get_project_by_title(db, project_name, user.id)
    if not project:
        raise Exception(f"no such project {project_name}. Provide valid project title")

    utterances = project.utterances[from_idx: from_idx + cfg.editor.max_utterance]
    res = [project.text, len(project.utterances)]
    for utterance in utterances:
        res.append(gr.Textbox.update(value=utterance.text, visible=True))
        res.append(gr.Number.update(value=utterance.utterance_idx))
        res.append(gr.Textbox.update(value=utterance.speaker.name, visible=True))

        gen, sample_rate = sf.read(utterance.get_audio_path())
        res.append(gr.Audio.update(value=(sample_rate, gen), visible=True))
        res.append(gr.Button.update(visible=True))
    db.close()

    # padding
    if (delta := cfg.editor.max_utterance - len(utterances)) > 0:
        for _ in range(delta):
            res.append(gr.Textbox.update(visible=False))
            res.append(gr.Number.update(visible=False))
            res.append(gr.Textbox.update(visible=False))
            res.append(gr.Audio.update(visible=False))
            res.append(gr.Button.update(visible=False))
    return res


def reread(title, text, utterance_idx, speaker_name, user_email):
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    if not user:
        raise Exception(f"User {user_email} not found. Provide valid email")
    project: Project = crud.get_project_by_title(db, title, user.id)
    if not project:
        raise Exception(f"Project {title} doesn't exists!"
                        f"Normally this shouldn't happen")
    new_speaker: Speaker = crud.get_speaker_by_name(db, speaker_name, user.id)
    if not new_speaker:
        raise Exception(f"Speaker {speaker_name} doesn't exists. Add it first")

    utterance_db: Utterance = crud.get_utterance(db, utterance_idx, project.id)
    if not utterance_db:
        raise Exception(f"Something went wrong, Utterance {utterance_idx} doesn't exists."
                        f"Normally this shouldn't happen")
    start_time = datetime.now()
    voice_samples, conditioning_latents = load_voices([speaker_name], [new_speaker.get_speaker_data_root().parent])
    gen = tts_model.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                    preset=cfg.tts.preset, k=cfg.tts.candidates, use_deterministic_seed=cfg.tts.seed,
                                    num_autoregressive_samples=cfg.tts.num_autoregressive_samples)
    gen = gen.cpu().numpy().squeeze()

    sf.write(utterance_db.get_audio_path(), gen, cfg.tts.sample_rate)
    update_dict = {
        'text': text,
        'speaker_id': new_speaker.id,
        'date_created': start_time,
        'date_completed': datetime.now()
    }
    crud.update_any_db_row(db, utterance_db, **update_dict)
    db.close()
    return (cfg.tts.sample_rate, gen), new_speaker.name


def combine(title, user_email):
    db = SessionLocal()

    user: User = crud.get_user_by_email(db, user_email)
    if not user:
        raise Exception(f"User {user_email} not found. Provide valid email")

    project = crud.get_project_by_title(db, title, user.id)
    if not project:
        raise Exception(f"no such project {title}. Provide valid project title")

    utterances: list[Utterance] = project.utterances
    combined_dir = utterances[0].get_audio_path().parent.joinpath('combined')
    combined_dir.mkdir(exist_ok=True)

    if not list(combined_dir.glob("*.wav")):
        unique_speakers = set([utter.speaker_id for utter in utterances])
        project_audio_tracks = defaultdict(list)
        metadata = []
        start_sec = 0
        for utterance in utterances:
            utter_audio, sample_rate = sf.read(utterance.get_audio_path())

            assert utter_audio.ndim == 1
            n_sample = utter_audio.shape[0]
            project_audio_tracks[utterance.speaker_id].extend(utter_audio)

            silence_audio = np.zeros_like(utter_audio)
            for other_spkr_id in (unique_speakers - {utterance.speaker_id}):
                project_audio_tracks[other_spkr_id].extend(silence_audio)

            end_sec = start_sec + n_sample / sample_rate
            metadata.append({
                'start_sec': f'{start_sec:.3f}',
                'end_sec': f'{end_sec:.3f}',
                'speaker_id': utterance.speaker_id,
                'speaker_name': utterance.speaker.name
            })
            start_sec = end_sec

        for k, v in project_audio_tracks.items():
            sf.write(combined_dir.joinpath(f'combined_{k}.wav'), v, cfg.tts.sample_rate)

        with open(combined_dir.joinpath('metadata.json'), 'w') as fd:
            json.dump(metadata, fd)

    db.close()

    tracks = []
    for track_path in combined_dir.glob("*.wav"):
        combined_wav, sample_rate = sf.read(track_path)
        tracks.append(combined_wav)
    wav_overlayed = np.array(tracks).mean(axis=0)

    return gr.Audio.update(value=(cfg.tts.sample_rate, wav_overlayed), visible=True)
