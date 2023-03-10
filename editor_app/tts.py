import bisect
import json
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime

import gradio as gr
import soundfile as sf
import numpy as np
import pandas as pd
import torch
from sqlalchemy.orm import Session

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voices, load_audio
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.wav2vec_alignment import Wav2VecAlignment

from utils import compute_string_similarity, split_on_raw_utterances, raw_speaker_re, time_re, normalize_text, \
    find_single_repetition
from datatypes import RawUtterance
from . import schemas, crud, cfg
from .database import SessionLocal
from .models import User, Utterance, Speaker, CrossProject, Translation
from .stt import stt_model, compute_and_store_score, get_or_compute_score, calculate_project_score

tts_model = TextToSpeech()
aligner = Wav2VecAlignment()


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
    user: User = crud.get_user_by_email(db, user_email)
    cross_project = crud.get_cross_project_by_title(db, cross_project_name, user.id)
    db.close()
    data = defaultdict(list)
    for spkr in cross_project.speakers:
        data['name'] += [spkr.name]
    return pd.DataFrame(data=data).head(limit)


def get_cross_projects(user_email, limit=10) -> pd.DataFrame:
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    db.close()
    data = defaultdict(list)
    for proj in user.crosslingual_projects:
        for translation in proj.translations:
            data['title'] += [proj.title]
            data['lang'] += [translation.lang]
            data['date_completed'] += [translation.date_completed]
    return pd.DataFrame(data=data).sort_values(by=['date_completed'], ascending=False, na_position='first').head(limit)


def read(title, lang, raw_text, user_email, check_for_repetitions=False):
    if len(title) == 0:
        raise Exception(f"Project title {title} can't be empty.")

    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    translation: Translation = crud.get_translation_by_title_and_lang(db, title, lang, user.id)
    if not translation:
        # raise Exception(f"Project {title} already exists! Try to load it")
        raise Exception(f"At the moment only {title} existing translations are supported.")

    speakers_to_features = dict()
    data: list[RawUtterance] = []
    for utter in split_on_raw_utterances(raw_text):

        db_speaker: Speaker = crud.get_speaker_by_name(db, utter.speaker, translation.cross_project_id)
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
        utterance: Utterance = crud.create_utterance(db, utterance_data, translation.id, speakers_to_features[utter.speaker]['id'])
        sf.write(utterance.get_audio_path(), gen, cfg.tts.sample_rate)
        crud.update_any_db_row(db, utterance, date_completed=datetime.now())

    project = crud.update_any_db_row(db, translation, date_completed=datetime.now())

    for utter in project.utterances:
        # score already computed in editor.
        if len(utter.utterance_stt) > 0:
            continue
        score = compute_and_store_score(db, utter)

    if check_for_repetitions:
        key_func = lambda x: x.date
        for utter in project.utterances:
            stt_utterance = sorted(utter.utterance_stt, key=key_func)[-1]
            if stt_utterance.levenstein_similarity == 1.0:
                continue
            text_norm = normalize_text(utter.text)
            text_stt_norm = normalize_text(stt_utterance.text)
            res = find_single_repetition(text_stt_norm, text_norm)
            if not res:
                continue
            audio_path = utter.get_audio_path()
            wav, sample_rate = sf.read(audio_path, dtype='float32')
            wav = torch.from_numpy(wav).unsqueeze(0)
            new_res_wav = aligner.redact(wav, res)
            new_res_wav = new_res_wav.cpu().numpy().squeeze()
            new_text_stt = stt_model.transcribe(new_res_wav)['text']
            new_similarity_score = compute_string_similarity(utter.text, new_text_stt)
            if new_similarity_score > stt_utterance.levenstein_similarity:
                print(f'Yay! looks like repetition caught and corrected, for text {utter.text}'
                      f'new score {new_similarity_score} > {stt_utterance.levenstein_similarity}')
                utter_stt = schemas.UtteranceSTTCreate(orig_utterance_id=utter.id,
                                                       text=new_text_stt,
                                                       levenstein_similarity=new_similarity_score,
                                                       date=datetime.now())
                crud.create_utterance_stt(db, utter_stt)
                sf.write(utter.get_audio_path(), new_res_wav, sample_rate)

    db.close()


def playground_read(text, speaker_name, user_email):
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    # support of multiple speakers
    spkr_to_spkr_root = dict()
    for spkr in speaker_name.split('&'):
        db_speaker: Speaker = crud.get_speaker_by_name(db, spkr, user.id)
        if not db_speaker:
            raise Exception(f"Speaker {spkr} doesn't exists. Add it first")
        spkr_to_spkr_root[db_speaker.name] = db_speaker.get_speaker_data_root().parent

    voice_samples, conditioning_latents = load_voices(list(spkr_to_spkr_root.keys()), list(spkr_to_spkr_root.values()))
    gen = tts_model.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                    preset=cfg.tts.preset, k=cfg.tts.playground.candidates, use_deterministic_seed=None,
                                    num_autoregressive_samples=cfg.tts.num_autoregressive_samples)
    db.close()
    outputs = []
    for g in gen:
        g = g.cpu().numpy().squeeze()
        stt_res = stt_model.transcribe(g)
        stt_text = stt_res['text']
        similarity = compute_string_similarity(text, stt_text)
        outputs.append(stt_text)
        outputs.append(similarity)
        outputs.append((cfg.tts.sample_rate, g))

    return outputs


def load_translation(cross_project_name: str, lang: str, user_email: str):
    db = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    translation_project: Translation = crud.get_translation_by_title_and_lang(db, cross_project_name, lang, user.id)
    speakears = get_speakers(user_email, cross_project_name)
    return translation_project.text, speakears


def load(cross_project_name: str, lang: str, user_email: str, from_idx: int, score_threshold=None):
    db = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)

    project: Translation = crud.get_translation_by_title_and_lang(db, cross_project_name, lang, user.id)
    if not project:
        raise Exception(f"no such project {cross_project_name}. Provide valid project title")
    all_utterances = project.utterances

    avg_project_score, all_scores = calculate_project_score(db, project)
    speakers = get_speakers(user_email, cross_project_name)
    res = [speakers, project.text, len(all_utterances), avg_project_score]

    if score_threshold > 0.0:
        temp = [(sc, ut) for sc, ut in sorted(zip(all_scores, all_utterances), key=lambda x: x[0])]
        sorted_from_idx = bisect.bisect_right(temp, x=score_threshold, key=lambda x: x[0])
        all_utterances = [ut for (sc, ut) in temp[sorted_from_idx:]]

    utterances = all_utterances[from_idx: from_idx + cfg.editor.max_utterance]

    for utterance in utterances:
        res.append(gr.Textbox.update(value=utterance.text, visible=True))
        res.append(gr.Number.update(value=utterance.utterance_idx))
        res.append(gr.Textbox.update(value=utterance.speaker.name, visible=True))

        score = get_or_compute_score(db, utterance)
        res.append(gr.Number.update(value=score, visible=True))

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
            res.append(gr.Number.update(visible=False))
            res.append(gr.Audio.update(visible=False))
            res.append(gr.Button.update(visible=False))
    return res


def reread(cross_project_name, lang, text, utterance_idx, speaker_name, user_email):
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)

    translation: Translation = crud.get_translation_by_title_and_lang(db, cross_project_name, lang, user.id)
    if not translation:
        raise Exception(f"Project {cross_project_name} doesn't exists! "
                        f"Normally this shouldn't happen")
    new_speaker: Speaker = crud.get_speaker_by_name(db, speaker_name, translation.cross_project_id)
    if not new_speaker:
        raise Exception(f"Speaker {speaker_name} doesn't exists. Add it first")

    utterance: Utterance = crud.get_utterance(db, utterance_idx, translation.id)
    if not utterance:
        raise Exception(f"Something went wrong, Utterance {utterance_idx} doesn't exists. "
                        f"Normally this shouldn't happen")
    start_time = datetime.now()
    voice_samples, conditioning_latents = load_voices([speaker_name], [new_speaker.get_speaker_data_root().parent])
    # in favour of variation seed should be None
    gen = tts_model.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                    preset=cfg.tts.preset, k=cfg.tts.candidates, use_deterministic_seed=None,
                                    num_autoregressive_samples=cfg.tts.num_autoregressive_samples)
    gen = gen.cpu().numpy().squeeze()

    sf.write(utterance.get_audio_path(), gen, cfg.tts.sample_rate)
    update_dict = {
        'text': text,
        'speaker_id': new_speaker.id,
        'date_created': start_time,
        'date_completed': datetime.now()
    }
    utterance = crud.update_any_db_row(db, utterance, **update_dict)
    score = compute_and_store_score(db, utterance, lang='en' if lang == 'EN-US' else None)
    db.close()
    return (cfg.tts.sample_rate, gen), speaker_name, score


def combine(cross_project_name, lang, user_email, load_duration_sec=120):
    db = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)

    project: Translation = crud.get_translation_by_title_and_lang(db, cross_project_name, lang, user.id)
    if not project:
        raise Exception(f"no such project {cross_project_name}. Provide valid project title")

    utterances: list[Utterance] = project.utterances
    combined_dir = utterances[0].get_audio_path().parent.joinpath('combined')
    combined_dir.mkdir(exist_ok=True)

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
            'speaker_name': utterance.speaker.name,
            'text': utterance.text,
        })
        start_sec = end_sec

    for k, v in project_audio_tracks.items():
        sf.write(combined_dir.joinpath(f'combined_{k}.wav'), v, cfg.tts.sample_rate)

    # quick mixing of all channels
    all_channels = list(project_audio_tracks.values())
    combined_waveform = np.array(all_channels[0])
    for i in range(1, len(all_channels)):
        combined_waveform += all_channels[i]
    # normalize
    combined_waveform /= np.abs(combined_waveform).max()
    combined_wav_path = combined_dir.joinpath(f'{project.cross_project.title}.wav')

    sf.write(combined_wav_path, combined_waveform, cfg.tts.sample_rate)

    ffmpeg_path = shutil.which('ffmpeg')
    res = subprocess.run([
        f"{ffmpeg_path}",
        "-hide_banner",
        "-loglevel", "error",
        "-i", f"{combined_wav_path}",
        "-ab", "320k",
        f"{combined_wav_path.with_suffix('.mp3')}"],
        check=True,
        # stdout=subprocess.DEVNULL,
    )

    with open(combined_dir.joinpath('metadata.json'), 'w') as fd:
        json.dump(metadata, fd)

    db.close()

    # there is no need it sending gigabytes of data to front-end,
    # so loading load_duration_sec amount is sufficient.
    tracks = []
    for track_path in combined_dir.glob("combined_*.wav"):
        start = int(0 * cfg.tts.sample_rate)
        stop = start + int(load_duration_sec * cfg.tts.sample_rate)
        combined_wav, sample_rate = sf.read(track_path, start=start, stop=stop)
        tracks.append(combined_wav)
    wav_overlayed = np.array(tracks).mean(axis=0)

    return gr.Audio.update(value=(cfg.tts.sample_rate, wav_overlayed), visible=True)


def time_to_sec(time: str) -> float:
    hour, min, other = time.split(':')
    sec, msec = other.split('.')
    return int(hour) * 60 * 60 + int(min) * 60 + int(sec) + int(msec) / 1000


def timecode_to_timerange(timecode: str, sample_rate: int) -> tuple[int, int]:
    res = time_re.findall(timecode)
    assert len(res) == 2
    start_sec = time_to_sec(res[0])
    end_sec = time_to_sec(res[1])
    start_frame = int(start_sec * sample_rate)
    end_frame = int(end_sec * sample_rate)
    return start_frame, end_frame


def timecode_based_combine(utterances: list[Utterance]):
    """
    This method will only make sense if total duration of all utterances
    is shorter than duration of reference video.

    """
    temp_utterances = []
    cur_timecode = utterances[0].timecode
    cur_group = []
    for utter in utterances:
        utter_audio, sample_rate = sf.read(utter.get_audio_path())

        assert utter_audio.ndim == 1

        if utter.timecode == cur_timecode:
            cur_group.extend(list(utter_audio))
        else:
            time_range = timecode_to_timerange(cur_timecode, sample_rate)
            temp_utterances.append((time_range, cur_group))
            cur_timecode = utter.timecode
            cur_group = list(utter_audio)

    res = []
    for item in temp_utterances:
        time_range, utter_wav = item
        s, e = time_range
        # pad with silence
        print(s, e, len(res))
        if s > len(res):
            silence_duration = len(res) - s
            silence_seg = [0] * silence_duration
            res.extend(silence_seg)
        res.extend(utter_wav)

    combined_dir = utterances[0].get_audio_path().parent.joinpath('combined')
    combined_dir.mkdir(exist_ok=True)
    sf.write(combined_dir.joinpath(f'timecode_combined.wav'), res, cfg.tts.sample_rate)
