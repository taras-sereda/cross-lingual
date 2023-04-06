import shutil
import tempfile
from collections import Counter
from datetime import datetime

import gradio as gr
import soundfile as sf
import whisper
from pyannote.audio import Pipeline
from sqlalchemy.orm import Session

from media_utils import download_youtube_media, extract_and_resample_audio_ffmpeg, get_youtube_embed_code, media_has_video_steam
from string_utils import validate_and_preprocess_title, get_random_string
from utils import compute_string_similarity, get_user_from_request
from utils import gradio_read_audio_data
from config import cfg
from db import crud, schemas
from db.database import SessionLocal
from db.models import Utterance, CrossProject

stt_model = whisper.load_model(cfg.stt.model_size)
diarization_model = Pipeline.from_pretrained(cfg.diarization.model_name, use_auth_token=cfg.diarization.auth_token)


def transcribe(input_media, media_link, project_name: str, language: str, options: list, request: gr.Request):
    """Transcribe input media with speaker diarization, resulting transcript will be in form:
    [ HH:MM:SS.sss --> HH:MM:SS.sss ]
    {SPEAKER}
    Transcribed text
    """
    demo_run, save_speakers = False, False
    if 'Demo Run' in options:
        demo_run = True
    if 'Save speakers' in options:
        save_speakers = True
    if len(language) == 0:
        language = None
    user_email = get_user_from_request(request)
    project_name = validate_and_preprocess_title(project_name)
    db: Session = SessionLocal()
    user = crud.get_user_by_email(db, user_email)

    cross_project = crud.get_cross_project_by_title(db, project_name, user.id, ensure_exists=False)
    if cross_project is not None:
        raise Exception(f"CrossProject {project_name} already exists, pick another name")
    if media_link is not None:
        tmp_media_path = download_youtube_media(media_link, tempfile.gettempdir())
        name = tmp_media_path.name
    elif input_media is not None:
        tmp_media_path = input_media.name
        name = get_random_string()
    else:
        raise Exception(f"either media_link or media file should be provided")

    # TODO refactor, I need to have media path before I create crosslingual db entry.
    # TODO Add more checks and atomicity.
    # TODO I don't want to manually cleanup disk and database if something went wrong while project creation.

    cross_project_data = schemas.CrossProjectCreate(title=project_name, media_name=name)
    cross_project = crud.create_cross_project(db, cross_project_data, user.id)
    media_path = shutil.copy(tmp_media_path, cross_project.get_media_path())
    wav_16k_path = cross_project.get_raw_wav_path(sample_rate=cfg.stt.sample_rate)
    wav_22k_path = cross_project.get_raw_wav_path(sample_rate=cfg.tts.spkr_emb_sample_rate)

    res0 = extract_and_resample_audio_ffmpeg(media_path, wav_16k_path, cfg.stt.sample_rate)
    res1 = extract_and_resample_audio_ffmpeg(media_path, wav_22k_path, cfg.tts.spkr_emb_sample_rate)

    waveform_16k, _ = gradio_read_audio_data(wav_16k_path)
    waveform_22k, _ = gradio_read_audio_data(wav_22k_path)

    if demo_run:
        waveform_16k = waveform_16k[:(10*60*cfg.stt.sample_rate)]
        waveform_22k = waveform_22k[:(10*60*cfg.tts.spkr_emb_sample_rate)]
        # unset ad offset for projects shorter than offset
        if len(waveform_16k) / cfg.stt.sample_rate < (cfg.demo.ad_offset_sec + cfg.demo.duration_sec):
            ad_offset = -1
        else:
            ad_offset = cfg.demo.ad_offset_sec

    diarization = diarization_model({'waveform': waveform_16k.unsqueeze(0), 'sample_rate': cfg.stt.sample_rate})
    demo_duration = 0
    res_text, ffmpeg_str = '', ''
    segments, speaker_samples = [], dict()
    # in case of unspecified input language, utterances can have various languages.
    # this is a reason for using a counter
    detected_languages = Counter()

    for idx, (seg, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):

        if demo_run:
            if seg.start < ad_offset:
                continue
        speaker = speaker.lower()
        seg_wav_16k = waveform_16k[int(seg.start * cfg.stt.sample_rate): int(seg.end * cfg.stt.sample_rate)]
        seg_wav_22k = waveform_22k[int(seg.start * cfg.tts.spkr_emb_sample_rate): int(seg.end * cfg.tts.spkr_emb_sample_rate)]
        decode_options = whisper.DecodingOptions(fp16=cfg.stt.half_precision,
                                                 language=language)
        seg_res = stt_model.transcribe(seg_wav_16k, **decode_options.__dict__)
        text = seg_res['text']
        if len(text) == 0:
            continue
        lang = seg_res['language']
        segments.append([seg, speaker, text, lang])
        res_text += f"{seg}\n{{{speaker}}}\n{text}\n\n"
        detected_languages[lang] += 1

        if save_speakers:
            seg_name = f'output_{idx:03}.wav'
            db_spkr = crud.get_speaker_by_name(db, speaker, cross_project.id)
            if not db_spkr:
                db_spkr = crud.create_speaker(db, speaker, cross_project.id)
            seg_path = db_spkr.get_speaker_data_root().joinpath(seg_name)
            sf.write(seg_path, seg_wav_22k, cfg.tts.spkr_emb_sample_rate)
            ffmpeg_str += f' -ss {seg.start} -to {seg.end} -c copy {seg_name}'

        if speaker not in speaker_samples:
            speaker_samples[speaker] = seg

        if demo_run:
            demo_duration += (seg.end - seg.start)
            if demo_duration >= cfg.demo.duration_sec:
                break

    # quick and dirty way to cut audio on pieces with ffmpeg.
    print(ffmpeg_str)
    # use most common language as a project lvl language
    detected_lang = detected_languages.most_common(1)[0][0]

    results = [res_text, detected_lang]

    detected_spkrs = ''
    for spkr, seg_duration in speaker_samples.items():
        row = f'{spkr}: {seg_duration}\n'
        detected_spkrs += row
    results.append(detected_spkrs)
    results.extend(add_src_media_components(cross_project, media_link))
    return results


def add_src_media_components(cross_project: CrossProject, media_link: str | None):
    res = []
    if media_link is not None:
        iframe_val = get_youtube_embed_code(media_link)
        res.append(gr.HTML.update(value=iframe_val))
        res.append(gr.Audio.update(visible=False))
        res.append(gr.Video.update(visible=False))
    else:
        res.append(gr.HTML.update(visible=False))
        media_path = cross_project.get_media_path()
        if media_has_video_steam(media_path):
            res.append(gr.Audio.update(visible=False))
            res.append(gr.Video.update(visible=True, value=str(media_path)))
        else:
            res.append(gr.Audio.update(visible=True, value=str(media_path)))
            res.append(gr.Video.update(visible=False))
    return res


def load_transcript(project_name: str, request: gr.Request) -> list:
    user_email = get_user_from_request(request)
    project_name = validate_and_preprocess_title(project_name)
    db: Session = SessionLocal()
    user = crud.get_user_by_email(db, user_email)
    cross_project = crud.get_cross_project_by_title(db, project_name, user.id, ensure_exists=True)
    if len(cross_project.transcript) == 0:
        Exception("CrossProject doesn't have a transcript, normally this shouldn't happen, contact me!!!")

    return [project_name, cross_project.transcript[0].text]


def save_transcript(project_name: str, text: str, lang: str, request: gr.Request) -> str:
    user_email = get_user_from_request(request)
    project_name = validate_and_preprocess_title(project_name)
    db: Session = SessionLocal()
    user = crud.get_user_by_email(db, user_email)
    cross_project = crud.get_cross_project_by_title(db, project_name, user.id, ensure_exists=True)

    if len(cross_project.transcript) > 0:
        Exception("Transcript already saved!!!")

    transcript_data = schemas.TranscriptCreate(text=text, lang=lang)
    transcript_db = crud.create_transcript(db, transcript_data, cross_project.id)

    with open(transcript_db.get_path(), 'w') as f:
        f.write(text)
    return project_name


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
    if len(scores) == 0:
        mean = 0
    else:
        mean = sum(scores) / len(scores)
    return mean, scores
