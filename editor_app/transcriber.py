import subprocess
from typing import Tuple

import gradio as gr
import numpy as np
import soundfile as sf
from pathlib import Path
import shutil

from sqlalchemy.orm import Session
from torchaudio.transforms import Resample
from pyannote.audio import Pipeline

from . import cfg, data_root, example_voice_sample_path, crud
from .database import SessionLocal
from .models import User
from .stt import stt_model
from utils import gradio_read_audio_data, not_raw_speaker_re

diarization_model = Pipeline.from_pretrained(cfg.diarization.model_name, use_auth_token=cfg.diarization.auth_token)


def detect_speakers(input_media, project_name, user_email):
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    if not user:
        raise Exception(f"User {user_email} not found. Provide valid email")
    name = input_media.name.split('/')[-1]
    cross_project = crud.create_cross_project(db, {'title': project_name, 'media_name': name}, user.id)
    media_path = shutil.copy(input_media.name, cross_project.get_media_path())
    raw_media_path = Path(media_path).with_suffix(".16kHz.wav")
    # ffmpeg
    command = f"/usr/bin/ffmpeg -i {media_path} -ac 1 -ar 16000 {raw_media_path}"
    subprocess.run(command, shell=True)
    waveform, sample_rate = gradio_read_audio_data(raw_media_path)

    diarization = diarization_model({'waveform': waveform, 'sample_rate': sample_rate})

    speakers = set(diarization.labels())
    speaker_samples = dict()

    for seg, lbl, spkr in diarization.itertracks(yield_label=True):
        if spkr not in speaker_samples:
            speaker_samples[spkr] = seg
        if len(speakers) == len(speaker_samples):
            break
    res = ''
    for spkr in sorted(speakers):
        row = f'{spkr}: {speaker_samples[spkr]}\n'
        res += row
    return res


def transcribe(audio_data: Tuple[int, np.ndarray], language: str, named_speakers: str):
    """Transcribe input media with speaker diarization, resulting transcript will be in form:
    [ HH:MM:SS.sss --> HH:MM:SS.sss ]
    {SPEAKER}
    Transcribed text
    """
    if len(language) == 0:
        language = None

    waveform, sample_rate = gradio_read_audio_data(audio_data)
    diarizations = diarization_model({'waveform': waveform, 'sample_rate': sample_rate})

    speakers = []
    for speaker in named_speakers.lower().split(','):
        spkr = not_raw_speaker_re.sub('', speaker)
        speakers.append(spkr)

    assert len(diarizations.labels()) == len(speakers)
    diarizations = diarizations.rename_labels(generator=iter(speakers), copy=False)

    stt_waveform = Resample(orig_freq=sample_rate, new_freq=cfg.stt.sample_rate)(waveform)
    stt_waveform = stt_waveform.squeeze(0)
    stt_waveform = stt_waveform / 32768.0
    char_count = 0
    res = ''
    segments = []
    ffmpeg_str = ''

    for idx, (seg, _, speaker) in enumerate(diarizations.itertracks(yield_label=True)):
        seg_wav = stt_waveform[int(seg.start * cfg.stt.sample_rate): int(seg.end * cfg.stt.sample_rate)]
        seg_res = stt_model.transcribe(seg_wav, language=language)
        text = seg_res['text']
        lang = seg_res['language']
        segments.append([seg, speaker, text, lang])
        res += f"{seg}\n{{{speaker}}}\n{text}\n\n"
        char_count += len(seg_res['text'])
        seg_name = f'output_{idx:03}.wav'
        ffmpeg_str += f' -ss {seg.start} -to {seg.end} -c copy {seg_name}'
        # TODO. investigate. saved wavs sound really bad, looks like they are broken.
        # seg_path = temp_dir_path.joinpath(seg_name)
        # orig_seg_wav = waveform[0, int(seg.start * sample_rate): int(seg.end * sample_rate)]
        # sf.write(seg_path, orig_seg_wav, cfg.stt.sample_rate)
    # quick and dirty way to cut audio on pieces with ffmpeg.
    print(ffmpeg_str)
    detected_lang = segments[0][-1]

    return res, detected_lang, char_count


with gr.Blocks() as transcriber:
    with gr.Row() as row0:
        with gr.Column(scale=1) as col0:
            email = gr.Text(label='user', placeholder='Enter user email', value=cfg.user.email)
            project_name = gr.Text(label='Project name', placeholder="enter your project name")
            file = gr.File(label='input media')
            detect_spkr_button = gr.Button(value='Detect speakers')
            detected_speakers = gr.Text(label='Ordinal speakers')
            named_speakers = gr.Text(label='Named speakers')
            input_lang = gr.Text(label='input language')

        with gr.Column(scale=1) as col1:
            text = gr.Text(label='Text transcription', interactive=True)
            detected_lang = gr.Text(label='Detected language')
            num_chars = gr.Number(label='Number of characters')
            transcribe_button = gr.Button(value='Transcribe!')

        detect_spkr_button.click(detect_speakers, inputs=[file, project_name, email], outputs=[detected_speakers])
        transcribe_button.click(transcribe,
                                inputs=[file, input_lang, named_speakers],
                                outputs=[text, detected_lang, num_chars])
