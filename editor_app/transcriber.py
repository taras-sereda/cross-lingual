from typing import Tuple

import gradio as gr
import torch
import numpy as np
import soundfile as sf

from torchaudio.transforms import Resample
from pyannote.audio import Pipeline

from . import cfg, data_root, example_voice_sample_path
from .stt import stt_model

diarization_model = Pipeline.from_pretrained(cfg.diarization.model_name, use_auth_token=cfg.diarization.auth_token)

temp_dir_path = data_root.joinpath('temp')
temp_dir_path.mkdir(exist_ok=True)


def transcribe(audio_data: Tuple[int, np.ndarray], language):
    """Transcribe input media with speaker diarization, resulting transcript will be in form:
    [ HH:MM:SS.sss --> HH:MM:SS.sss ]
    {SPEAKER}
    Transcribed text
    """
    if len(language) == 0:
        language = None

    sample_rate, waveform = audio_data
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    if waveform.ndim == 2 and np.argmin(waveform.shape) == 1:
        waveform = waveform.transpose()
    waveform = torch.from_numpy(waveform).to(torch.float32)

    diarizations = diarization_model({'waveform': waveform, 'sample_rate': sample_rate})

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
            project_name = gr.Text(label='Project name', placeholder="enter your project name")
            audio = gr.Audio(label='Audio for transcription')
            input_lang = gr.Text(label='input language')

        with gr.Column(scale=1) as col1:
            text = gr.Text(label='Text transcription', interactive=True)
            detected_lang = gr.Text(label='Detected language')
            num_chars = gr.Number(label='Number of characters')
            button = gr.Button(value='Go!')
        button.click(transcribe, inputs=[audio, input_lang], outputs=[text, detected_lang, num_chars])

    gr.Markdown("Audio examples")
    gr.Examples([example_voice_sample_path], [audio])

# TODO. Add speakers mapping
