from typing import Tuple

import gradio as gr
import torch
import numpy as np
import whisper
from torchaudio.transforms import Resample
from pyannote.audio import Pipeline


from . import cfg, example_voice_sample_path

stt_model = whisper.load_model(cfg.stt.model_size)
diarization_model = Pipeline.from_pretrained(cfg.diarization.model_name, use_auth_token=cfg.diarization.auth_token)


def transcribe(audio_data: Tuple[int, np.ndarray]):
    """Transcribe input media with speaker diarization, resulting transcript will be in form:
    [ HH:MM:SS.sss --> HH:MM:SS.sss ]
    {SPEAKER}
    Transcribed text
    """

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

    segments = []
    for seg, _, speaker in diarizations.itertracks(yield_label=True):
        seg_wav = stt_waveform[int(seg.start*cfg.stt.sample_rate): int(seg.end*cfg.stt.sample_rate)]
        seg_trans_res = stt_model.transcribe(seg_wav)
        segments.append([seg, speaker, seg_trans_res['text'], seg_trans_res['language']])
    res = ''
    for seg in segments:
        res += f"{seg[0]}\n{{{seg[1]}}}\n{seg[2]}\n\n"

    return res, segments[0][-1]


with gr.Blocks() as transcriber:
    with gr.Row() as row0:
        with gr.Column(scale=1) as col0:
            project_name = gr.Text(label='Project name', placeholder="enter your project name")
            audio = gr.Audio(label='Audio for transcription')

        with gr.Column(scale=1) as col1:
            text = gr.Text(label='Text transcription', interactive=True)
            lang = gr.Text(label='Detected language')
            button = gr.Button(value='Go!')
        button.click(transcribe, inputs=[audio], outputs=[text, lang])

    gr.Markdown("Audio examples")
    gr.Examples([example_voice_sample_path], [audio])
