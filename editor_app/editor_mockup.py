import warnings

import gradio as gr
import librosa
from tortoise.utils.text import split_and_recombine_text
from sqlalchemy.orm import Session

from . import example_text, example_voice_sample_path
from . import crud
from .database import SessionLocal

MAX_UTTERANCE = 20


def dummy_add_speaker(audio_tuple, speaker_name):
    sample_rate, audio = audio_tuple
    # save data on disk

    # save data in db
    available_speakers = f"""
    1. {speaker_name}
    """
    return available_speakers


def dummy_read(text):
    texts = split_and_recombine_text(text)
    res = []
    for text in texts:
        gen, sample_rate = librosa.load(librosa.example('brahms'))
        res.append(gr.Textbox.update(value=text, visible=True))
        res.append(gr.Audio.update(value=(sample_rate, gen), visible=True))
        res.append(gr.Button.update(visible=True))
    n_utterance = len(texts)

    # padding
    if (delta := MAX_UTTERANCE - n_utterance) > 0:
        for _ in range(delta):
            res.append(gr.Textbox.update(visible=False))
            res.append(gr.Audio.update(visible=False))
            res.append(gr.Button.update(visible=False))
    # clipping
    else:
        warnings.warn("Output is clipped!")
        pass

    return [len(texts)] + res


def dummy_reread(text):
    gen, sample_rate = librosa.load(librosa.example('pistachio'))
    return sample_rate, gen


def get_users_gradio():
    db: Session = SessionLocal()
    users = crud.get_users(db=db)
    db.close()

    return [user.email for user in users]


with gr.Blocks() as editor:
    with gr.Row() as row0:
        with gr.Column(scale=1) as col0:
            users = gr.Text(value=get_users_gradio, label='users')

            reference_audio = gr.Audio(label='reference audio')
            speaker_name = gr.Textbox(value="joe", label='Speaker name')
            add_speaker_button = gr.Button('Add speaker')
            gr.Markdown("### Available Speakers")
            available_speakers = gr.Markdown()

            add_speaker_button.click(dummy_add_speaker, inputs=[reference_audio, speaker_name],
                                     outputs=[available_speakers])

        with gr.Column(scale=1) as col1:
            button_create = gr.Button("Create!")
            button_load = gr.Button("Load")

            project_name = gr.Text(label='Project name', placeholder="enter your project name")
            text = gr.Text(label='Text for synthesis')
            button = gr.Button(value='Go!')
            outputs = [gr.Number(label='number of utterances')]

        with gr.Column(scale=1, variant='compact') as col2:
            for i in range(MAX_UTTERANCE):
                utterance = gr.Textbox(label=f'utterance_{i}', visible=False)
                audio = gr.Audio(label=f'audio_{i}', visible=False)

                try_again = gr.Button(value='try again', visible=False)
                try_again.click(fn=dummy_reread, inputs=[utterance], outputs=[audio])

                outputs.extend([utterance, audio, try_again])

        button.click(fn=dummy_read, inputs=[text], outputs=outputs)

    gr.Markdown("Text examples")
    gr.Examples([example_text], [text])
    gr.Markdown("Audio examples")
    gr.Examples([example_voice_sample_path], [reference_audio])
