import warnings
from datetime import datetime

import gradio as gr
import librosa
import soundfile as sf
from tortoise.utils.text import split_and_recombine_text

from . import example_text, example_voice_sample_path, schemas
from .crud_gradio import get_user_by_mail_gradio, create_speaker_gradio, create_project_gradio
from .models import User, Project


MAX_UTTERANCE = 20


def add_speaker(audio_tuple, speaker_name, user_email):
    user: User = get_user_by_mail_gradio(user_email)
    if not user:
        return f"User {user_email} not found. Provide valid email"

    if speaker_name in get_speakers(user_email):
        return f"Speaker {speaker_name} already exists!"

    # write data to db
    speaker = create_speaker_gradio(speaker_name, user.id)

    # save data on disk
    wav_path = speaker.get_speaker_data_root().joinpath('1.wav')
    sample_rate, audio = audio_tuple
    sf.write(wav_path, audio, sample_rate)


def get_speakers(user_email):
    user: User = get_user_by_mail_gradio(user_email)
    # return all available speakers
    return [speaker.name for speaker in user.speakers]


def dummy_read(title, text, user_email):
    user: User = get_user_by_mail_gradio(user_email)
    if not user:
        return f"User {user_email} not found. Provide valid email"

    data = schemas.ProjectCreate(title=title, text=text, date_created=datetime.now())
    project = create_project_gradio(data, user.id)

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


with gr.Blocks() as editor:
    with gr.Row() as row0:
        with gr.Column(scale=1) as col0:
            user_email = gr.Text(label='user', placeholder='Enter user email', value='taras.y.sereda@proton.me')

            reference_audio = gr.Audio(label='reference audio')
            speaker_name = gr.Textbox(label='Speaker name', placeholder='Enter speaker name')
            add_speaker_button = gr.Button('Add speaker')
            speakers = gr.Textbox(label='speakers')
            errors = gr.Textbox(label='error messages')
            get_speakers_button = gr.Button('Get speakers')
            add_speaker_button.click(add_speaker, inputs=[reference_audio, speaker_name, user_email], outputs=[errors])
            get_speakers_button.click(get_speakers, inputs=[user_email], outputs=[speakers])

        with gr.Column(scale=1) as col1:
            button_create = gr.Button("Create!")
            button_load = gr.Button("Load")

            title = gr.Text(label='Title', placeholder="enter your project title", value="2_B_R_0_2_B")
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

        button.click(fn=dummy_read, inputs=[title, text, user_email], outputs=outputs)

    gr.Markdown("Text examples")
    gr.Examples([example_text], [text])
    gr.Markdown("Audio examples")
    gr.Examples([example_voice_sample_path], [reference_audio])
