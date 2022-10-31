import warnings
from datetime import datetime

import gradio as gr
import librosa
import soundfile as sf
from sqlalchemy.orm import Session
from tortoise.utils.text import split_and_recombine_text

from . import example_text, example_voice_sample_path, schemas, crud
from .database import SessionLocal
from .models import User, Project, Utterance

MAX_UTTERANCE = 20


def add_speaker(audio_tuple, speaker_name, user_email):
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    if not user:
        raise Exception(f"User {user_email} not found. Provide valid email")

    if speaker_name in get_speakers(db, user_email):
        raise Exception(f"Speaker {speaker_name} already exists!")

    # write data to db
    speaker = crud.create_speaker(db, speaker_name, user.id)

    # save data on disk
    wav_path = speaker.get_speaker_data_root().joinpath('1.wav')
    sample_rate, audio = audio_tuple
    sf.write(wav_path, audio, sample_rate)
    db.close()


def get_speakers(db, user_email):
    user: User = crud.get_user_by_email(db, user_email)
    # return all available speakers
    return [speaker.name for speaker in user.speakers]


def dummy_read(title, text, user_email):
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    if not user:
        raise Exception(f"User {user_email} not found. Provide valid email")
    if crud.get_project_by_title(db, title, user.id):
        raise Exception(f"Project {title} already exists! Try to load it")

    project_data = schemas.ProjectCreate(title=title, text=text, date_created=datetime.now())

    project = crud.create_project(db, project_data, user.id)

    texts = split_and_recombine_text(project.text)
    speaker_name = 'joe_rogan'
    unique_speakers_dict = {
        speaker_name: crud.get_speaker_by_name(db, speaker_name, user.id)
    }

    for utter_idx, text in enumerate(texts):
        gen, sample_rate = librosa.load(librosa.example('brahms'))

        utterance_data = schemas.UtteranceCreate(text=text, utterance_idx=utter_idx, date_started=datetime.now())
        utterance: Utterance = crud.create_utterance(db, utterance_data, project.id, unique_speakers_dict[speaker_name].id)
        sf.write(utterance.get_audio_path(), gen, sample_rate)

        # TODO here synthesis should be started
        utterance.date_completed = datetime.now()

    res = load(project, db=db)

    project.date_completed = datetime.now()
    db.close()
    return [len(texts)] + res


def load(project: str | Project, user_email: str | None = None, db: Session | None = None):
    if not db:
        db = SessionLocal()
    if isinstance(project, str):
        user: User = crud.get_user_by_email(db, user_email)
        project = crud.get_project_by_title(db, project, user.id)

    res = []
    for utterance in project.utterances:
        res.append(gr.Textbox.update(value=utterance.text, visible=True))
        gen, sample_rate = sf.read(utterance.get_audio_path())
        res.append(gr.Audio.update(value=(sample_rate, gen), visible=True))
        res.append(gr.Button.update(visible=True))

    n_utterance = len(project.utterances)

    # padding
    if (delta := MAX_UTTERANCE - n_utterance) > 0:
        for _ in range(delta):
            res.append(gr.Textbox.update(visible=False))
            res.append(gr.Audio.update(visible=False))
            res.append(gr.Button.update(visible=False))
    # clipping
    else:
        warnings.warn("Output is clipped!")
    return res


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


if __name__ == '__main__':
    editor.launch(debug=True)
