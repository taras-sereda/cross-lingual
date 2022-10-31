import warnings
from datetime import datetime

import gradio as gr
import librosa
import soundfile as sf
from sqlalchemy.orm import Session
from tortoise.utils.text import split_and_recombine_text

from . import example_text, example_voice_sample_path, schemas, crud
from .database import SessionLocal
from .models import User, Project, Utterance, Speaker

MAX_UTTERANCE = 20


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
        crud.update_any_db_row(db, utterance, date_completed=datetime.now())

    crud.update_any_db_row(db, project, date_completed=datetime.now())

    res = load(project, db=db)

    db.close()
    return res


def load(project: str | Project, user_email: str | None = None, db: Session | None = None):
    #TODO. Think more.
    # This might be dangerous, I'm creating a new db session if there is no one provided and not closing it.
    # So potentillay after a lot of load calls from UI i'd have orphan sessions, so far this is not harmful though.
    if not db:
        db = SessionLocal()
    if isinstance(project, str):
        project_name = project
        user: User = crud.get_user_by_email(db, user_email)
        if not user:
            raise Exception(f"User {user_email} not found. Provide valid email")

        project = crud.get_project_by_title(db, project, user.id)
        if not project:
            raise Exception(f"no such project {project_name}. Provice valid project title")

    res = []
    for utterance in project.utterances:
        res.append(gr.Textbox.update(value=utterance.text, visible=True))
        res.append(gr.Number.update(value=utterance.utterance_idx))
        res.append(gr.Textbox.update(value=utterance.speaker.name, visible=True))

        gen, sample_rate = sf.read(utterance.get_audio_path())
        res.append(gr.Audio.update(value=(sample_rate, gen), visible=True))
        res.append(gr.Button.update(visible=True))

    n_utterance = len(project.utterances)

    # padding
    if (delta := MAX_UTTERANCE - n_utterance) > 0:
        for _ in range(delta):
            res.append(gr.Textbox.update(visible=False))
            res.append(gr.Number.update(visible=False))
            res.append(gr.Textbox.update(visible=False))
            res.append(gr.Audio.update(visible=False))
            res.append(gr.Button.update(visible=False))
    # clipping
    else:
        warnings.warn("Output is clipped!")
    return res


def dummy_reread(title, text, utterance_idx, speaker_name, user_email):
    utterance_idx = int(utterance_idx)
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

    gen, sample_rate = librosa.load(librosa.example('pistachio'))
    sf.write(utterance_db.get_audio_path(), gen, sample_rate)
    update_dict = {
        'text': text,
        'speaker_id': new_speaker.id,
        'date_created' : start_time,
        'date_completed': datetime.now()
    }
    crud.update_any_db_row(db, utterance_db, **update_dict)
    db.close()
    return (sample_rate, gen), new_speaker.name


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

        outputs = []
        with gr.Column(scale=1, variant='compact') as col2:
            for i in range(MAX_UTTERANCE):
                utterance = gr.Textbox(label=f'utterance_{i}', visible=False)
                utterance_idx = gr.Number(visible=False)
                utter_speaker = gr.Textbox(label=f'speaker name', visible=False)
                audio = gr.Audio(label=f'audio_{i}', visible=False)

                try_again = gr.Button(value='try again', visible=False)
                try_again.click(fn=dummy_reread,
                                inputs=[title, utterance, utterance_idx, utter_speaker, user_email],
                                outputs=[audio, utter_speaker])

                outputs.extend([utterance, utterance_idx, utter_speaker, audio, try_again])

        button.click(fn=dummy_read, inputs=[title, text, user_email], outputs=outputs)
        button_load.click(fn=load, inputs=[title, user_email], outputs=outputs)

    gr.Markdown("Text examples")
    gr.Examples([example_text], [text])
    gr.Markdown("Audio examples")
    gr.Examples([example_voice_sample_path], [reference_audio])


if __name__ == '__main__':
    editor.launch(debug=True)
