import warnings
from datetime import datetime

import gradio as gr
import soundfile as sf
import torch
import numpy as np
from sqlalchemy.orm import Session
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voices
from tortoise.utils.text import split_and_recombine_text

from utils import split_on_speaker_change
from . import example_text, example_voice_sample_path, schemas, crud, cfg
from .database import SessionLocal
from .models import User, Project, Utterance, Speaker

tts = TextToSpeech()

if torch.cuda.is_available():
    preset = 'standard'
else:
    preset = 'ultra_fast'


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
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    if not user:
        raise Exception(f"User {user_email} not found. Provide valid email")
    if crud.get_project_by_title(db, title, user.id):
        raise Exception(f"Project {title} already exists! Try to load it")

    speakers_to_features = dict()
    data = []
    for spkr_name, spkr_text in split_on_speaker_change(raw_text):
        db_speaker: Speaker = crud.get_speaker_by_name(db, spkr_name, user.id)
        if not db_speaker:
            raise Exception(f"No such speaker {spkr_name} {db_speaker}")

        # load voice samples and conditioning latents.
        if db_speaker.name not in speakers_to_features:
            spkr_data_root = db_speaker.get_speaker_data_root().parent
            voice_samples, conditioning_latents = load_voices([db_speaker.name], [spkr_data_root])
            speakers_to_features[db_speaker.name] = {
                'id': db_speaker.id,
                'voice_samples': voice_samples,
                'conditioning_latents': conditioning_latents,}

        texts = split_and_recombine_text(spkr_text)
        for text in texts:
            data.append((db_speaker.name, text))

    project_data = schemas.ProjectCreate(title=title, text=raw_text, date_created=datetime.now())
    project = crud.create_project(db, project_data, user.id)

    for idx, (spkr_name, spkr_text) in enumerate(data):

        gen = tts.tts_with_preset(spkr_text,
                                  voice_samples=speakers_to_features[spkr_name]['voice_samples'],
                                  conditioning_latents=speakers_to_features[spkr_name]['conditioning_latents'],
                                  preset=preset, k=cfg.tts.candidates, use_deterministic_seed=cfg.tts.seed)
        gen = gen.cpu().numpy().squeeze()

        utterance_data = schemas.UtteranceCreate(text=spkr_text, utterance_idx=idx, date_started=datetime.now())
        utterance: Utterance = crud.create_utterance(db, utterance_data, project.id, speakers_to_features[spkr_name]['id'])
        sf.write(utterance.get_audio_path(), gen, cfg.tts.sample_rate)
        crud.update_any_db_row(db, utterance, date_completed=datetime.now())

    crud.update_any_db_row(db, project, date_completed=datetime.now())
    res = load(project, db=db)
    db.close()
    return res


def load(project: str | Project, user_email: str | None = None, db: Session | None = None):
    #TODO. Think more.
    # This might be dangerous, I'm creating a new db session if there is no one provided and not closing it.
    # So potentially after a lot of load calls from UI i'd have orphan sessions, so far this is not harmful though.
    if not db:
        db = SessionLocal()
    if isinstance(project, str):
        project_name = project
        user: User = crud.get_user_by_email(db, user_email)
        if not user:
            raise Exception(f"User {user_email} not found. Provide valid email")

        project = crud.get_project_by_title(db, project, user.id)
        if not project:
            raise Exception(f"no such project {project_name}. Provide valid project title")

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
    if (delta := cfg.editor.max_utterance - n_utterance) > 0:
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


def reread(title, text, utterance_idx, speaker_name, user_email):
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
    voice_samples, conditioning_latents = load_voices([speaker_name], [new_speaker.get_speaker_data_root().parent])
    gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                              preset=preset, k=cfg.tts.candidates, use_deterministic_seed=cfg.tts.seed)
    gen = gen.cpu().numpy().squeeze()

    sf.write(utterance_db.get_audio_path(), gen, cfg.tts.sample_rate)
    update_dict = {
        'text': text,
        'speaker_id': new_speaker.id,
        'date_created' : start_time,
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

    project_audio = []
    for utterance in project.utterances:
        utter_audio, sample_rate = sf.read(utterance.get_audio_path())
        project_audio.append(utter_audio)
    db.close()
    project_audio = np.concatenate(project_audio)

    return gr.Audio.update(value=(sample_rate, project_audio), visible=True)


with gr.Blocks() as editor:
    with gr.Row() as row0:
        with gr.Column(scale=1) as col0:
            email = gr.Text(label='user', placeholder='Enter user email', value='taras.y.sereda@proton.me')

            reference_audio = gr.Audio(label='reference audio')
            speaker_name = gr.Textbox(label='Speaker name', placeholder='Enter speaker name, allowed symbols: lower case letters, numbers, and _')
            add_speaker_button = gr.Button('Add speaker')
            speakers = gr.Textbox(label='speakers')
            errors = gr.Textbox(label='error messages')
            get_speakers_button = gr.Button('Get speakers')
            user_projects = gr.Textbox(label='user projects')
            get_user_projects_button = gr.Button('Get projects')
            add_speaker_button.click(add_speaker, inputs=[reference_audio, speaker_name, email], outputs=[errors])
            get_speakers_button.click(get_speakers, inputs=[email], outputs=[speakers])
            get_user_projects_button.click(get_projects, inputs=[email], outputs=[user_projects])

        with gr.Column(scale=1) as col1:
            project_title = gr.Text(label='Title', placeholder="enter your project title")
            project_text = gr.Text(label='Text for synthesis', interactive=True)

            button = gr.Button(value='Go!', variant='primary')
            button_load = gr.Button(value='Load')
            button_combine = gr.Button(value='Combine')
            combined_audio = gr.Audio(visible=False)

        outputs = []
        with gr.Column(scale=1, variant='compact') as col2:
            for i in range(cfg.editor.max_utterance):
                utterance = gr.Textbox(label=f'utterance_{i}', visible=False)
                utterance_idx = gr.Number(visible=False)
                utter_speaker = gr.Textbox(label=f'speaker name', visible=False)
                audio = gr.Audio(label=f'audio_{i}', visible=False)

                try_again = gr.Button(value='try again', visible=False)
                try_again.click(fn=reread,
                                inputs=[project_title, utterance, utterance_idx, utter_speaker, email],
                                outputs=[audio, utter_speaker])

                outputs.extend([utterance, utterance_idx, utter_speaker, audio, try_again])

        button.click(fn=read, inputs=[project_title, project_text, email], outputs=outputs)
        button_load.click(fn=load, inputs=[project_title, email], outputs=outputs)
        button_combine.click(fn=combine, inputs=[project_title, email], outputs=[combined_audio])

    gr.Markdown("Text examples")
    gr.Examples([example_text], [project_text])
    gr.Markdown("Audio examples")
    gr.Examples([example_voice_sample_path], [reference_audio])


if __name__ == '__main__':
    editor.launch(debug=True)

# TODO add combination of synthesized segments
# TODO add playground, where one can play with different ways of pronunciation of a particular word
# TODO integrate whisper for judging of synthesis quality.
