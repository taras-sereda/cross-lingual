from datetime import datetime

import deepl
import gradio as gr
from sqlalchemy.orm import Session

from config import cfg
from db import crud, schemas
from db.database import SessionLocal
from editor_app.common import get_cross_projects
from editor_app.stt import load_transcript
from string_utils import validate_and_preprocess_title
from utils import split_on_raw_utterances, get_user_from_request

deepl_translator = deepl.Translator(cfg.translation.auth_token)


def translate(text: str, tgt_lang: str):
    """Translate
    """
    segments = split_on_raw_utterances(text)
    num_src_char = 0
    num_tgt_char = 0

    result = ''

    for seg in segments:

        seg_res = deepl_translator.translate_text(seg.text, target_lang=tgt_lang)
        num_src_char += len(seg.text)
        num_tgt_char += len(seg_res.text)

        components = []
        if seg.timecode is not None:
            components.append(seg.timecode)
        components.extend([f'{{{seg.speaker}}}', seg_res.text, '\n'])
        line = '\n'.join(components)
        result += line

    return result, num_src_char, num_tgt_char


def get_num_char(text: str):
    segments = split_on_raw_utterances(text)
    num_char = 0
    for seg in segments:
        num_char += len(seg.text)
    return num_char


def gradio_translate(project_name, tgt_lang, request: gr.Request):
    user_email = get_user_from_request(request)
    project_name = validate_and_preprocess_title(project_name)
    db: Session = SessionLocal()
    user = crud.get_user_by_email(db, user_email)
    cross_project = crud.get_cross_project_by_title(db, project_name, user.id)
    # TODO. ensure one-to-one relationship.
    assert len(cross_project.transcript) == 1
    transcript = cross_project.transcript[0]
    translation_db = crud.get_translation_by_title_and_lang(db, project_name, tgt_lang, user.id)
    if translation_db is None:
        translation, num_src_char, num_tgt_char = translate(transcript.text, tgt_lang)
    else:
        translation = translation_db.text

    num_src_char = get_num_char(transcript.text)
    num_tgt_char = get_num_char(translation)
    return transcript.text, translation, num_src_char, num_tgt_char


def save_translation(project_name, text, lang, request: gr.Request):
    user_email = get_user_from_request(request)
    project_name = validate_and_preprocess_title(project_name)
    db: Session = SessionLocal()
    user = crud.get_user_by_email(db, user_email)
    cross_project = crud.get_cross_project_by_title(db, project_name, user.id, ensure_exists=True)
    translation_db = crud.get_translation_by_title_and_lang(db, project_name, lang, user.id)

    if translation_db is None:
        translation_data = schemas.TranslationCreate(text=text, lang=lang, date_created=datetime.now())
        translation_db = crud.create_translation(db, translation_data, cross_project.id)
    else:
        crud.update_any_db_row(db, translation_db, text=text, lang=lang, date_created=datetime.now())

    with open(translation_db.get_path(), 'w') as f:
        f.write(text)

    num_tgt_char = get_num_char(translation_db.text)
    return num_tgt_char


with gr.Blocks() as translator:
    with gr.Row() as row0:
        with gr.Column(scale=1, variant='panel') as col0:
            user_projects = gr.Dataframe(label='user projects')
            with gr.Row():
                project_name = gr.Text(label='Project name', placeholder="enter your project name")
                tgt_lang = gr.Text(label='Target Language', value='EN-US')
            src_text = gr.Text(label='Text transcription', interactive=True)

        with gr.Column(scale=1) as col1:
            tgt_text = gr.Text(label='Text translation', interactive=True)
            with gr.Row():
                num_src_chars = gr.Number(label='[Source language] Number of characters')
                num_tgt_chars = gr.Number(label='[Target language] Number of characters')
            with gr.Row():
                load_button = gr.Button(value='Load Transcript')
                translate_button = gr.Button(value='Translate')
                save_button = gr.Button(value='Save Translation')

        load_button.click(load_transcript, inputs=[project_name], outputs=[project_name, src_text])
        translate_button.click(gradio_translate, inputs=[project_name, tgt_lang], outputs=[src_text, tgt_text, num_src_chars, num_tgt_chars])
        save_button.click(save_translation, inputs=[project_name, tgt_text, tgt_lang], outputs=[num_tgt_chars])

    translator.load(get_cross_projects, outputs=[user_projects])

if __name__ == '__main__':
    translator.launch(debug=True)
