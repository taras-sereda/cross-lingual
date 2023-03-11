from datetime import datetime

import deepl
import gradio as gr
from sqlalchemy.orm import Session

from editor_app import cfg, crud, schemas, html_menu
from editor_app.database import SessionLocal
from editor_app.tts import get_cross_projects
from utils import split_on_raw_utterances

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


def gradio_translate(project_name, tgt_lang, user_email):
    db: Session = SessionLocal()
    user = crud.get_user_by_email(db, user_email)
    cross_project = crud.get_cross_project_by_title(db, project_name, user.id)

    # TODO. ensure one-to-one relationship.
    assert len(cross_project.transcript) == 1
    transcript = cross_project.transcript[0]

    translation_db = crud.get_translation_by_title_and_lang(db, project_name, tgt_lang, user.id)
    if translation_db is None:
        translation, num_src_char, num_tgt_char = translate(transcript.text, tgt_lang)
        translation_data = schemas.TranslationCreate(text=translation, lang=tgt_lang, date_created=datetime.now())
        translation_db = crud.create_translation(db, translation_data, cross_project.id)
        with open(translation_db.get_path(), 'w') as f:
            f.write(translation)
    else:
        num_src_char = get_num_char(transcript.text)
        num_tgt_char = get_num_char(translation_db.text)
        print('translation loaded from db')

    return transcript.text, translation_db.text, num_src_char, num_tgt_char


with gr.Blocks() as translator:
    with gr.Row() as row0:
        with gr.Column(scale=1) as col0:
            menu = gr.HTML(html_menu)
            email = gr.Text(label='user', placeholder='Enter user email', value=cfg.user.email)
            user_projects = gr.Dataframe(label='user projects')

            project_name = gr.Text(label='Project name', placeholder="enter your project name")
            src_text = gr.Text(label='Text transcription', interactive=True)
            tgt_lang = gr.Text(label='Target Language', value='EN-US')
        with gr.Column(scale=1) as col1:
            tgt_text = gr.Text(label='Text translation', interactive=True)
            num_src_chars = gr.Number(label='[Source language] Number of characters')
            num_tgt_chars = gr.Number(label='[Target language] Number of characters')
            button = gr.Button(value='Go!')
            button2 = gr.Button(value='Load and go!')
        button.click(translate, inputs=[src_text, tgt_lang], outputs=[tgt_text, num_src_chars, num_tgt_chars])
        button2.click(gradio_translate, inputs=[project_name, tgt_lang, email], outputs=[src_text, tgt_text, num_src_chars, num_tgt_chars])

    translator.load(get_cross_projects, inputs=[email], outputs=[user_projects])

if __name__ == '__main__':
    translator.launch(debug=True)
