import gradio as gr

from editor_app.database import SessionLocal
from editor_app.tts import read, combine, add_tgt_media_components, get_translation_wrapped
from editor_app.common import get_cross_projects
from editor_app.stt import transcribe, save_transcript, add_src_media_components
from editor_app.translator import gradio_translate, save_translation
from utils import get_user_from_request


def end2end_pipeline(file, media_link, project_name, src_lang, tgt_lang, options, request: gr.Request):
    src_text, src_lang, _, *src_components = transcribe(file, media_link, project_name, src_lang, options, request)
    _ = save_transcript(project_name, src_text, src_lang, request)
    _, tgt_text, *_ = gradio_translate(project_name, tgt_lang, request)
    _ = save_translation(project_name, tgt_text, tgt_lang, request)
    _ = read(project_name, tgt_lang, tgt_text, request)
    tgt_components = combine(project_name, tgt_lang, request)
    return [*src_components, *tgt_components]


def load_src_and_tgt(cross_project_name, media_link, tgt_lang, request: gr.Request):
    user_email = get_user_from_request(request)
    db = SessionLocal()
    translation_db = get_translation_wrapped(cross_project_name, tgt_lang, db, user_email)

    res = add_src_media_components(translation_db.cross_project, media_link)
    res.extend(add_tgt_media_components(translation_db))
    return res


with gr.Blocks() as e2e:
    with gr.Row() as row0:
        with gr.Column(scale=1, variant='panel') as col0:
            user_projects = gr.Dataframe(label='user projects')
            project_name = gr.Text(label='Project name', placeholder="enter your project name, can't be empty.")
            media_link = gr.Text(label='Youtube Link', placeholder='Link to youtube video.')
            file = gr.File(label='input media')
            with gr.Row():
                src_lang = gr.Text(label='Source language', placeholder="Provide medial language, if it's known, "
                                                                        "otherwise language will be detected. "
                                                                        "Example: Uk, En, De.")
                tgt_lang = gr.Text(label='Target Language', value="EN-US")
            options = gr.CheckboxGroup(label='transcription options',
                                       choices=['Demo Run', 'Save speakers'],
                                       value=['Demo Run', 'Save speakers'])
            with gr.Row():
                load_button = gr.Button(value="Load")
                go_button = gr.Button(value="CrossLingualize 👾")

        with gr.Column(scale=1) as col1:
            src_iframe = gr.HTML(label='youtube video')
            src_audio = gr.Audio(visible=False)
            src_video = gr.Video(visible=False)

            tgt_audio = gr.Audio(visible=False)
            tgt_video = gr.Video(visible=False)

        go_button.click(
            end2end_pipeline,
            inputs=[file, media_link, project_name, src_lang, tgt_lang, options],
            outputs=[src_iframe, src_audio, src_video, tgt_audio, tgt_video])

        load_button.click(
            load_src_and_tgt,
            inputs=[project_name, media_link, tgt_lang],
            outputs=[src_iframe, src_audio, src_video, tgt_audio, tgt_video])

    e2e.load(get_cross_projects, outputs=[user_projects])

if __name__ == '__main__':
    e2e.launch(debug=True)
