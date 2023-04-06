import datetime

import gspread
import gradio as gr
import numpy as np

from sqlalchemy.orm import Session

from db.database import SessionLocal
from editor_app.tts import read, combine, add_tgt_media_components, get_translation_wrapped
from editor_app.common import get_cross_projects
from editor_app.stt import transcribe, save_transcript, add_src_media_components, calculate_project_score
from editor_app.translator import gradio_translate, save_translation
from media_utils import get_youtube_embed_code, download_media, download_rss, media_extensions
from rss_utils import get_audio_link
from network_utils import upload_youtube_video
from string_utils import get_random_string
from utils import get_user_from_request, build_youtube_link_from_iframe
from datatypes import Cells, PseudoFile


def end2end_pipeline(file, media_link, project_name, src_lang, tgt_lang, options, request: gr.Request):
    src_text, src_lang, _, *src_components = transcribe(file, media_link, project_name, src_lang, options, request)
    _ = save_transcript(project_name, src_text, src_lang, request)
    _, tgt_text, *_ = gradio_translate(project_name, tgt_lang, request)
    _ = save_translation(project_name, tgt_text, tgt_lang, request)
    _ = read(project_name, tgt_lang, tgt_text, request)
    tgt_components = combine(project_name, tgt_lang, request)
    res = [*src_components, *tgt_components]

    user_email = get_user_from_request(request)
    db: Session = SessionLocal()
    translation_db = get_translation_wrapped(project_name, tgt_lang, db, user_email)
    prj_score, _ = calculate_project_score(db, translation_db)
    output_media_link = upload_youtube_video(translation_db)
    if output_media_link:
        iframe_val = get_youtube_embed_code(output_media_link)
        res.append(gr.HTML.update(value=iframe_val))
    else:
        res.append(gr.HTML.update(visible=False))

    res.append(gr.Number.update(value=prj_score, visible=True))
    return res


def load_src_and_tgt(cross_project_name, media_link, tgt_lang, request: gr.Request):
    user_email = get_user_from_request(request)
    db = SessionLocal()
    translation_db = get_translation_wrapped(cross_project_name, tgt_lang, db, user_email)

    res = add_src_media_components(translation_db.cross_project, media_link)
    res.extend(add_tgt_media_components(translation_db))
    return res


def run_bulk_processing(options, request: gr.Request):
    gc = gspread.service_account()
    sh = gc.open("cross-lingual-bulk-demos")

    data = np.array(sh.sheet1.get_all_values())
    for idx, row in enumerate(data):
        row_idx = idx + 1

        src_url_idx = Cells.A.value
        tgt_url_idx = Cells.C.value
        date_coll_idx = Cells.D.value
        status_coll_idx = Cells.E.value
        score_coll_idx = Cells.F.value

        row_status = row[status_coll_idx - 1]
        if row_status:
            continue

        media_link = row[src_url_idx - 1]

        file, youtube_link = None, None
        try:
            if any(ext in media_link for ext in media_extensions):
                file_path = download_media(media_link, save_path="/tmp")
                file = PseudoFile(name=file_path)
            elif 'youtube' in media_link:
                youtube_link = media_link
            else:
                rss_file_path = download_rss(media_link, save_path="/tmp")
                audio_url = get_audio_link(rss_file_path)
                file_path = download_media(audio_url, save_path="/tmp")
                file = PseudoFile(name=file_path)

            project_name = f"bulk_project_{get_random_string()}"
            src_lang = ""
            tgt_lang = "EN-US"
            *_, gr_tgt_iframe, gr_prj_score = end2end_pipeline(file, youtube_link, project_name, src_lang, tgt_lang, options, request)
        except:
            sh.sheet1.update_cell(row_idx, status_coll_idx, "Failed")
            raise ValueError(f"Unrecognized media source {media_link}")

        sh.sheet1.update_cell(row_idx, tgt_url_idx, build_youtube_link_from_iframe(gr_tgt_iframe["value"]))
        sh.sheet1.update_cell(row_idx, date_coll_idx, str(datetime.datetime.now()))
        sh.sheet1.update_cell(row_idx, status_coll_idx, "Done")
        sh.sheet1.update_cell(row_idx, score_coll_idx, gr_prj_score["value"])


with gr.Blocks() as e2e:
    with gr.Row() as row0:
        with gr.Column(scale=1, variant='panel') as col0:
            user_projects = gr.Dataframe(label='user projects')
            project_name = gr.Text(label='Project name', placeholder="enter your project name, can't be empty.")
            media_link = gr.Text(label='Youtube Link', placeholder='Link to youtube video.', value=None)
            file = gr.File(label='input media')
            with gr.Row():
                src_lang = gr.Text(label='Source language', placeholder="Provide medial language, if it's known, "
                                                                        "otherwise language will be detected. "
                                                                        "Example: Uk, En, De.")
                tgt_lang = gr.Text(label='Target Language', value="EN-US")
            options = gr.CheckboxGroup(label='transcription options',
                                       choices=['Demo Run', 'Save speakers', 'Upload to youtube'],
                                       value=['Demo Run', 'Save speakers', 'Upload to youtube'])
            with gr.Row():
                load_button = gr.Button(value="Load")
                go_button = gr.Button(value="CrossLingualize 👾")
                bulk_button = gr.Button(value="Bulk processing")

        with gr.Column(scale=1) as col1:
            src_iframe = gr.HTML(label='source youtube video')
            src_audio = gr.Audio(visible=False)
            src_video = gr.Video(visible=False)

            tgt_audio = gr.Audio(visible=False)
            tgt_video = gr.Video(visible=False)
            tgt_iframe = gr.HTML(label='CrossLingual youtube video')
            tgt_score = gr.Number(visible=False)

        go_button.click(
            end2end_pipeline,
            inputs=[file, media_link, project_name, src_lang, tgt_lang, options],
            outputs=[src_iframe, src_audio, src_video, tgt_audio, tgt_video, tgt_iframe, tgt_score],
            api_name="end2end")

        load_button.click(
            load_src_and_tgt,
            inputs=[project_name, media_link, tgt_lang],
            outputs=[src_iframe, src_audio, src_video, tgt_audio, tgt_video])

        bulk_button.click(
            run_bulk_processing,
            inputs=[options])

    e2e.load(get_cross_projects, outputs=[user_projects])

if __name__ == '__main__':
    e2e.launch(debug=True)
