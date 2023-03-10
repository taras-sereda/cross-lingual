import tempfile

import gradio as gr
import soundfile as sf
import shutil

from sqlalchemy.orm import Session
from pyannote.audio import Pipeline

from editor_app import cfg, crud, schemas, html_menu
from editor_app.database import SessionLocal
from editor_app.models import User, CrossProject
from editor_app.stt import stt_model
from utils import gradio_read_audio_data, not_raw_speaker_re
from media_utils import download_youtube_media, extract_and_resample_audio_ffmpeg, extract_video_id, get_youtube_embed_code

diarization_model = Pipeline.from_pretrained(cfg.diarization.model_name, use_auth_token=cfg.diarization.auth_token)

DEMO_DURATION = 120  # duration of demo in seconds
AD_OFFSET = 120  # approx ad offset


def detect_speakers(input_media, media_link, project_name, user_email, options: list):

    demo_run, save_speakers = False, False
    if 'Demo Run' in options:
        demo_run = True

    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)

    cross_project: CrossProject = crud.get_cross_project_by_title(db, project_name, user.id)
    if cross_project is not None:
        raise Exception(f"CrossProject {project_name} already exists, pick another name")

    if media_link is not None:
        tmp_media_path = download_youtube_media(media_link, tempfile.gettempdir())
        name = tmp_media_path.name
    elif input_media is not None:
        tmp_media_path = input_media.name
        name = tmp_media_path.split('/')[-1]

    else:
        raise Exception(f"either media_link or media file should be provided")

    # TODO refactor, I need to have media path befor I create crosslingual db entry.
    # TODO Add more check and atomicity. I don't want to manually cleanup disk and database if something went wrong while project creation.

    cross_project_data = schemas.CrossProjectCreate(title=project_name, media_name=name)
    cross_project = crud.create_cross_project(db, cross_project_data, user.id)
    media_path = shutil.copy(tmp_media_path, cross_project.get_media_path())
    raw_wav_path = cross_project.get_raw_wav_path(sample_rate=cfg.stt.sample_rate)

    res0 = extract_and_resample_audio_ffmpeg(media_path, raw_wav_path, cfg.stt.sample_rate)
    res1 = extract_and_resample_audio_ffmpeg(media_path, cross_project.get_raw_wav_path(sample_rate=22050), 22050)

    waveform, sample_rate = gradio_read_audio_data(raw_wav_path)
    if demo_run:
        waveform = waveform[:(10*60*cfg.stt.sample_rate)]

    diarization = diarization_model({'waveform': waveform.unsqueeze(0), 'sample_rate': sample_rate})

    speakers = set(diarization.labels())
    speaker_samples = dict()

    for seg, lbl, spkr in diarization.itertracks(yield_label=True):
        if spkr not in speaker_samples:
            speaker_samples[spkr] = seg
        if len(speakers) == len(speaker_samples):
            break
    res = ''
    for spkr in sorted(speakers):
        row = f'{spkr}: {speaker_samples[spkr]}\n'
        res += row

    results = [res]
    if media_link is not None:
        youtube_id = extract_video_id(media_link)
        iframe_val = get_youtube_embed_code(youtube_id)
        results.append(gr.HTML.update(value=iframe_val))
    else:
        # https://www.youtube.com/watch?v=rgOylRHp1gM&ab_channel=Fluppy
        iframe_val = get_youtube_embed_code("rgOylRHp1gM")
        results.append(gr.HTML.update(value=iframe_val))

    return results


def transcribe(project_name, language: str, named_speakers: str, user_email: str, options: list):
    """Transcribe input media with speaker diarization, resulting transcript will be in form:
    [ HH:MM:SS.sss --> HH:MM:SS.sss ]
    {SPEAKER}
    Transcribed text
    """
    demo_run, save_speakers = False, False
    if 'Demo Run' in options:
        demo_run = True
    if 'Save speakers' in options:
        save_speakers = True

    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)

    cross_project: CrossProject = crud.get_cross_project_by_title(db, project_name, user.id)
    if cross_project is None:
        raise Exception(f"CrossProject {project_name} doesn't exists")

    if len(language) == 0:
        language = None

    waveform, sample_rate = gradio_read_audio_data(cross_project.get_raw_wav_path(sample_rate=cfg.stt.sample_rate))
    waveform_22k, _ = gradio_read_audio_data(cross_project.get_raw_wav_path(sample_rate=22050))
    if demo_run:
        waveform = waveform[:(10*60*cfg.stt.sample_rate)]
        waveform_22k = waveform_22k[:(10*60*22050)]
    diarization = diarization_model({'waveform': waveform.unsqueeze(0), 'sample_rate': sample_rate})

    if named_speakers:
        speakers = []
        for speaker in named_speakers.lower().split(','):
            spkr = not_raw_speaker_re.sub('', speaker)
            speakers.append(spkr)

        assert len(diarization.labels()) == len(speakers)
        diarization = diarization.rename_labels(generator=iter(speakers), copy=False)

        if save_speakers:
            db_spkrs = []
            for spkr in speakers:
                db_spkr = crud.get_speaker_by_name(db, spkr, cross_project.id)
                if not db_spkr:
                    db_spkrs.append(crud.create_speaker(db, spkr, cross_project.id))
                else:
                    db_spkrs.append(db_spkr)

    char_count = 0
    res = ''
    segments = []
    ffmpeg_str = ''
    demo_duration = 0
    for idx, (seg, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):

        if demo_run:
            if seg.start < AD_OFFSET:
                continue

        seg_wav = waveform[int(seg.start * cfg.stt.sample_rate): int(seg.end * cfg.stt.sample_rate)]
        seg_wav_22k = waveform_22k[int(seg.start * 22050): int(seg.end * 22050)]
        seg_res = stt_model.transcribe(seg_wav, language=language)
        text = seg_res['text']
        lang = seg_res['language']
        segments.append([seg, speaker, text, lang])
        res += f"{seg}\n{{{speaker}}}\n{text}\n\n"
        char_count += len(seg_res['text'])
        seg_name = f'output_{idx:03}.wav'

        # seg_path = cross_project.get_data_root().joinpath(seg_name)
        # sf.write(seg_path, seg_wav, cfg.stt.sample_rate)

        if named_speakers and save_speakers:
            db_spkr = crud.get_speaker_by_name(db, speaker, user.id)
            seg_path = db_spkr.get_speaker_data_root().joinpath(seg_name)
            sf.write(seg_path, seg_wav_22k, 22050)

        if demo_run:
            demo_duration += (seg.end - seg.start)
            if demo_duration >= DEMO_DURATION:
                break

        ffmpeg_str += f' -ss {seg.start} -to {seg.end} -c copy {seg_name}'
    # quick and dirty way to cut audio on pieces with ffmpeg.
    print(ffmpeg_str)
    detected_lang = segments[0][-1]

    return res, detected_lang, char_count


def save_transcript(project_name, text, lang, user_email):
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)

    cross_project: CrossProject = crud.get_cross_project_by_title(db, project_name, user.id)
    if cross_project is None:
        raise Exception(f"CrossProject {project_name} doesn't exists")

    if len(cross_project.transcript) > 0:
        print('Transcript already saved!!!')
        return gr.Image.update(visible=True)

    transcript_data = schemas.TranscriptCreate(text=text, lang=lang)
    transcript_db = crud.create_transcript(db, transcript_data, cross_project.id)

    with open(transcript_db.get_path(), 'w') as f:
        f.write(text)

    return gr.Image.update(visible=True)


with gr.Blocks() as transcriber:
    with gr.Row() as row0:
        with gr.Column(scale=1) as col0:
            menu = gr.HTML(value=html_menu)
            email = gr.Text(label='user', placeholder='Enter user email', value=cfg.user.email)
            project_name = gr.Text(label='Project name', placeholder="enter your project name")
            media_link = gr.Text(label='link', placeholder='Link to youtube video, or any audio file')
            iframe = gr.HTML(label='youtube video')
            file = gr.File(label='input media')
            detect_spkr_button = gr.Button(value='Detect speakers')
            detected_speakers = gr.Text(label='Ordinal speakers')
            named_speakers = gr.Text(label='Named speakers')
            input_lang = gr.Text(label='input language')

        with gr.Column(scale=1) as col1:
            text = gr.Text(label='Text transcription', interactive=True)
            detected_lang = gr.Text(label='Detected language')
            num_chars = gr.Number(label='Number of characters')
            transcribe_button = gr.Button(value='Transcribe!')
            options = gr.CheckboxGroup(choices=['Demo Run', 'Save speakers'], value=['Demo Run', 'Save speakers'])
            save_transcript_button = gr.Button(value='save')
            BASENJI_PIC = 'https://www.akc.org/wp-content/uploads/2017/11/Basenji-On-White-01.jpg'
            success_image = gr.Image(value=BASENJI_PIC, visible=False)

        detect_spkr_button.click(detect_speakers,
                                 inputs=[file, media_link, project_name, email, options],
                                 outputs=[detected_speakers, iframe])
        transcribe_button.click(
            transcribe,
            inputs=[project_name, input_lang, named_speakers, email, options],
            outputs=[text, detected_lang, num_chars])
        save_transcript_button.click(
            save_transcript,
            inputs=[project_name, text, detected_lang, email],
            outputs=[success_image])

if __name__ == '__main__':
    transcriber.launch(debug=True)
