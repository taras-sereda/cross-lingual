import subprocess

import gradio as gr
import soundfile as sf
import shutil

from sqlalchemy.orm import Session
from pyannote.audio import Pipeline

from editor_app import cfg, crud, schemas
from editor_app.database import SessionLocal
from editor_app.models import User, CrossProject
from editor_app.stt import stt_model
from utils import gradio_read_audio_data, not_raw_speaker_re

diarization_model = Pipeline.from_pretrained(cfg.diarization.model_name, use_auth_token=cfg.diarization.auth_token)

DEMO_DURATION = 120  # duration of demo in seconds

def detect_speakers(input_media, project_name, user_email):
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    if user is None:
        raise Exception(f"User {user_email} not found. Provide valid email")

    cross_project: CrossProject = crud.get_cross_project_by_title(db, project_name, user.id)
    if cross_project is not None:
        raise Exception(f"CrossProject {project_name} already exists, pick another name")

    name = input_media.name.split('/')[-1]

    cross_project_data = schemas.CrossProjectCreate(title=project_name, media_name=name)
    cross_project = crud.create_cross_project(db, cross_project_data, user.id)
    media_path = shutil.copy(input_media.name, cross_project.get_media_path())
    raw_wav_path = cross_project.get_raw_wav_path()
    # ffmpeg
    ffmpeg_path = shutil.which('ffmpeg')
    # command = f"{ffmpeg_path} -i {media_path} -ac 1 -ar 16000 {raw_media_path}"
    # print(os.environ.get('PATH'))
    res = subprocess.run([
        f"{ffmpeg_path}",
        "-hide_banner",
        "-loglevel", "error",
        "-i", f"{media_path}",
        "-ac", "1",
        "-ar", f"{cfg.stt.sample_rate}",
        # "-acodec", f"pcm_s16le",
        f"{raw_wav_path}"],
        check=True,
        # stdout=subprocess.DEVNULL,
    )
    # print(res)
    waveform, sample_rate = gradio_read_audio_data(raw_wav_path)

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
    return res


def transcribe(project_name, language: str, named_speakers: str, user_email: str, demo_run: bool = True):
    """Transcribe input media with speaker diarization, resulting transcript will be in form:
    [ HH:MM:SS.sss --> HH:MM:SS.sss ]
    {SPEAKER}
    Transcribed text
    """
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    if user is None:
        raise Exception(f"User {user_email} not found. Provide valid email")

    cross_project: CrossProject = crud.get_cross_project_by_title(db, project_name, user.id)
    if cross_project is None:
        raise Exception(f"CrossProject {project_name} doesn't exists")

    if len(language) == 0:
        language = None

    waveform, sample_rate = gradio_read_audio_data(cross_project.get_raw_wav_path())
    diarization = diarization_model({'waveform': waveform.unsqueeze(0), 'sample_rate': sample_rate})

    if named_speakers:
        speakers = []
        for speaker in named_speakers.lower().split(','):
            spkr = not_raw_speaker_re.sub('', speaker)
            speakers.append(spkr)

        assert len(diarization.labels()) == len(speakers)
        diarization = diarization.rename_labels(generator=iter(speakers), copy=False)

    char_count = 0
    res = ''
    segments = []
    ffmpeg_str = ''

    for idx, (seg, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):

        if demo_run and seg.start > DEMO_DURATION:
            break

        seg_wav = waveform[int(seg.start * cfg.stt.sample_rate): int(seg.end * cfg.stt.sample_rate)]
        seg_res = stt_model.transcribe(seg_wav, language=language)
        text = seg_res['text']
        lang = seg_res['language']
        segments.append([seg, speaker, text, lang])
        res += f"{seg}\n{{{speaker}}}\n{text}\n\n"
        char_count += len(seg_res['text'])
        seg_name = f'output_{idx:03}.wav'

        seg_path = cross_project.get_data_root().joinpath(seg_name)
        sf.write(seg_path, seg_wav, cfg.stt.sample_rate)

        ffmpeg_str += f' -ss {seg.start} -to {seg.end} -c copy {seg_name}'
    # quick and dirty way to cut audio on pieces with ffmpeg.
    print(ffmpeg_str)
    detected_lang = segments[0][-1]

    return res, detected_lang, char_count


def save_transcript(project_name, text, lang, user_email):
    db: Session = SessionLocal()
    user: User = crud.get_user_by_email(db, user_email)
    if user is None:
        raise Exception(f"User {user_email} not found. Provide valid email")

    cross_project: CrossProject = crud.get_cross_project_by_title(db, project_name, user.id)
    if cross_project is None:
        raise Exception(f"CrossProject {project_name} doesn't exists")

    if len(cross_project.transcript) > 0:
        print('Transcript already saved!!!')
        return gr.Image.update(visible=True)

    transcript_data = schemas.TranscriptCreate(text=text, lang=lang)
    transcript_db = crud.create_transcript(db, transcript_data, user.id, cross_project.id)

    with open(transcript_db.get_path(), 'w') as f:
        f.write(text)

    return gr.Image.update(visible=True)


with gr.Blocks() as transcriber:
    with gr.Row() as row0:
        with gr.Column(scale=1) as col0:
            email = gr.Text(label='user', placeholder='Enter user email', value=cfg.user.email)
            project_name = gr.Text(label='Project name', placeholder="enter your project name")
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
            demo_checkbox = gr.Checkbox(val=True, label='Demo Run')
            save_transcript_button = gr.Button(value='save')
            BASENJI_PIC = 'https://www.akc.org/wp-content/uploads/2017/11/Basenji-On-White-01.jpg'
            success_image = gr.Image(value=BASENJI_PIC, visible=False)

        detect_spkr_button.click(detect_speakers, inputs=[file, project_name, email], outputs=[detected_speakers])
        transcribe_button.click(
            transcribe,
            inputs=[project_name, input_lang, named_speakers, email, demo_checkbox],
            outputs=[text, detected_lang, num_chars])
        save_transcript_button.click(
            save_transcript,
            inputs=[project_name, text, detected_lang, email],
            outputs=[success_image])

if __name__ == '__main__':
    transcriber.launch(debug=True)
