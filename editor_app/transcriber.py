import gradio as gr

import whisper

from . import example_voice_sample_path

stt_model = whisper.load_model('small')


def transcribe(audio_data):
    res = stt_model.transcribe(audio_data)

    return res['text'], res['language']


with gr.Blocks() as transcriber:
    with gr.Row() as row0:
        with gr.Column(scale=1) as col0:
            project_name = gr.Text(label='Project name', placeholder="enter your project name")
            audio = gr.Audio(label='Audio for transcription', type='filepath')

        with gr.Column(scale=1) as col1:
            text = gr.Text(label='Text transcription', interactive=True)
            lang = gr.Text(label='Detected language')
            button = gr.Button(value='Go!')
        button.click(transcribe, inputs=[audio], outputs=[text, lang])

    gr.Markdown("Audio examples")
    gr.Examples([example_voice_sample_path], [audio])
