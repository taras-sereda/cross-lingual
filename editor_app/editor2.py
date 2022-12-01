
import gradio as gr

from . import cfg, example_text, example_voice_sample_path
from .tts_common import add_speaker, get_speakers, get_projects, read, playground_read, reread, load, combine

with gr.Blocks() as submitter:
    with gr.Row() as row0:
        with gr.Column(scale=1) as col0:
            email = gr.Text(label='user', placeholder='Enter user email', value=cfg.user.email)

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
            title = gr.Text(label='Title', placeholder="enter project title")
            text = gr.Text(label='Text')
            button = gr.Button(value='Go!', variant='primary')

            with gr.Box() as b:
                playground_header = gr.Markdown(value="Playground")
                playground_text = gr.Text(label='text')
                playground_spkr = gr.Text(label='speaker')
                playground_audio = gr.Audio(label='audio')
                playground_button = gr.Button(value='try again')
                playground_button.click(fn=playground_read,
                                        inputs=[playground_text, playground_spkr, email],
                                        outputs=[playground_audio, playground_spkr])

        button.click(fn=read, inputs=[title, text, email], outputs=[])

    gr.Markdown("Text examples")
    gr.Examples([example_text], [text])
    gr.Markdown("Audio examples")
    gr.Examples([example_voice_sample_path], [reference_audio])


with gr.Blocks() as editor:
    with gr.Row() as editor_row0:
        with gr.Column(scale=1) as editor_col0:
            email = gr.Text(label='user', placeholder='Enter user email', value=cfg.user.email)
            speakers = gr.Textbox(label='speakers')
            get_speakers_button = gr.Button('Get speakers')
            user_projects = gr.Textbox(label='user projects')
            get_user_projects_button = gr.Button('Get projects')
            get_speakers_button.click(get_speakers, inputs=[email], outputs=[speakers])
            get_user_projects_button.click(get_projects, inputs=[email], outputs=[user_projects])

            title = gr.Text(label='Title', placeholder="enter project title")
            utter_from_idx = gr.Number(value=0, precision=0)
            button_load = gr.Button(value='Load', variant='primary')
            text = gr.Text(label='Text')

            button_combine = gr.Button(value='Combine')
            combined_audio = gr.Audio(visible=False)

        outputs = [text]
        with gr.Column(scale=1, variant='compact') as editor_col1:
            for i in range(cfg.editor.max_utterance):
                utterance = gr.Textbox(label=f'utterance_{i}', visible=False, show_label=False)
                utterance_idx = gr.Number(visible=False, precision=0)
                utter_speaker = gr.Textbox(label=f'speaker name', visible=False)
                audio = gr.Audio(label=f'audio_{i}', visible=False, show_label=False)

                try_again = gr.Button(value='try again', visible=False)
                try_again.click(fn=reread,
                                inputs=[title, utterance, utterance_idx, utter_speaker, email],
                                outputs=[audio, utter_speaker])

                outputs.extend([utterance, utterance_idx, utter_speaker, audio, try_again])

        button_load.click(fn=load, inputs=[title, email, utter_from_idx], outputs=outputs)
        button_combine.click(fn=combine, inputs=[title, email], outputs=[combined_audio])


if __name__ == '__main__':
    submitter.launch(debug=True)

# TODO integrate whisper for judging of synthesis quality.
