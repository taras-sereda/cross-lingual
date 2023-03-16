import gradio as gr

from editor_app import cfg
from editor_app.tts import playground_read, add_speaker

with gr.Blocks() as playground:
    with gr.Row() as row0:
        with gr.Column(scale=1, variant='panel') as col0:
            user_projects = gr.Dataframe(label='user projects')
            reference_audio = gr.Files(label='reference audio', file_types=['audio'])
            speaker_name = gr.Textbox(label='Speaker name', placeholder="Enter speaker name, allowed symbols: "
                                                                        "lower case letters, numbers, and _")
            add_speaker_button = gr.Button('Add speaker')
            errors = gr.Textbox(label='error messages')
            add_speaker_button.click(add_speaker, inputs=[reference_audio, speaker_name], outputs=[errors])

        with gr.Column(scale=1) as col1:
            with gr.Box() as b:
                playground_header = gr.Markdown(value="Playground")
                playground_spkr = gr.Text(label='speaker')
                playground_text = gr.Text(label='text')
                playground_button = gr.Button(value='try again')

                playground_outputs = []
                for i_audio in range(cfg.tts.playground.candidates):
                    with gr.Row():
                        elems = [gr.Text(label=f'STT text {i_audio}'),
                                 gr.Number(label=f'similarity score {i_audio}', precision=3),
                                 gr.Audio(label=f'audio {i_audio}')
                                 ]
                        playground_outputs.extend(elems)

            playground_button.click(fn=playground_read,
                                    inputs=[playground_text, playground_spkr],
                                    outputs=playground_outputs)

if __name__ == '__main__':
    playground.launch(debug=True)
