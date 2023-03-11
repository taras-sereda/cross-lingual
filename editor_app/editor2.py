import gradio as gr

from editor_app import cfg, html_menu
from editor_app.tts import add_speaker, read, playground_read, reread, load, combine, get_cross_projects, load_translation

with gr.Blocks() as submitter:
    with gr.Row() as row0:
        with gr.Column(scale=1) as col0:
            menu = gr.HTML(html_menu)
            email = gr.Text(label='user', placeholder='Enter user email', value=cfg.user.email)
            user_projects = gr.Dataframe(label='user projects')

            reference_audio = gr.Files(label='reference audio', file_types=['audio'])
            speaker_name = gr.Textbox(label='Speaker name', placeholder="Enter speaker name, allowed symbols: "
                                                                        "lower case letters, numbers, and _")
            add_speaker_button = gr.Button('Add speaker')
            errors = gr.Textbox(label='error messages')
            speakers = gr.Dataframe(label='speakers')

            add_speaker_button.click(add_speaker, inputs=[reference_audio, speaker_name, email], outputs=[errors])

        with gr.Column(scale=1) as col1:
            title = gr.Text(label='Title', placeholder="enter project title")
            lang = gr.Text(label='Lang', value="EN-US")
            text = gr.Text(label='Text')
            button = gr.Button(value='Go!', variant='primary')
            load_translation_button = gr.Button('Load')

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
                                    inputs=[playground_text, playground_spkr, email],
                                    outputs=playground_outputs)

        button.click(fn=read, inputs=[title, lang, text, email], outputs=[])
        load_translation_button.click(load_translation, inputs=[title, lang, email], outputs=[text, speakers])

    submitter.load(get_cross_projects, inputs=[email], outputs=[user_projects])


with gr.Blocks() as editor:
    with gr.Row() as editor_row0:
        with gr.Column(scale=1) as editor_col0:
            menu = gr.HTML(html_menu)
            email = gr.Text(label='user', placeholder='Enter user email', value=cfg.user.email)
            user_projects = gr.Dataframe(label='projects')

            speakers = gr.Dataframe(label='speakers', col_count=1)

            with gr.Row() as proj_row:
                title = gr.Text(label='Title', placeholder="enter project title")
                lang = gr.Text(label='Lang', value="EN-US")

            utter_from_idx = gr.Number(label='utterance start index', value=0, precision=0)
            score_slider = gr.Slider(0, 1.0, label='min utterance score')
            button_load = gr.Button(value='Load', variant='primary')
            with gr.Row():
                num_utterance = gr.Number(label='Total number of utterances', precision=0)
                avg_prj_score = gr.Number(label='Average project score', precision=3)
            text = gr.Text(label='Text')

            button_combine = gr.Button(value='Combine')
            combined_audio = gr.Audio(visible=False)

        outputs = [speakers, text, num_utterance, avg_prj_score]
        with gr.Column(scale=1, variant='compact') as editor_col1:
            for i in range(cfg.editor.max_utterance):

                utter_idx = gr.Number(visible=False, precision=0)
                with gr.Row():
                    utter_speaker = gr.Textbox(label='speaker name', visible=False)
                    utter_score = gr.Number(label='score', visible=False)

                utter_text = gr.Textbox(label=f'utterance_{i}', visible=False, show_label=False)
                utter_audio = gr.Audio(label=f'audio_{i}', visible=False, show_label=False)

                try_again = gr.Button(value='try again', visible=False)
                try_again.click(fn=reread,
                                inputs=[title, lang, utter_text, utter_idx, utter_speaker, email],
                                outputs=[utter_audio, utter_speaker, utter_score])

                outputs.extend([utter_text, utter_idx, utter_speaker, utter_score, utter_audio, try_again])

        button_load.click(fn=load, inputs=[title, lang, email, utter_from_idx, score_slider], outputs=outputs)
        button_combine.click(fn=combine, inputs=[title, lang, email], outputs=[combined_audio])

    editor.load(get_cross_projects, inputs=[email], outputs=[user_projects])


if __name__ == '__main__':
    # submitter.launch(debug=True)
    editor.launch(debug=True)
