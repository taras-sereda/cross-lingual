import gradio as gr

from editor_app import cfg
from editor_app.tts import read, reread, load, combine, get_cross_projects, load_translation

with gr.Blocks() as submitter:
    with gr.Row() as row0:
        with gr.Column(scale=1, variant='panel') as col0:
            user_projects = gr.Dataframe(label='user projects')
            speakers = gr.Dataframe(label='speakers', col_count=1)
            with gr.Row():
                title = gr.Text(label='Title', placeholder="enter project title")
                lang = gr.Text(label='Lang', value="EN-US")
            load_translation_button = gr.Button('Load')

        with gr.Column(scale=1) as col1:
            text = gr.Text(label='Text')
            button = gr.Button(value='Go!', variant='primary')

        load_translation_button.click(load_translation, inputs=[title, lang], outputs=[text, speakers])
        button.click(fn=read, inputs=[title, lang, text])

    submitter.load(get_cross_projects, outputs=[user_projects])


with gr.Blocks() as editor:
    with gr.Row() as editor_row0:
        with gr.Column(scale=1, variant='panel') as editor_col0:
            user_projects = gr.Dataframe(label='user projects')
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
            video = gr.Video(visible=False)
            audio = gr.Audio(visible=False)

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
                                inputs=[title, lang, utter_text, utter_idx, utter_speaker],
                                outputs=[utter_audio, utter_speaker, utter_score])

                outputs.extend([utter_text, utter_idx, utter_speaker, utter_score, utter_audio, try_again])

        button_load.click(fn=load, inputs=[title, lang, utter_from_idx, score_slider], outputs=outputs)
        button_combine.click(fn=combine, inputs=[title, lang], outputs=[video, audio])

    editor.load(get_cross_projects, outputs=[user_projects])


if __name__ == '__main__':
    # submitter.launch(debug=True)
    editor.launch(debug=True)
