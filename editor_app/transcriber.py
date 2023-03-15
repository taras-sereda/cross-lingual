import gradio as gr

from editor_app import BASENJI_PIC
from editor_app.stt import transcribe, save_transcript
from editor_app.tts import get_cross_projects

with gr.Blocks() as transcriber:
    with gr.Row() as row0:
        with gr.Column(scale=1, variant='panel') as col0:
            user_projects = gr.Dataframe(label='user projects')
            project_name = gr.Text(label='Project name', placeholder="enter your project name, can't be empty.")
            media_link = gr.Text(label='Youtube Link', placeholder='Link to youtube video.')
            file = gr.File(label='input media')
            with gr.Row():
                lang = gr.Text(label='language', placeholder="Provide medial language, if it's known, "
                                                             "otherwise language will be detected. "
                                                             "Example: Uk, En, De.")
                options = gr.CheckboxGroup(label='transcription options',
                                           choices=['Demo Run', 'Save speakers'],
                                           value=['Demo Run', 'Save speakers'])
            transcribe_button = gr.Button(value='Transcribe!')

        with gr.Column(scale=1) as col1:
            speakers = gr.Text(label='speakers')
            text = gr.Text(label='Text transcription', interactive=True)
            save_transcript_button = gr.Button(value='save')
            iframe = gr.HTML(label='youtube video')
            audio = gr.Audio(visible=False)
            video = gr.Video(visible=False)
            success_image = gr.Image(value=BASENJI_PIC, visible=False)

        transcribe_button.click(
            transcribe,
            inputs=[file, media_link, project_name, lang, options],
            outputs=[text, lang, speakers, iframe, audio, video])
        save_transcript_button.click(
            save_transcript,
            inputs=[project_name, text, lang],
            outputs=[success_image])

    transcriber.load(get_cross_projects, inputs=[], outputs=[user_projects])

if __name__ == '__main__':
    transcriber.launch(debug=True)
