import gradio as gr

from editor_app.stt import transcribe, save_transcript, load_transcript
from editor_app.common import get_transcripts

with gr.Blocks() as transcriber:
    with gr.Row() as row0:
        with gr.Column(scale=1, variant='panel') as col0:
            user_projects = gr.Dataframe(label='user most recent transcripts')
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
            with gr.Row():
                load_button = gr.Button(value='Load')
                transcribe_button = gr.Button(value='Transcribe!')

        with gr.Column(scale=1) as col1:
            speakers = gr.Text(label='speakers')
            text = gr.Text(label='Text transcription', interactive=True)
            save_transcript_button = gr.Button(value='save')
            iframe = gr.HTML(label='youtube video')
            audio = gr.Audio(visible=False)
            video = gr.Video(visible=False)

        transcribe_button.click(
            transcribe,
            inputs=[file, media_link, project_name, lang, options],
            outputs=[text, lang, speakers, iframe, audio, video])
        load_button.click(
            load_transcript,
            inputs=[project_name],
            outputs=[project_name, text])
        save_transcript_button.click(
            save_transcript,
            inputs=[project_name, text, lang],
            outputs=[project_name])

    transcriber.load(get_transcripts, outputs=[user_projects])

if __name__ == '__main__':
    transcriber.launch(debug=True)
