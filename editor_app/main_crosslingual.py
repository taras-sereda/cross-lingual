import gradio as gr

from transcriber import transcriber
from translator import translator
from editor2 import submitter, editor

with gr.Blocks() as cross_lingual:
    with gr.Tab("transcribe") as transcribe:
        transcriber.render()
    with gr.Tab("translate") as translate:
        translator.render()
    with gr.Tab("submit") as submit:
        submitter.render()
    with gr.Tab("edit") as edit:
        editor.render()
user_db = [
    ("taras.y.sereda@proton.me", "qwerty"),
    ("bohdan@crosslingual.io", "qwerty")
]
if __name__ == '__main__':
    cross_lingual.launch(auth=user_db, server_name="0.0.0.0", server_port=8000)
