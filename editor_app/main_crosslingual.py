import gradio as gr

from transcriber import transcriber
from translator import translator
from editor2 import submitter, editor
from end2end import e2e

with gr.Blocks() as cross_lingual:
    with gr.Tab("end2end") as e2e_tab:
        e2e.render()
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
auth_msg = "Welcome to CrossLingual"
if __name__ == '__main__':
    cross_lingual.launch(auth=user_db, auth_message=auth_msg, server_name="0.0.0.0",
                         server_port=8000, show_error=True, debug=True)
