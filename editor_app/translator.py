import gradio as gr

import deepl

from utils import convert_text_to_segments
from . import cfg

deepl_translator = deepl.Translator(cfg.translation.auth_token)


def translate(text: str, tgt_lang: str):
    """Translate
    """
    segments = convert_text_to_segments(text)
    num_src_char = 0
    num_tgt_char = 0

    result = ''
    with open('uklon_en_translation.txt', 'w') as fd:
        for seg in segments:
            # TODO. speaker mapping - should be moved to transcriber
            if seg['speaker'] == '[SPEAKER_00]':
                spkr = 'sergiy_smus_before_and_after_uklon'
            elif seg['speaker'] == '[SPEAKER_01]':
                spkr = 'denys_marakin_before_and_after_intro'
            else:
                raise ValueError('Unknown speaker')

            seg_res = deepl_translator.translate_text(seg['text'], target_lang=tgt_lang)
            num_src_char += len(seg['text'])
            num_tgt_char += len(seg_res.text)

            components = []
            if seg['timecode'] is not None:
                components.append(str(seg['timecode']))
            components.extend([f'{{{spkr}}}', seg_res.text, '\n'])
            line = '\n'.join(components)
            fd.write(line)
            result += line

    return result, num_src_char, num_tgt_char


with gr.Blocks() as translator:
    with gr.Row() as row0:
        with gr.Column(scale=1) as col0:
            project_name = gr.Text(label='Project name', placeholder="enter your project name")
            src_text = gr.Text(label='Text transcription', interactive=True)
            tgt_lang = gr.Text(label='Target Language', value='EN-US')
        with gr.Column(scale=1) as col1:
            tgt_text = gr.Text(label='Text translation', interactive=True)
            num_src_chars = gr.Number(label='Number of characters')
            num_tgt_chars = gr.Number(label='Number of characters')
            button = gr.Button(value='Go!')
        button.click(translate, inputs=[src_text, tgt_lang], outputs=[tgt_text, num_src_chars, num_tgt_chars])
