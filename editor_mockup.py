import gradio as gr
import librosa
from tortoise.utils.text import split_and_recombine_text


def dummy_read(text):
    texts = split_and_recombine_text(text)
    res = []
    for text in texts:
        gen, sample_rate = librosa.load(librosa.example('brahms'))
        res.append(text)
        res.append((sample_rate, gen))
    return [len(texts)] + res


def dummy_reread(text):
    gen, sample_rate = librosa.load(librosa.example('pistachio'))
    return sample_rate, gen


with gr.Blocks() as block:
    with gr.Row() as row0:
        with gr.Column() as col0:
            text = gr.Text(label='Text for synthesis')
            button = gr.Button(value='Go!')
            outputs = [gr.Number(label='number of utterances')]

        with gr.Column() as col1:
            for i in range(1):
                utterance = gr.Text(label=f'utterance_{i}')
                audio = gr.Audio(label=f'audio_{i}')

                try_again = gr.Button(value='try again')
                try_again.click(fn=dummy_reread, inputs=[utterance], outputs=[audio])

                outputs.extend([utterance, audio])

        button.click(fn=dummy_read, inputs=[text], outputs=outputs)

block.launch()
