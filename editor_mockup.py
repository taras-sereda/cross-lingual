import os.path
import warnings

import gradio as gr
import librosa
from tortoise.utils.text import split_and_recombine_text

MAX_UTTERANCE = 20


def dummy_read(text, audio_tuple):
    sample_rate, audio = audio_tuple

    texts = split_and_recombine_text(text)
    res = []
    for text in texts:
        gen, sample_rate = librosa.load(librosa.example('brahms'))
        res.append(gr.Textbox.update(value=text, visible=True))
        res.append(gr.Audio.update(value=(sample_rate, gen), visible=True))
        res.append(gr.Button.update(visible=True))
    n_utterance = len(texts)

    # padding
    if (delta := MAX_UTTERANCE - n_utterance) > 0:
        for _ in range(delta):
            res.append(gr.Textbox.update(visible=False))
            res.append(gr.Audio.update(visible=False))
            res.append(gr.Button.update(visible=False))
    # clipping
    else:
        warnings.warn("Output is clipped!")
        pass
    print('delta', delta)
    return [len(texts)] + res


def dummy_reread(text):
    gen, sample_rate = librosa.load(librosa.example('pistachio'))
    return sample_rate, gen


with gr.Blocks() as block:
    with gr.Row() as row0:
        with gr.Column() as col0:
            text = gr.Text(label='Text for synthesis')
            reference_audio = gr.Audio(label='reference audio')
            button = gr.Button(value='Go!')
            outputs = [gr.Number(label='number of utterances')]

        with gr.Column(variant='compact') as col1:
            for i in range(MAX_UTTERANCE):
                utterance = gr.Textbox(label=f'utterance_{i}', visible=False)
                audio = gr.Audio(label=f'audio_{i}', visible=False)

                try_again = gr.Button(value='try again', visible=False)
                try_again.click(fn=dummy_reread, inputs=[utterance], outputs=[audio])

                outputs.extend([utterance, audio, try_again])

        button.click(fn=dummy_read, inputs=[text, reference_audio], outputs=outputs)

    example_text = """
    Everything was perfectly swell.
    
    There were no prisons, no slums, no insane asylums, no cripples, no
    poverty, no wars.
    
    All diseases were conquered. So was old age.
    
    Death, barring accidents, was an adventure for volunteers.
    
    The population of the United States was stabilized at forty-million
    souls.
    
    One bright morning in the Chicago Lying-in Hospital, a man named Edward
    K. Wehling, Jr., waited for his wife to give birth. He was the only man
    waiting. Not many people were born a day any more.
    
    Wehling was fifty-six, a mere stripling in a population whose average
    age was one hundred and twenty-nine.
    
    X-rays had revealed that his wife was going to have triplets. The
    children would be his first.
    
    """
    gr.Markdown("Text examples")
    gr.Examples([example_text], [text])
    gr.Markdown("Audio examples")
    gr.Examples([os.path.join(os.path.dirname(__file__), 'data/VLND2ptAOio.clip.24000.wav')], [reference_audio])

block.launch()
