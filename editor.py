import os
import warnings

import gradio as gr
from tortoise.api import TextToSpeech
from tortoise.utils.text import split_and_recombine_text

tts = TextToSpeech()
seed = 42
candidates = 1
model_sample_rate = 24_000

MAX_UTTERANCE = 20


def read(text, audio_tuple):
    sample_rate, audio = audio_tuple

    texts = split_and_recombine_text(text)
    res = []
    for text in texts:
        gen = tts.tts_with_preset(text, voice_samples=None, conditioning_latents=None,
                                  preset='ultra_fast', k=candidates, use_deterministic_seed=seed)
        res.append(gr.Textbox.update(value=text, visible=True))
        res.append(gr.Audio.update(value=(model_sample_rate, gen.cpu().numpy()), visible=True))
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


def reread(text):
    gen = tts.tts_with_preset(text, voice_samples=None, conditioning_latents=None,
                              preset='ultra_fast', k=candidates, use_deterministic_seed=seed)
    return model_sample_rate, gen.cpu().numpy()


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
                try_again.click(fn=reread, inputs=[utterance], outputs=[audio])

                outputs.extend([utterance, audio, try_again])

        button.click(fn=read, inputs=[text, reference_audio], outputs=outputs)

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
