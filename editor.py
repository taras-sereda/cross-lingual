from time import sleep

import gradio as gr
import librosa
from tortoise.api import TextToSpeech
from tortoise.utils.text import split_and_recombine_text

tts = TextToSpeech()
seed = 42
candidates = 1


def read(text):
    sample_rate = 24_000
    texts = split_and_recombine_text(text)
    for text in texts:
        gen = tts.tts_with_preset(text, voice_samples=None, conditioning_latents=None,
                                  preset='ultra_fast', k=1, use_deterministic_seed=seed)
        if candidates == 1:
            gen = gen.squeeze(0).cpu()
            # add saving of synthesized example on host's disk.
            # torchaudio.save(os.path.join(voice_outpath, f'{j}.wav'), gen, sample_rate)

        yield text, (sample_rate, gen.numpy())


demo = gr.Interface(fn=read,
                    inputs=[gr.Textbox(value="add your text here"), gr.State()],
                    outputs=['text', 'audio', gr.State()],
                    allow_flagging="never")

demo.queue()
demo.launch(debug=True)
