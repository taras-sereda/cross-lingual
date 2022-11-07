import os
import warnings

import gradio as gr
import soundfile as sf
import torch

from tortoise.api import TextToSpeech
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.audio import load_voices

from . import cfg, example_voice_sample_path, example_text

tts = TextToSpeech()

voices_dir = os.path.join(os.path.dirname(__file__), '../user_data/voices')

if torch.cuda.is_available():
    preset = 'standard'
else:
    preset = 'ultra_fast'


def read(text, audio_tuple, speaker_name, speaker_vector):
    sample_rate, audio = audio_tuple

    # write audio file
    speaker_dir = os.path.join(voices_dir, speaker_name)
    os.makedirs(speaker_dir, exist_ok=True)
    audio_file_name = os.path.join(speaker_dir, '0.wav')
    sf.write(audio_file_name, audio, sample_rate)

    # read audio file
    voice_samples, conditioning_latents = load_voices([speaker_name], [voices_dir])
    speaker_vector['voice_samples'] = voice_samples
    speaker_vector['conditioning_latents'] = conditioning_latents

    texts = split_and_recombine_text(text)
    res = []
    for text in texts:
        gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                  preset=preset, k=cfg.tts.candidates, use_deterministic_seed=cfg.tts.seed)
        res.append(gr.Textbox.update(value=text, visible=True))
        res.append(gr.Audio.update(value=(cfg.tts.sample_rate, gen.cpu().numpy()), visible=True))
        res.append(gr.Button.update(visible=True))
    n_utterance = len(texts)

    # padding
    if (delta := cfg.editor.max_utterance - n_utterance) > 0:
        for _ in range(delta):
            res.append(gr.Textbox.update(visible=False))
            res.append(gr.Audio.update(visible=False))
            res.append(gr.Button.update(visible=False))
    # clipping
    else:
        warnings.warn("Output is clipped!")
        pass
    return [speaker_vector, len(texts)] + res


def reread(text, speaker_vector):
    voice_samples = speaker_vector['voice_samples']
    conditioning_latents = speaker_vector['conditioning_latents']

    gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                              preset=preset, k=cfg.tts.candidates, use_deterministic_seed=cfg.tts.seed)
    return cfg.tts.sample_rate, gen.cpu().numpy()


with gr.Blocks() as editor:
    speaker_vector = gr.State(dict())
    outputs = []
    outputs.append(speaker_vector)
    with gr.Row() as row0:
        with gr.Column() as col0:
            text = gr.Textbox(label='Text for synthesis')
            reference_audio = gr.Audio(label='reference audio')
            speaker_name = gr.Textbox(value="joe", label='Speaker name')
            button = gr.Button(value='Go!')
            outputs.append(gr.Number(label='number of utterances'))

        with gr.Column(variant='compact') as col1:
            for i in range(cfg.editor.max_utterance):
                utterance = gr.Textbox(label=f'utterance_{i}', visible=False)
                audio = gr.Audio(label=f'audio_{i}', visible=False)

                try_again = gr.Button(value='try again', visible=False)
                try_again.click(fn=reread, inputs=[utterance, speaker_vector], outputs=[audio])

                outputs.extend([utterance, audio, try_again])

        button.click(fn=read, inputs=[text, reference_audio, speaker_name, speaker_vector], outputs=outputs)

    gr.Markdown("Text examples")
    gr.Examples([example_text], [text])
    gr.Markdown("Audio examples")
    gr.Examples([example_voice_sample_path], [reference_audio])
