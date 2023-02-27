import re
from collections import Counter
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import Levenshtein

from datatypes import RawUtterance

timecode_re = re.compile(r"\[[\d\s:\.\->]+\]")
time_re = re.compile(r"[\d:\.]+")
raw_speaker_re = re.compile(r"\w+")
not_raw_speaker_re = re.compile(r"[^a-z0-9_]")
speaker_re = re.compile(r"{\w+}")
double_new_line = re.compile(r"\n\s*\n")
punctuation_re = re.compile(r"[^\w\s]")
acronym_re = re.compile(r"[A-Z]{2,}")
email_re = re.compile(r"\w+@\w+\.\w+")
multi_space_re = re.compile(r" +")


def split_on_speaker_change(raw_text: str):
    """
    speakers are expected to be enclosed in {}
    """

    speaker_matches = speaker_re.finditer(raw_text)
    if not speaker_matches:
        return [(None, raw_text)]

    s, e = next(speaker_matches).span()
    pos = e
    spkr = raw_text[s: e].replace('{', '').replace('}', '')
    spks = [spkr]
    segs = []
    for match in speaker_matches:
        s, e = match.span()
        seg = raw_text[pos: s]
        spk = raw_text[s: e].replace('{', '').replace('}', '')
        segs.append(seg)
        spks.append(spk)
        pos = e

    segs.append(raw_text[pos: len(raw_text)])
    return zip(spks, segs)


def split_on_raw_utterances(raw_text: str) -> list[RawUtterance]:
    raw_text = raw_text.strip()
    utterances = []
    for idx, raw_utter in enumerate(double_new_line.split(raw_text)):

        utter_lines = raw_utter.strip().split('\n')
        utter_lines = list(map(lambda x: x.lstrip(), utter_lines))

        timecode = None
        if timecode_re.match(utter_lines[0]):
            timecode = str(utter_lines.pop(0))

        assert speaker_re.match(utter_lines[0]), utter_lines
        speaker = utter_lines.pop(0)
        speaker = speaker.replace('{', '').replace('}', '')
        text = ' '.join(utter_lines)
        utterances.append(RawUtterance(timecode, speaker, text))
    return utterances


def compute_string_similarity(str1: str, str2: str, normalize=True) -> float:
    """
    String similarity ignoring punctuation.
    """
    if normalize:
        str1 = normalize_text(str1)
        str2 = normalize_text(str2)

    dist = Levenshtein.distance(str1, str2)
    return round(1 - dist / max(len(str1), len(str2)), 3)


def acronym_preprocessing(text):
    # So far it's a table based approach
    # return re.sub('AI', 'Artificial intelligence', text)
    # re sub accepts functions as a repl arguments, are cool!
    return re.sub(acronym_re, lambda m: '.'.join(m.group()), text)


def normalize_text(text: str) -> str:
    text = re.sub(punctuation_re, '', text)
    text = text.strip().lower()
    return text


def find_single_repetition(stt_str: str, tts_str: str) -> Optional[str]:
    """
    This algorithm  looks for substring repetitions withing a string.
    It's designed to handle a single fact of repetition of an arbitrary pattern in stt_string.
    Repetition patter will be enclosed in square brackets.
    Example:
    they are trying to blackmail us with that danger that danger is there that danger is there that cannot be ignored
    they are trying to blackmail us with that danger [ that danger is there ] that danger is there that cannot be ignored

    """

    stt_tokens = stt_str.split()
    tts_tokens = tts_str.split()

    freq_stt = Counter(stt_tokens)
    freq_tts = Counter(tts_tokens)

    repeated_words = Counter()
    for word, stt_word_cnt in freq_stt.items():
        if (tts_word_cnt := freq_tts.get(word)) is not None:
            num_repetitions = stt_word_cnt - tts_word_cnt
            if num_repetitions > 0:
                repeated_words[word] = num_repetitions

    if repeated_words.total() == 0:
        return

    mask = np.zeros(len(stt_tokens), dtype=np.int8)
    for elem in repeated_words.elements():
        for i in range(len(stt_tokens)):
            if elem == stt_tokens[i]:
                mask[i] = 1

    non_zero_idxs = np.concatenate(([0], (mask != 0).view(np.int8), [0]))
    # find all runs
    runs = np.where(np.abs(np.diff(non_zero_idxs)) == 1)[0].reshape(-1, 2)
    # find longest run
    longest_run = sorted(runs, key=lambda itm: itm[1] - itm[0])[-1]
    # try to place braces:
    start, end = longest_run
    assert end - start >= repeated_words.total() * 2

    kernel_size = repeated_words.total()
    found = False
    for i in range(start, end - kernel_size):
        if Counter(stt_tokens[i: i + kernel_size]) == repeated_words:
            found = True
            break
    if found:
        res = ' '.join(stt_tokens[:i] + ['['] + stt_tokens[i: i + kernel_size] + [']'] + stt_tokens[i + kernel_size:])
        return res

    return


def gradio_read_audio_data(audio_data: tuple[int, np.ndarray] | str | Path) -> (torch.Tensor, int):
    dtype = torch.float32
    if isinstance(audio_data, Path | str):
        import soundfile as sf
        waveform, sample_rate = sf.read(audio_data, dtype=str(dtype).split('.')[-1])
    else:
        sample_rate, waveform = audio_data
    return torch.from_numpy(waveform).to(dtype), sample_rate
