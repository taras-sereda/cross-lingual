import re

import Levenshtein

from datatypes import RawUtterance

timecode_re = re.compile(r"\[[\d\s:\.\->]+\]")
speaker_re = re.compile(r"{\w+}")
double_new_line = re.compile(r"\n\s*\n")
punctuation_re = re.compile(r"[^\w\s]")
acronym_re = re.compile(r"[A-Z]{2,}")
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


def compute_string_similarity(str1: str, str2: str) -> float:
    """
    String similarity ignoring punctuation.
    """
    str1 = re.sub(punctuation_re, '', str1)
    str1 = str1.strip()

    str2 = re.sub(punctuation_re, '', str2)
    str2 = str2.strip()
    dist = Levenshtein.distance(str1, str2)

    return 1 - dist / max(len(str1), len(str2))


def acronym_preprocessing(text):
    # So far it's a table based approach
    # return re.sub('AI', 'Artificial intelligence', text)
    # re sub accepts functions as a repl arguments, are cool!
    return re.sub(acronym_re, lambda m: '.'.join(m.group()), text)
