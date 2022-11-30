import re

timecode_re = re.compile(r"\[[\d\s:\.\->]+\]")
speaker_re = re.compile(r"{\w+}")
double_new_line = re.compile(r"\n\s*\n")


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


def convert_text_to_segments(raw_text: str):
    speaker_re = re.compile(r"\[\w+\]")
    char_count = 0
    segments = []
    for idx, utterance in enumerate(double_new_line.split(raw_text)):

        ut = utterance.strip()
        ut_lines = ut.split('\n')
        ut_lines = list(map(lambda x: x.lstrip(), ut_lines))

        timecode = None
        if timecode_re.match(ut_lines[0]):
            timecode = ut_lines.pop(0)

        assert speaker_re.match(ut_lines[0]), ut_lines
        speaker = ut_lines.pop(0)
        text = ' '.join(ut_lines)
        char_count += len(text)

        segments.append({
            'timecode': timecode,
            'speaker': speaker,
            'text': text})
    return segments
