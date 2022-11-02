import re


def split_on_speaker_change(raw_text: str):
    """
    speakers are expected to be enclosed in {}
    """

    speaker_re = re.compile(r"{[\w\s]+}")
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
