from dataclasses import dataclass


@dataclass
class RawUtterance:
    timecode: str
    speaker: str
    text: str
