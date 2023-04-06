import enum
from collections import namedtuple
from dataclasses import dataclass


# A - URL
# B - mail
# C - generated youtube URL
# D - date processed
# E - status
# F - score
class Cells(enum.Enum):
    A = 1
    B = 2
    C = 3
    D = 4
    E = 5
    F = 6


@dataclass
class RawUtterance:
    timecode: str
    speaker: str
    text: str

PseudoFile = namedtuple("PseudoFile", ["name"])