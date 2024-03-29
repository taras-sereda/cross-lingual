from datetime import datetime

from pydantic import BaseModel


class CrossProjectBase(BaseModel):
    title: str
    media_name: str


class CrossProjectCreate(CrossProjectBase):
    pass


class CrossProject(CrossProjectBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True


class TranslationBase(BaseModel):
    text: str
    lang: str
    date_created: datetime


class TranslationCreate(TranslationBase):
    pass


class Translation(TranslationBase):
    id: int
    owner_id: int
    cross_project_id: int

    class Config:
        orm_mode = True


class TranscriptBase(BaseModel):
    text: str
    lang: str


class TranscriptCreate(TranscriptBase):
    pass


class Transcript(TranscriptBase):
    id: int
    owner_id: int
    cross_project_id: int

    class Config:
        orm_mode = True


class UtteranceBase(BaseModel):
    text: str
    utterance_idx: int
    date_started: datetime
    timecode: str | None = None


class UtteranceCreate(UtteranceBase):
    pass


class Utterance(UtteranceBase):
    id: int
    project_id: int
    speaker_id: int
    date_completed: datetime | None = None

    class Config:
        orm_mode = True


class UtteranceSTTBase(BaseModel):
    text: str
    orig_utterance_id: int
    date: datetime
    levenstein_similarity: float


class UtteranceSTTCreate(UtteranceSTTBase):
    pass


class UtteranceSTT(UtteranceSTTBase):
    id: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    email: str
    name: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    cross_projects: list[CrossProject] = []

    class Config:
        orm_mode = True


class SpeakerBase(BaseModel):
    name: str


class SpeakerCreate(SpeakerBase):
    pass


class Speaker(SpeakerBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True
