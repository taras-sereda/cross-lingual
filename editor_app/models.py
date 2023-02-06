import pathlib
from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, Boolean, Float
from sqlalchemy.orm import relationship

from . import data_root
from .database import Base


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    projects = relationship("Project", back_populates="owner", lazy='joined')
    speakers = relationship("Speaker", back_populates="owner", lazy='joined')

    def get_user_data_root(self) -> pathlib.Path:
        dir_name = f'{self.id:03}_{self.name.lower()}'
        dir_path = data_root.joinpath(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

        return dir_path


class Project(Base):
    __tablename__ = "project"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    date_created = Column(DateTime, nullable=False)
    date_completed = Column(DateTime)
    owner_id = Column(Integer, ForeignKey("user.id"))
    owner = relationship("User", back_populates="projects")
    utterances = relationship("Utterance", back_populates="project", cascade="all,delete-orphan")

    def get_project_data_root(self) -> pathlib.Path:
        project_path: pathlib.Path = self.owner.get_user_data_root().joinpath('projects', self.title.strip())
        if not project_path.exists():
            project_path.mkdir(parents=True, exist_ok=True)
        return project_path


class Speaker(Base):
    __tablename__ = "speaker"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    owner_id = Column(Integer, ForeignKey("user.id"))
    owner: User = relationship("User", back_populates="speakers", lazy='joined')
    utterances = relationship("Utterance", back_populates="speaker")

    def get_speaker_data_root(self) -> pathlib.Path:
        speaker_dir_path = self.owner.get_user_data_root().joinpath('voices', f'{self.name.lower()}')
        if not speaker_dir_path.exists():
            speaker_dir_path.mkdir(parents=True, exist_ok=True)

        return speaker_dir_path


class Utterance(Base):
    __tablename__ = "utterance"

    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)
    utterance_idx = Column(Integer, nullable=False)  # ordinal idx withing the project
    date_started = Column(DateTime, nullable=False)
    date_completed = Column(DateTime)
    project_id = Column(Integer, ForeignKey("project.id"))
    speaker_id = Column(Integer, ForeignKey("speaker.id"))
    timecode = Column(String, nullable=True)
    project = relationship("Project", back_populates="utterances")
    speaker = relationship("Speaker", back_populates="utterances", lazy="joined")
    utterance_stt = relationship("UtteranceSTT", backref='utter_stt')

    def get_audio_path(self) -> pathlib.Path:
        return self.project.get_project_data_root().joinpath(f'{self.utterance_idx}.wav')


class UtteranceSTT(Base):
    __tablename__ = "utterance_stt"

    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    levenstein_similarity = Column(Float)
    orig_utterance_id = Column(Integer, ForeignKey("utterance.id"))
