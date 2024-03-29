import pathlib
from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, Float, UniqueConstraint
from sqlalchemy.orm import relationship

from config import data_root
from db.database import Base


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    crosslingual_projects = relationship("CrossProject", back_populates="owner", lazy="joined")

    def get_user_data_root(self) -> pathlib.Path:
        dir_name = f"{self.id:03}_{self.name.lower()}"
        dir_path = data_root.joinpath(dir_name)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path


class CrossProject(Base):
    __tablename__ = "crosslingual_project"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    media_name = Column(String, nullable=False)
    owner_id = Column(Integer, ForeignKey("user.id"))
    owner = relationship("User", back_populates="crosslingual_projects")
    transcript = relationship("Transcript", back_populates="cross_project", lazy="joined")
    translations = relationship("Translation", back_populates="cross_project", lazy="joined")
    speakers = relationship("Speaker", back_populates="cross_project", lazy="joined")

    def get_data_root(self) -> pathlib.Path:
        project_path: pathlib.Path = self.owner.get_user_data_root().joinpath("cross_projects", self.title.strip())
        project_path.mkdir(parents=True, exist_ok=True)
        return project_path

    def get_media_path(self) -> pathlib.Path:
        return self.get_data_root().joinpath(self.media_name)

    def get_raw_wav_path(self, sample_rate=None) -> pathlib.Path:
        if sample_rate:
            ext = f".{sample_rate}.wav"
        else:
            ext = ".wav"
        return self.get_media_path().with_suffix(ext)

    def get_vocals_wav_path(self) -> pathlib.Path:
        vocals_name = f"{self.get_media_path().stem}.vocals.wav"
        return self.get_media_path().parent.joinpath(f"htdemucs/{vocals_name}")



class Transcript(Base):
    __tablename__ = "transcript"

    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    lang = Column(String, nullable=False)
    cross_project_id = Column(Integer, ForeignKey("crosslingual_project.id"))
    cross_project = relationship("CrossProject", back_populates="transcript")

    def get_path(self):
        return self.cross_project.get_data_root().joinpath("transcript.txt")


class Translation(Base):
    __tablename__ = "translation"
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    lang = Column(String, nullable=False)
    date_created = Column(DateTime, nullable=False)
    date_completed = Column(DateTime)
    cross_project_id = Column(Integer, ForeignKey("crosslingual_project.id"))
    cross_project = relationship("CrossProject", back_populates="translations", lazy="joined")
    utterances = relationship("Utterance", back_populates="translation", cascade="all,delete-orphan")

    def get_data_root(self):
        root: pathlib.Path = self.cross_project.get_data_root().joinpath(f"translation_{self.id}")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def get_path(self):
        return self.get_data_root().joinpath(f"translation.{self.lang}.txt")


class Speaker(Base):
    __tablename__ = "speaker"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    alt_name = Column(String)
    cross_project_id = Column(Integer, ForeignKey("crosslingual_project.id"))
    cross_project = relationship("CrossProject", back_populates="speakers", lazy="joined")
    utterances = relationship("Utterance", back_populates="speaker")
    # Withing single cross project, speaker name is enforced being unique.
    UniqueConstraint("name", "cross_project_id", name="uq_name_cross_project_id")

    def get_speaker_data_root(self) -> pathlib.Path:
        speaker_dir_path = self.cross_project.get_data_root().joinpath("voices", f"{self.name.lower()}")
        speaker_dir_path.mkdir(parents=True, exist_ok=True)
        return speaker_dir_path


class Utterance(Base):
    __tablename__ = "utterance"

    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)
    utterance_idx = Column(Integer, nullable=False)  # ordinal idx withing the project
    date_started = Column(DateTime, nullable=False)
    date_completed = Column(DateTime)
    translation_id = Column(Integer, ForeignKey("translation.id"))
    speaker_id = Column(Integer, ForeignKey("speaker.id"))
    timecode = Column(String, nullable=True)
    translation = relationship("Translation", back_populates="utterances")
    speaker = relationship("Speaker", back_populates="utterances", lazy="joined")
    utterance_stt = relationship("UtteranceSTT", backref="utter_stt")

    def get_audio_path(self) -> pathlib.Path:
        return self.translation.get_data_root().joinpath(f"{self.utterance_idx}.wav")


class UtteranceSTT(Base):
    __tablename__ = "utterance_stt"

    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    levenstein_similarity = Column(Float)
    orig_utterance_id = Column(Integer, ForeignKey("utterance.id"))
