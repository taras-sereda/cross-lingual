import pathlib
from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, Boolean
from sqlalchemy.orm import relationship


from .database import Base, cfg


# project_to_speaker = Table(
#     "project_to_speaker",
#     Base.metadata,
#     Column("project_id", ForeignKey("project.id"), primary_key=True),
#     Column("speaker_id", ForeignKey("speaker.id"), primary_key=True),
# )
#
# user_to_speaker = Table(
#     "user_to_speaker",
#     Base.metadata,
#     Column("user_id", ForeignKey("user.id"), primary_key=True),
#     Column("speaker_id", ForeignKey("speaker.id"), primary_key=True),
# )


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    projects = relationship("Project", back_populates="owner")
    speakers = relationship("Speaker", back_populates="owner", lazy='joined')

    def get_user_data_root(self) -> pathlib.Path:
        dir_name = f'{self.id:03}_{self.name.lower()}'
        dir_path = pathlib.Path(cfg.db.data_root).joinpath(dir_name)
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
    completed = Column(Boolean, default=False)
    owner_id = Column(Integer, ForeignKey("user.id"))
    owner = relationship("User", back_populates="projects")
    utterances = relationship("Utterance", back_populates="project")


class Speaker(Base):
    __tablename__ = "speaker"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    owner_id = Column(Integer, ForeignKey("user.id"))
    owner: User = relationship("User", back_populates="speakers", lazy='joined')
    utterances = relationship("Utterance", back_populates="speaker")

    def get_speaker_data_root(self) -> pathlib.Path:
        speaker_dir_path = self.owner.get_user_data_root().joinpath('voices', f'{self.id}_{self.name.lower()}')
        if not speaker_dir_path.exists():
            speaker_dir_path.mkdir(parents=True, exist_ok=True)

        return speaker_dir_path


class Utterance(Base):
    __tablename__ = "utterance"

    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)
    audio_path = Column(String, nullable=False)
    project_id = Column(Integer, ForeignKey("project.id"))
    speaker_id = Column(Integer, ForeignKey("speaker.id"))
    project = relationship("Project", back_populates="utterances")
    speaker = relationship("Speaker", back_populates="utterances")

