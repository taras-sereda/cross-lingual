import pathlib
from sqlalchemy import Column, Integer, String, ForeignKey, Table
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
    name = Column(String, nullable=False)
    owner_id = Column(Integer, ForeignKey("user.id"))
    owner = relationship("User", back_populates="projects")
#     utterance = relationship("Utterance")
#     speaker = relationship("Speaker", secondary=project_to_speaker, back_populates="projects")


class Speaker(Base):
    __tablename__ = "speaker"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    owner_id = Column(Integer, ForeignKey("user.id"))
    owner: User = relationship("User", back_populates="speakers", lazy='joined')
    # utterance = relationship("Utterance")
    # project = relationship("Project", secondary=project_to_speaker, back_populates="speakers")

    def get_speaker_data_root(self) -> pathlib.Path:
        speaker_dir_path = self.owner.get_user_data_root().joinpath('voices', f'{self.id}_{self.name.lower()}')
        if not speaker_dir_path.exists():
            speaker_dir_path.mkdir(parents=True, exist_ok=True)

        return speaker_dir_path


# class Utterance(Base):
#     __tablename__ = "utterance"
#
#     id = Column(Integer, primary_key=True)
#     project_id = Column(Integer, ForeignKey("project.id"))
#     speaker_id = Column(Integer, ForeignKey("speaker.id"))
#     text = Column(String)
#     audio_path = Column(String)
#
#
