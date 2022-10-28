from sqlalchemy import Column, Integer, String, ForeignKey, Table
from sqlalchemy.orm import relationship

from .database import Base

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
    projects = relationship("Project", back_populates="owner")
    # speaker = relationship("Speaker", secondary=user_to_speaker, back_populates="users")


class Project(Base):
    __tablename__ = "project"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    owner_id = Column(Integer, ForeignKey("user.id"))
    owner = relationship("User", back_populates="projects")
#     utterance = relationship("Utterance")
#     speaker = relationship("Speaker", secondary=project_to_speaker, back_populates="projects")
#
#
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
# class Speaker(Base):
#     __tablename__ = "speaker"
#
#     id = Column(Integer, primary_key=True)
#     name = Column(String)
#     speaker_dir = Column(String)
#     utterance = relationship("Utterance")
#     project = relationship("Project", secondary=project_to_speaker, back_populates="speakers")
#     user = relationship("User", secondary=user_to_speaker, back_populates="speakers")
