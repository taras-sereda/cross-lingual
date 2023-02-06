from sqlalchemy import and_
from sqlalchemy.orm import Session

from . import models, schemas


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> schemas.User:
    return db.query(models.User).filter(models.User.email == email).first()


def get_users(db: Session):
    return db.query(models.User).all()


def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(email=user.email, name=user.name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def create_speaker(db: Session, name: str, user_id: int):
    db_speaker = models.Speaker(name=name, owner_id=user_id)
    db.add(db_speaker)
    db.commit()
    db.refresh(db_speaker)
    return db_speaker


def get_speaker_by_name(db: Session, name: str, user_id: int):
    return db.query(models.Speaker).filter(and_(models.Speaker.name == name, models.Speaker.owner_id == user_id)).first()


def create_project(db: Session, project: schemas.ProjectCreate, user_id: int):
    db_project = models.Project(**project.dict(), owner_id=user_id)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project


def get_project_by_title(db: Session, title: str, user_id: int):
    return db.query(models.Project).filter(and_(models.Project.title == title, models.Project.owner_id == user_id)).first()


def create_utterance(db: Session, utterance: schemas.UtteranceCreate, project_id: int, speaker_id: int):
    db_utterance = models.Utterance(**utterance.dict(), project_id=project_id, speaker_id=speaker_id)
    db.add(db_utterance)
    db.commit()
    db.refresh(db_utterance)
    return db_utterance


def get_utterance(db: Session, utterance_idx: int, project_id: int):
    return db.query(models.Utterance).filter(
        and_(models.Utterance.utterance_idx == utterance_idx,
             models.Utterance.project_id == project_id)).first()


def update_any_db_row(db: Session, db_row, **kwargs):
    columns = db_row.__mapper__.attrs.keys()
    for k, v in kwargs.items():
        if k in columns:
            setattr(db_row, k, v)
    db.commit()
    db.refresh(db_row)
    return db_row


def create_utterance_stt(db: Session, utterance_stt: schemas.UtteranceSTTCreate):
    db_utterance_stt = models.UtteranceSTT(**utterance_stt.dict())
    db.add(db_utterance_stt)
    db.commit()
    db.refresh(db_utterance_stt)
    return db_utterance_stt


def get_utterances_stt(db: Session):
    return db.query(models.UtteranceSTT).all()
