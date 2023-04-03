from sqlalchemy import and_
from sqlalchemy.orm import Session

from db import models, schemas


def get_user(db: Session, user_id: int) -> models.User:
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str, ensure_exists=True) -> models.User:
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user and ensure_exists:
        raise Exception(f"User {email} not found. Provide valid email")
    return user


def get_users(db: Session) -> list[models.User]:
    return db.query(models.User).all()


def create_user(db: Session, user: schemas.UserCreate) -> models.User:
    db_user = models.User(email=user.email, name=user.name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def create_speaker(db: Session, name: str, cross_project_id: int) -> models.Speaker:
    db_speaker = models.Speaker(name=name, cross_project_id=cross_project_id)
    db.add(db_speaker)
    db.commit()
    db.refresh(db_speaker)
    return db_speaker


def get_speaker_by_name(db: Session, name: str, cross_project_id: int) -> models.Speaker:
    return db.query(models.Speaker).filter(and_(models.Speaker.name == name, models.Speaker.cross_project_id == cross_project_id)).first()


def create_cross_project(db: Session, cross_project: schemas.CrossProjectCreate, user_id: int) -> models.CrossProject:
    db_project = models.CrossProject(**cross_project.dict(), owner_id=user_id)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project


def get_cross_project_by_title(db: Session, title: str, user_id: int, ensure_exists=True) -> models.CrossProject:
    proj = db.query(models.CrossProject).filter(and_(models.CrossProject.title == title, models.CrossProject.owner_id == user_id)).first()
    if ensure_exists and proj is None:
        raise Exception(f"CrossProject {title} doesn't exists")
    return proj


def create_transcript(db: Session, transcript: schemas.TranscriptCreate, cross_project_id: int) -> models.Transcript:
    db_transcript = models.Transcript(**transcript.dict(), cross_project_id=cross_project_id)
    db.add(db_transcript)
    db.commit()
    db.refresh(db_transcript)
    return db_transcript


def create_translation(db: Session, translation: schemas.TranslationCreate, cross_project_id: int) -> models.Translation:
    db_translation = models.Translation(**translation.dict(), cross_project_id=cross_project_id)
    db.add(db_translation)
    db.commit()
    db.refresh(db_translation)
    return db_translation


def get_translation_by_title_and_lang(db: Session, title: str, lang: str, user_id: int) -> models.Translation:
    cross_proj_db = get_cross_project_by_title(db, title, user_id)
    return db.query(models.Translation).filter(and_(models.Translation.cross_project_id == cross_proj_db.id,
                                                    models.Translation.lang == lang,
                                                    )).first()


def create_utterance(db: Session, utterance: schemas.UtteranceCreate, translation_id: int, speaker_id: int) -> models.Utterance:
    db_utterance = models.Utterance(**utterance.dict(), translation_id=translation_id, speaker_id=speaker_id)
    db.add(db_utterance)
    db.commit()
    db.refresh(db_utterance)
    return db_utterance


def get_utterance(db: Session, utterance_idx: int, translation_id: int) -> models.Utterance:
    return db.query(models.Utterance).filter(
        and_(models.Utterance.utterance_idx == utterance_idx,
             models.Utterance.translation_id == translation_id)).first()


def update_any_db_row(db: Session, db_row, **kwargs):
    columns = db_row.__mapper__.attrs.keys()
    for k, v in kwargs.items():
        if k in columns:
            setattr(db_row, k, v)
    db.commit()
    db.refresh(db_row)
    return db_row


def create_utterance_stt(db: Session, utterance_stt: schemas.UtteranceSTTCreate) -> models.UtteranceSTT:
    db_utterance_stt = models.UtteranceSTT(**utterance_stt.dict())
    db.add(db_utterance_stt)
    db.commit()
    db.refresh(db_utterance_stt)
    return db_utterance_stt


def get_utterances_stt(db: Session) -> list[models.UtteranceSTT]:
    return db.query(models.UtteranceSTT).all()
