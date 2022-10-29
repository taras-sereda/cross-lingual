from sqlalchemy.orm import Session

from . import crud
from .database import SessionLocal


def get_user_by_mail_gradio(email: str):
    db: Session = SessionLocal()
    user = crud.get_user_by_email(db=db, email=email)
    db.close()

    return user


def create_speaker_gradio(name: str, user_id: int):
    db: Session = SessionLocal()
    speaker = crud.create_speaker(db, name, user_id)
    db.close()

    return speaker
