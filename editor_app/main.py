from fastapi import Depends, FastAPI, HTTPException
import gradio as gr
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import SessionLocal, engine

from .editor_mockup import editor

models.Base.metadata.create_all(bind=engine)

app = FastAPI()


# Dependency
def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def main():
    return {"message": "ProsodyLab"}


@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail='Email already registered')
    return crud.create_user(db=db, user=user)


@app.get("/users/", response_model=list[schemas.User])
def get_users(db: Session = Depends(get_db)):
    return crud.get_users(db=db)


gradio_editor = gr.routes.App.create_app(editor)
app.mount("/editor", gradio_editor)
