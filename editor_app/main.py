from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import gradio as gr
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import SessionLocal, engine

from .editor2 import editor
from .transcriber import transcriber
from .translator import translator

models.Base.metadata.create_all(bind=engine)

app = FastAPI()


# Dependency
def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
def main():
    return """
    <html>
        <head>
            <title>ProsodyLab tools</title>
        </head>
        <body>
            <h1>How To</h1>
            <ol type=1>
                <li>Transcribe audio file in <a href="/transcriber">transcriber</a> </li>
                <li>Translate transcript to English <a href="/translator">translator</a> </li>
                <li>Add speakers with samples of their speech in <a href="/editor">editor</a> </li>
                <li>Add speakers enclosed in curly braces in translated text and run audio generation in <a href="/editor">editor</a> </li>
                <p>
                Example: <br><br>
                {taras_sereda}<br>
                BandCamp is awesome!<br>
                {host}<br>
                Listen to the music, that's about community, culture and something unexplainable. Dope!<br>
                </p>
            </ol>
        </body>
    </html>
    """


@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail='Email already registered')
    return crud.create_user(db=db, user=user)


@app.get("/users/", response_model=list[schemas.User])
def get_users(db: Session = Depends(get_db)):
    return crud.get_users(db=db)


app.mount("/editor", gr.routes.App.create_app(editor))
app.mount("/transcriber", gr.routes.App.create_app(transcriber))
app.mount("/translator", gr.routes.App.create_app(translator))
