from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from gradio import routes
from sqlalchemy.orm import Session

from db import crud, models, schemas
from db.database import SessionLocal, engine

from .editor2 import submitter, editor
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
            <title>CrossLingual tools</title>
        </head>
        <body>
            <h1>How To</h1>
            <ol type=1>
                <li><a href="/transcribe">transcribe</a> audio file</li>
                <li><a href="/translate">translate</a> transcript to English</li>
                <li>Add speakers with samples of their speech in; Add speakers enclosed in curly braces 
                before each paragraph in text and <a href="/submit">submit</a> generaiton job</li>
                <li><a href="/editor">view/editor</a> View and edit generated waveforms</li>
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
    db_user = crud.get_user_by_email(db, email=user.email, ensure_exists=False)
    if db_user:
        raise HTTPException(status_code=400, detail='Email already registered')
    return crud.create_user(db=db, user=user)


@app.get("/users/", response_model=list[schemas.User])
def get_users(db: Session = Depends(get_db)):
    return crud.get_users(db=db)


app = routes.mount_gradio_app(app, translator, "/translate")
app = routes.mount_gradio_app(app, transcriber, "/transcribe")
app = routes.mount_gradio_app(app, submitter, "/submit")
app = routes.mount_gradio_app(app, editor, "/editor")
