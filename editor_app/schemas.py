from pydantic import BaseModel


class ProjectBase(BaseModel):
    name: str


class ProjectCreate(ProjectBase):
    pass


class Project(ProjectBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    email: str
    name: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    projects: list[Project] = []

    class Config:
        orm_mode = True


class SpeakerBase(BaseModel):
    name: str


class SpeakerCreate(SpeakerBase):
    pass


class Speaker(SpeakerBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True
