from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from typing import List
import model
from database import engine, SessionLocal
from sqlalchemy.orm import Session

app = FastAPI()
model.Base.metadata.create_all(bind=engine)


class ChoiceBase(BaseModel):
    choice_text: str
    is_correct: bool


class QuestionBase(BaseModel):
    question_text: str
    choices: List[ChoiceBase]


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/questions/{question_id}")
async def read_question(question_id: int, db: Session = Depends(get_db)):
    result = (
        db.query(model.Questions).filter(model.Questions.idx == question_id).first()
    )
    if not result:
        raise HTTPException(status_code=404, detail="Question is not found.")
    return result


@app.get("/choices/{question_id}")
async def read_choices(question_id: int, db: Session = Depends(get_db)):
    result = (
        db.query(model.Choices).filter(model.Choices.question_id == question_id).all()
    )
    if not result:
        raise HTTPException(status_code=404, detail="Choices are not found.")
    return result


@app.post("/questions/")
async def create_questions(question: QuestionBase, db: Session = Depends(get_db)):
    db_question = model.Questions(question_text=question.question_text)
    db.add(db_question)
    db.commit()
    db.refresh(db_question)

    for choice in question.choices:
        db_choices = model.Choices(
            choice_text=choice.choice_text,
            is_correct=choice.is_correct,
            question_id=db_question.idx,
        )
        db.add(db_choices)
    db.commit()
