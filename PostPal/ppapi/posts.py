from typing import List
from fastapi import HTTPException, Depends, status
from sqlalchemy.orm import Session
import model
import schema
from fastapi import APIRouter
from database import get_db

router = APIRouter(prefix="/posts", tags=["Posts"])


@router.get("/", response_model=List[schema.CreatePost])
def test_posts(db: Session = Depends(get_db)):
    post = db.query(model.Post).all()
    return post


@router.post(
    "/", status_code=status.HTTP_201_CREATED, response_model=List[schema.CreatePost]
)
def create_posts(post_create: schema.CreatePost, db: Session = Depends(get_db)):
    new_post = model.Post(**post_create.model_dump())
    db.add(new_post)
    db.commit()
    db.refresh(new_post)
    return [new_post]
