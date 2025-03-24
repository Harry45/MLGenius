from pydantic import BaseModel


class PostBase(BaseModel):
    content: str
    title: str

    class Config:
        from_attributes = True


class CreatePost(PostBase):
    class Config:
        from_attributes = True
