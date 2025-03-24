from fastapi import FastAPI
import model
from database import engine, SessionLocal
import posts

app = FastAPI()
model.Base.metadata.create_all(bind=engine)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


app.include_router(posts.router)
