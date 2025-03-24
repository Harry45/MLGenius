import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# if running on Docker
SQLALCHEMY_URL = f"postgresql://postgres:admin@db:5432/PostPal"

# if running locally
# SQLALCHEMY_URL = f"postgresql://postgres:admin@localhost:5432/PostPal"

# then create engine
engine = create_engine(SQLALCHEMY_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
