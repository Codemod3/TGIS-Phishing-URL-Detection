import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in environment variables. Please check your .env file.")

# Create the SQLAlchemy engine
# Note: For PostgreSQL, the default is synchronous. 
engine = create_engine(DATABASE_URL)

# Create a SessionLocal class for database transactions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for our ORM models to inherit from
Base = declarative_base()

def get_db():
    """
    Dependency to provide a database session for FastAPI requests.
    Ensures the session is correctly closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
