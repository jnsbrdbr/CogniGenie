from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite (dev)
DATABASE_URL = "sqlite:///./app.db"
# PostgreSQL: DATABASE_URL = "postgresql+psycopg2://user:pass@localhost:5432/mydb"
# MySQL:      DATABASE_URL = "mysql+pymysql://user:pass@localhost:3306/mydb"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# FastAPI dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()