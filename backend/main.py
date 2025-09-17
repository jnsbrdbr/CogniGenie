# backend/main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from .database import Base, engine, get_db
from .models import User
from .schemas import UserCreate, UserOut, Token
from .auth import hash_password, verify_password, create_access_token, SECRET_KEY, ALGORITHM
from .rag_engine import query_rag, query_rag_with_sources, ensure_ready, reload_index

app = FastAPI(title="Local RAG API")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

class ChatRequest(BaseModel):
    question: str

@app.on_event("startup")
def _startup():
    Base.metadata.create_all(bind=engine)
    ensure_ready()

@app.post("/register", response_model=UserOut)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    user = User(username=payload.username, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.post("/login", response_model=Token)
def login(payload: UserCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(sub=user.username)
    return {"access_token": token, "token_type": "bearer"}

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@app.post("/chat")
def chat(req: ChatRequest, _: User = Depends(get_current_user)):
    answer = query_rag(req.question)
    return {"answer": answer}

@app.post("/chat_with_sources")
def chat_with_sources(req: ChatRequest, _: User = Depends(get_current_user)):
    return query_rag_with_sources(req.question)

@app.post("/reload")
def reload(_: User = Depends(get_current_user)):
    reload_index()
    return {"status": "reloaded"}

@app.get("/health")
def health():
    return {"status": "ok"}

