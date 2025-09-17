# backend/auth.py
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import jwt

# Choose one scheme:
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# For Argon2, use: pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

SECRET_KEY = "CHANGE_ME_TO_A_64+_CHAR_RANDOM_SECRET"  # put in env var for prod
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(sub: str, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    to_encode = {"sub": sub, "exp": datetime.utcnow() + timedelta(minutes=expires_minutes)}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
