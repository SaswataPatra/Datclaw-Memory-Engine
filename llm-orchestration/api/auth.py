"""
DAPPY Authentication Module
JWT-based auth with ArangoDB user storage and bcrypt password hashing.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel, Field, EmailStr

logger = logging.getLogger(__name__)

security = HTTPBearer()

# bcrypt has a 72-byte password limit; truncate consistently (passlib did this implicitly).
BCRYPT_MAX_BYTES = 72


def _password_bytes(password: str) -> bytes:
    b = password.encode("utf-8")
    if len(b) > BCRYPT_MAX_BYTES:
        logger.warning("Password truncated to %s bytes for bcrypt", BCRYPT_MAX_BYTES)
        return b[:BCRYPT_MAX_BYTES]
    return b

DEFAULT_SECRET_KEY = "dappy-dev-secret-change-in-production"
DEFAULT_ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=6)
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str
    display_name: Optional[str] = None
    expires_in: int


class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    display_name: Optional[str] = None
    created_at: str


class AuthManager:
    """Manages user authentication against ArangoDB."""

    COLLECTION = "users"

    def __init__(self, arango_db, secret_key: str = None, algorithm: str = None,
                 token_expire_minutes: int = None):
        self.db = arango_db
        self.secret_key = secret_key or DEFAULT_SECRET_KEY
        self.algorithm = algorithm or DEFAULT_ALGORITHM
        self.token_expire_minutes = token_expire_minutes or DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.db.has_collection(self.COLLECTION):
            col = self.db.create_collection(self.COLLECTION)
            col.add_hash_index(fields=["username"], unique=True)
            col.add_hash_index(fields=["email"], unique=True)
            logger.info("Created 'users' collection with unique indexes")

    def _hash_password(self, password: str) -> str:
        """Hash with bcrypt (compatible with hashes produced by passlib)."""
        raw = bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt())
        return raw.decode("ascii")

    def _verify_password(self, plain: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(
                _password_bytes(plain),
                hashed.encode("ascii"),
            )
        except (ValueError, TypeError):
            return False

    def _create_token(self, user_id: str, username: str) -> tuple[str, int]:
        expires_delta = timedelta(minutes=self.token_expire_minutes)
        expire = datetime.utcnow() + expires_delta
        payload = {
            "sub": user_id,
            "username": username,
            "exp": expire,
            "iat": datetime.utcnow(),
        }
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token, int(expires_delta.total_seconds())

    def decode_token(self, token: str) -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e

    def signup(self, req: SignupRequest) -> Dict[str, Any]:
        col = self.db.collection(self.COLLECTION)

        if col.find({"username": req.username}).count() > 0:
            raise HTTPException(status_code=409, detail="Username already taken")
        if col.find({"email": req.email}).count() > 0:
            raise HTTPException(status_code=409, detail="Email already registered")

        now = datetime.utcnow().isoformat()
        user_doc = {
            "username": req.username,
            "email": req.email,
            "password_hash": self._hash_password(req.password),
            "display_name": req.display_name or req.username,
            "created_at": now,
            "updated_at": now,
        }
        meta = col.insert(user_doc)
        user_id = meta["_key"]
        token, expires_in = self._create_token(user_id, req.username)

        return {
            "access_token": token,
            "token_type": "bearer",
            "user_id": user_id,
            "username": req.username,
            "display_name": user_doc["display_name"],
            "expires_in": expires_in,
        }

    def login(self, req: LoginRequest) -> Dict[str, Any]:
        col = self.db.collection(self.COLLECTION)
        cursor = col.find({"username": req.username}, limit=1)
        users = list(cursor)

        if not users:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        user = users[0]
        if not self._verify_password(req.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid username or password")

        user_id = user["_key"]
        token, expires_in = self._create_token(user_id, user["username"])

        return {
            "access_token": token,
            "token_type": "bearer",
            "user_id": user_id,
            "username": user["username"],
            "display_name": user.get("display_name"),
            "expires_in": expires_in,
        }

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        col = self.db.collection(self.COLLECTION)
        try:
            doc = col.get(user_id)
            if doc:
                return {
                    "user_id": doc["_key"],
                    "username": doc["username"],
                    "email": doc["email"],
                    "display_name": doc.get("display_name"),
                    "created_at": doc.get("created_at"),
                }
        except Exception:
            return None
        return None


# Singleton – set after startup
_auth_manager: Optional[AuthManager] = None


def set_auth_manager(manager: AuthManager):
    global _auth_manager
    _auth_manager = manager


def get_auth_manager() -> AuthManager:
    if _auth_manager is None:
        raise HTTPException(status_code=503, detail="Auth service not initialized")
    return _auth_manager


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """FastAPI dependency that validates JWT and returns the current user."""
    manager = get_auth_manager()
    payload = manager.decode_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user
