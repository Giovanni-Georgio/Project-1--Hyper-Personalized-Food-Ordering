# agentic_food_backend/utils/security.py

from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
import jwt
from agentic_food_backend.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    if not hashed_password:
        return False
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    """Decode JWT access token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token with longer expiration"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)  # 7 days for refresh token
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_refresh_token(token: str) -> Optional[dict]:
    """Verify and decode refresh token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            return None
        return payload
    except jwt.PyJWTError:
        return None

class SecurityUtils:
    """Utility class for security-related operations"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password"""
        return get_password_hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password"""
        return verify_password(plain_password, hashed_password)
    
    @staticmethod
    def create_tokens(user_id: int, email: str) -> dict:
        """Create both access and refresh tokens"""
        token_data = {"sub": str(user_id), "email": email}
        
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
        }
    
    @staticmethod
    def refresh_access_token(refresh_token: str) -> Optional[dict]:
        """Create new access token from refresh token"""
        payload = verify_refresh_token(refresh_token)
        if not payload:
            return None
        
        # Create new access token
        token_data = {
            "sub": payload.get("sub"),
            "email": payload.get("email")
        }
        
        new_access_token = create_access_token(token_data)
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    
    @staticmethod
    def validate_token_payload(payload: dict) -> bool:
        """Validate token payload structure"""
        required_fields = ["sub", "email", "exp"]
        return all(field in payload for field in required_fields)

# Global security utils instance
security_utils = SecurityUtils()