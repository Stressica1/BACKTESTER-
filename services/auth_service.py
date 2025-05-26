"""
Authentication Microservice
Handles user authentication, authorization, JWT tokens, and session management
"""

import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import jwt
from fastapi import FastAPI, HTTPException, Depends, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, validator
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update
import logging

from core.database import get_async_session, User, get_database_engine, Database
from core.redis_client import RedisClient
from core.message_queue import MessageQueueClient, QueueMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app for authentication service
auth_app = FastAPI(
    title="Trading Platform Authentication Service",
    description="Microservice for user authentication and authorization",
    version="1.0.0",
    docs_url="/auth/docs",
    redoc_url="/auth/redoc"
)

# CORS middleware
auth_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security configurations
SECRET_KEY = "your-super-secret-jwt-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
security = HTTPBearer()

# Pydantic models for requests/responses
class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    phone: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    permissions: List[str]

class UserProfile(BaseModel):
    id: str
    email: str
    first_name: str
    last_name: str
    phone: Optional[str]
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]
    permissions: List[str]

class PasswordReset(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

class TokenRefresh(BaseModel):
    refresh_token: str

class PermissionUpdate(BaseModel):
    user_id: str
    permissions: List[str]

# Authentication service class
class AuthenticationService:
    """Core authentication service with enterprise features"""
    
    def __init__(self, database: Database, redis: RedisClient, message_queue: MessageQueueClient):
        self.db_instance = database
        self.redis_client = redis
        self.message_queue = message_queue
        
    async def initialize(self):
        """Initialize service dependencies"""
        logger.info("AuthenticationService initialized with injected dependencies.")
        pass
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def generate_tokens(self, user: User) -> Dict[str, Any]:
        """Generate access and refresh tokens for a user"""
        now = datetime.utcnow()
        
        # Access token payload
        access_payload = {
            "sub": str(user.id),
            "email": user.email,
            "permissions": user.permissions or [],
            "iat": now,
            "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            "type": "access"
        }
        
        # Refresh token payload
        refresh_payload = {
            "sub": str(user.id),
            "iat": now,
            "exp": now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
            "type": "refresh",
            "jti": secrets.token_urlsafe(32)  # Unique token ID
        }
        
        access_token = jwt.encode(access_payload, SECRET_KEY, algorithm=ALGORITHM)
        refresh_token = jwt.encode(refresh_payload, SECRET_KEY, algorithm=ALGORITHM)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "token_type": "bearer"
        }
    
    async def create_user(self, user_data: UserRegistration, db: AsyncSession) -> User:
        """Create a new user account"""
        
        # Check if user already exists
        result = await db.execute(
            select(User).where(User.email == user_data.email)
        )
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Create new user
        hashed_password = self.hash_password(user_data.password)
        
        new_user = User(
            email=user_data.email,
            password_hash=hashed_password,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            phone=user_data.phone,
            is_active=True,
            is_verified=False,  # Email verification required
            permissions=["basic_trading", "view_portfolio"],  # Default permissions
            created_at=datetime.utcnow()
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        # Send welcome notification
        await self._send_welcome_notification(new_user)
        
        logger.info(f"New user created: {new_user.email}")
        return new_user
    
    async def authenticate_user(self, email: str, password: str, db: AsyncSession) -> Optional[User]:
        """Authenticate user credentials"""
        
        result = await db.execute(
            select(User).where(User.email == email)
        )
        user = result.scalar_one_or_none()
        
        if not user or not self.verify_password(password, user.password_hash):
            return None
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account is disabled"
            )
        
        # Update last login
        await db.execute(
            update(User)
            .where(User.id == user.id)
            .values(last_login=datetime.utcnow())
        )
        await db.commit()
        
        return user
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            
            # Check if token is blacklisted
            if await self.redis_client.get(f"blacklist:{token}"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def refresh_access_token(self, refresh_token: str, db: AsyncSession) -> Dict[str, Any]:
        """Refresh access token using refresh token"""
        
        try:
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload"
                )
            
            # Get user from database
            result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive"
                )
            
            # Generate new tokens
            return self.generate_tokens(user)
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
    
    async def logout_user(self, token: str) -> bool:
        """Logout user by blacklisting the token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            exp = payload.get("exp")
            
            if exp:
                # Calculate TTL for blacklist
                ttl = exp - datetime.utcnow().timestamp()
                if ttl > 0:
                    await self.redis_client.setex(
                        f"blacklist:{token}",
                        int(ttl),
                        "blacklisted"
                    )
            
            return True
            
        except jwt.JWTError:
            return False
    
    async def _send_welcome_notification(self, user: User) -> None:
        """Send welcome notification via message queue"""
        if not self.message_queue:
            logger.warning("Message queue client not available, skipping welcome notification.")
            return
        
        message_content = {
            "type": "user_registered",
            "user_id": str(user.id),
            "email": user.email,
            "first_name": user.first_name
        }
        
        try:
            await self.message_queue.publish_message(
                exchange_name="user_events",
                routing_key="user.registered",
                message=QueueMessage(content=message_content, content_type="application/json")
            )
            logger.info(f"Welcome notification sent for user {user.email}")
        except Exception as e:
            logger.error(f"Failed to send welcome notification for {user.email}: {e}")

# Global service instance
# This instance is used by the FastAPI routes defined in this file.
# For microservice architecture, main_enhanced.py will manage instances.
# auth_service = AuthenticationService() # Commented out: This will fail with new constructor and DI handled by main_enhanced.py

# Dependency functions
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: AsyncSession = Depends(get_async_session)
) -> User:
    """Get current authenticated user"""
    
    token = credentials.credentials
    payload = await auth_service.verify_token(token)
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def permission_checker(current_user: User = Depends(get_current_user)):
        if permission not in (current_user.permissions or []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return permission_checker

# API Routes
@auth_app.post("/auth/register", response_model=Dict[str, str])
async def register_user(
    user_data: UserRegistration,
    db: AsyncSession = Depends(get_async_session)
):
    """Register a new user account"""
    
    try:
        user = await auth_service.create_user(user_data, db)
        
        return {
            "message": "User registered successfully",
            "user_id": str(user.id),
            "email": user.email
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@auth_app.post("/auth/login", response_model=TokenResponse)
async def login_user(
    login_data: UserLogin,
    db: AsyncSession = Depends(get_async_session)
):
    """Authenticate user and return tokens"""
    
    user = await auth_service.authenticate_user(
        login_data.email,
        login_data.password,
        db
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    tokens = auth_service.generate_tokens(user)
    
    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=tokens["expires_in"],
        user_id=str(user.id),
        permissions=user.permissions or []
    )

@auth_app.post("/auth/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_data: TokenRefresh,
    db: AsyncSession = Depends(get_async_session)
):
    """Refresh access token"""
    
    tokens = await auth_service.refresh_access_token(
        refresh_data.refresh_token,
        db
    )
    
    return TokenResponse(**tokens)

@auth_app.post("/auth/logout")
async def logout_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Logout user by blacklisting token"""
    
    token = credentials.credentials
    success = await auth_service.logout_user(token)
    
    if success:
        return {"message": "Logged out successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Logout failed"
        )

@auth_app.get("/auth/profile", response_model=UserProfile)
async def get_user_profile(
    current_user: User = Depends(get_current_user)
):
    """Get current user profile"""
    
    return UserProfile(
        id=str(current_user.id),
        email=current_user.email,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        phone=current_user.phone,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at,
        last_login=current_user.last_login,
        permissions=current_user.permissions or []
    )

@auth_app.get("/auth/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "authentication",
        "timestamp": datetime.utcnow().isoformat()
    }

# Startup event
@auth_app.on_event("startup")
async def startup_event():
    """Auth service specific startup logic"""
    logger.info("Authentication Service starting up...")
    pass

@auth_app.on_event("shutdown")
async def shutdown_event():
    """Auth service specific shutdown logic"""
    logger.info("Authentication Service shutting down...")
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(auth_app, host="0.0.0.0", port=8001)
