"""
Core Database Module for Enhanced Trading Platform
Provides async database connectivity and ORM models
"""
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Float, DateTime, Boolean, Text, JSON
from datetime import datetime
from typing import Optional, Dict, Any, AsyncGenerator
import structlog
import os

logger = structlog.get_logger()

# Database URL for the dependency, defaulting to in-memory SQLite for safety.
# Tests will override this. main_enhanced.py uses its own Database class instance.
DATABASE_URL_FOR_DEPENDENCY = os.getenv("DATABASE_URL_CORE", "sqlite+aiosqlite:///:memory:")

_async_engine_for_dependency = create_async_engine(DATABASE_URL_FOR_DEPENDENCY, echo=False)
_async_session_maker_for_dependency = async_sessionmaker(
    _async_engine_for_dependency, class_=AsyncSession, expire_on_commit=False
)

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with _async_session_maker_for_dependency() as session:
        yield session

def get_database_engine(): # Synchronous for now, as its usage isn't clear
    return _async_engine_for_dependency


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


class User(Base):
    """User account model"""
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    settings: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)


class Exchange(Base):
    """Exchange configuration model"""
    __tablename__ = "exchanges"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    api_key: Mapped[str] = mapped_column(String(255), nullable=False)
    api_secret: Mapped[str] = mapped_column(String(255), nullable=False)
    sandbox: Mapped[bool] = mapped_column(Boolean, default=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    config: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)


class Trade(Base):
    """Trade execution model"""
    __tablename__ = "trades"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    exchange_id: Mapped[int] = mapped_column(Integer, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # buy/sell
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)  # market/limit/stop
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    filled: Mapped[float] = mapped_column(Float, default=0.0)
    remaining: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    additional_info: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)


class Strategy(Base):
    """Trading strategy model"""
    __tablename__ = "strategies"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    strategy_type: Mapped[str] = mapped_column(String(50), nullable=False)
    parameters: Mapped[Dict] = mapped_column(JSON, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    performance_metrics: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)


class Portfolio(Base):
    """Portfolio tracking model"""
    __tablename__ = "portfolios"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    total_value: Mapped[float] = mapped_column(Float, default=0.0)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    pnl_percentage: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    allocations: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)


class RiskProfile(Base):
    """User risk profile model"""
    __tablename__ = "risk_profiles"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    max_position_size: Mapped[float] = mapped_column(Float, default=0.1)
    max_daily_loss: Mapped[float] = mapped_column(Float, default=0.05)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.2)
    risk_per_trade: Mapped[float] = mapped_column(Float, default=0.02)
    max_open_trades: Mapped[int] = mapped_column(Integer, default=5)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    custom_rules: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)


class Database:
    """Enhanced database manager with connection pooling and health checks"""
    
    def __init__(self, url: str, echo: bool = False):
        self.url = url
        self.engine = create_async_engine(
            url,
            echo=echo,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        self._connected = False
    
    async def connect(self):
        """Initialize database connection and create tables"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._connected = True
            logger.info("Database connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self):
        """Close database connection"""
        await self.engine.dispose()
        self._connected = False
        logger.info("Database disconnected")
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            async with self.async_session_maker() as session:
                await session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_session(self) -> AsyncSession:
        """Get database session"""
        return self.async_session_maker()
    
    async def execute_query(self, query: str, params: Optional[Dict] = None):
        """Execute raw SQL query"""
        async with self.async_session_maker() as session:
            result = await session.execute(query, params or {})
            await session.commit()
            return result
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self._connected
