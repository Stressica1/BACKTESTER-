"""
Test configuration and fixtures for the trading platform
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool
from httpx import AsyncClient
import redis.asyncio as redis

from core.database import Base, get_async_session
from core.redis_client import RedisClient
from core.message_queue import MessageQueueClient
from main_enhanced import app

# Test database URL (in-memory SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture
async def async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    
    # Create test engine
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=True
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async with AsyncSession(engine) as session:
        yield session
    
    # Cleanup
    await engine.dispose()

@pytest_asyncio.fixture
async def redis_client() -> AsyncGenerator[redis.Redis, None]:
    """Create a test Redis client."""
    
    # Use a test Redis database (database 15)
    client = redis.Redis(
        host="localhost",
        port=6379,
        db=15,
        decode_responses=True
    )
    
    # Clear test database
    await client.flushdb()
    
    yield client
    
    # Cleanup
    await client.flushdb()
    await client.close()

@pytest_asyncio.fixture
async def message_queue_client() -> AsyncGenerator[MessageQueueClient, None]:
    """Create a test message queue client."""
    
    config = {
        "host": "localhost",
        "port": 5672,
        "username": "guest",
        "password": "guest",
        "vhost": "test_vhost"
    }
    
    client = MessageQueueClient(config)
    
    try:
        await client.connect()
        yield client
    finally:
        await client.close()

@pytest_asyncio.fixture
async def test_client(async_db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create a test HTTP client."""
    
    # Override database dependency
    def override_get_db():
        return async_db_session
    
    app.dependency_overrides[get_async_session] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    # Cleanup
    app.dependency_overrides.clear()

@pytest.fixture
def sample_user_data():
    """Sample user registration data for testing."""
    return {
        "email": "test@example.com",
        "password": "TestPassword123",
        "first_name": "Test",
        "last_name": "User",
        "phone": "+1234567890"
    }

@pytest.fixture
def sample_order_data():
    """Sample order data for testing."""
    return {
        "symbol": "BTCUSD",
        "side": "buy",
        "order_type": "limit",
        "quantity": "0.001",
        "price": "50000.00"
    }

@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "symbol": "BTCUSD",
        "price": "50000.00",
        "volume": "1000.0",
        "bid": "49999.00",
        "ask": "50001.00",
        "high_24h": "51000.00",
        "low_24h": "49000.00",
        "change_24h": "2.5"
    }

# Helper functions for testing
async def create_test_user(client: AsyncClient, user_data: dict) -> dict:
    """Create a test user and return the response."""
    response = await client.post("/auth/register", json=user_data)
    return response.json()

async def login_test_user(client: AsyncClient, email: str, password: str) -> dict:
    """Login a test user and return tokens."""
    response = await client.post(
        "/auth/login",
        json={"email": email, "password": password}
    )
    return response.json()

async def get_auth_headers(client: AsyncClient, user_data: dict) -> dict:
    """Create user, login, and return authorization headers."""
    await create_test_user(client, user_data)
    tokens = await login_test_user(client, user_data["email"], user_data["password"])
    return {"Authorization": f"Bearer {tokens['access_token']}"}

# Performance testing utilities
class PerformanceTimer:
    """Context manager for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = asyncio.get_event_loop().time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = asyncio.get_event_loop().time()
        self.duration = self.end_time - self.start_time

@pytest.fixture
def performance_timer():
    """Fixture for measuring performance."""
    return PerformanceTimer
