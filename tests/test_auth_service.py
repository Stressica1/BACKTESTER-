"""
Unit tests for the authentication service
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from services.auth_service import AuthenticationService, auth_service
from core.database import User

@pytest.mark.asyncio
class TestAuthenticationService:
    """Test cases for the authentication service."""
    
    async def test_create_user_success(self, async_db_session: AsyncSession, sample_user_data):
        """Test successful user creation."""
        
        user = await auth_service.create_user(
            sample_user_data,
            async_db_session
        )
        
        assert user.email == sample_user_data["email"]
        assert user.first_name == sample_user_data["first_name"]
        assert user.last_name == sample_user_data["last_name"]
        assert user.is_active is True
        assert user.is_verified is False
        assert user.permissions == ["basic_trading", "view_portfolio"]
    
    async def test_create_user_duplicate_email(self, async_db_session: AsyncSession, sample_user_data):
        """Test user creation with duplicate email fails."""
        
        # Create first user
        await auth_service.create_user(sample_user_data, async_db_session)
        
        # Try to create second user with same email
        with pytest.raises(Exception):  # Should raise HTTPException
            await auth_service.create_user(sample_user_data, async_db_session)
    
    async def test_authenticate_user_success(self, async_db_session: AsyncSession, sample_user_data):
        """Test successful user authentication."""
        
        # Create user
        await auth_service.create_user(sample_user_data, async_db_session)
        
        # Authenticate user
        user = await auth_service.authenticate_user(
            sample_user_data["email"],
            sample_user_data["password"],
            async_db_session
        )
        
        assert user is not None
        assert user.email == sample_user_data["email"]
    
    async def test_authenticate_user_wrong_password(self, async_db_session: AsyncSession, sample_user_data):
        """Test authentication with wrong password fails."""
        
        # Create user
        await auth_service.create_user(sample_user_data, async_db_session)
        
        # Try to authenticate with wrong password
        user = await auth_service.authenticate_user(
            sample_user_data["email"],
            "wrong_password",
            async_db_session
        )
        
        assert user is None
    
    async def test_authenticate_nonexistent_user(self, async_db_session: AsyncSession):
        """Test authentication of nonexistent user fails."""
        
        user = await auth_service.authenticate_user(
            "nonexistent@example.com",
            "password",
            async_db_session
        )
        
        assert user is None
    
    async def test_generate_tokens(self, async_db_session: AsyncSession, sample_user_data):
        """Test JWT token generation."""
        
        # Create user
        user = await auth_service.create_user(sample_user_data, async_db_session)
        
        # Generate tokens
        tokens = auth_service.generate_tokens(user)
        
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"
        assert tokens["expires_in"] == 30 * 60  # 30 minutes
    
    async def test_verify_token_valid(self, async_db_session: AsyncSession, sample_user_data):
        """Test verification of valid token."""
        
        # Create user and generate tokens
        user = await auth_service.create_user(sample_user_data, async_db_session)
        tokens = auth_service.generate_tokens(user)
        
        # Verify token
        payload = await auth_service.verify_token(tokens["access_token"])
        
        assert payload["sub"] == str(user.id)
        assert payload["email"] == user.email
        assert payload["type"] == "access"
    
    async def test_password_hashing(self):
        """Test password hashing and verification."""
        
        password = "test_password_123"
        
        # Hash password
        hashed = auth_service.hash_password(password)
        
        # Verify password
        assert auth_service.verify_password(password, hashed) is True
        assert auth_service.verify_password("wrong_password", hashed) is False

@pytest.mark.asyncio
class TestAuthenticationAPI:
    """Test cases for the authentication API endpoints."""
    
    async def test_register_endpoint_success(self, test_client: AsyncClient, sample_user_data):
        """Test successful user registration via API."""
        
        response = await test_client.post("/auth/register", json=sample_user_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert data["email"] == sample_user_data["email"]
        assert data["message"] == "User registered successfully"
    
    async def test_register_endpoint_invalid_data(self, test_client: AsyncClient):
        """Test registration with invalid data."""
        
        invalid_data = {
            "email": "invalid_email",
            "password": "weak",  # Too short
            "first_name": "",
            "last_name": ""
        }
        
        response = await test_client.post("/auth/register", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    async def test_login_endpoint_success(self, test_client: AsyncClient, sample_user_data):
        """Test successful login via API."""
        
        # Register user first
        await test_client.post("/auth/register", json=sample_user_data)
        
        # Login
        login_data = {
            "email": sample_user_data["email"],
            "password": sample_user_data["password"]
        }
        response = await test_client.post("/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "user_id" in data
        assert "permissions" in data
    
    async def test_login_endpoint_invalid_credentials(self, test_client: AsyncClient, sample_user_data):
        """Test login with invalid credentials."""
        
        # Register user first
        await test_client.post("/auth/register", json=sample_user_data)
        
        # Try to login with wrong password
        login_data = {
            "email": sample_user_data["email"],
            "password": "wrong_password"
        }
        response = await test_client.post("/auth/login", json=login_data)
        
        assert response.status_code == 401
    
    async def test_profile_endpoint_success(self, test_client: AsyncClient, sample_user_data):
        """Test getting user profile via API."""
        
        # Register and login user
        await test_client.post("/auth/register", json=sample_user_data)
        
        login_data = {
            "email": sample_user_data["email"],
            "password": sample_user_data["password"]
        }
        login_response = await test_client.post("/auth/login", json=login_data)
        tokens = login_response.json()
        
        # Get profile
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        response = await test_client.get("/auth/profile", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == sample_user_data["email"]
        assert data["first_name"] == sample_user_data["first_name"]
        assert data["last_name"] == sample_user_data["last_name"]
        assert data["is_active"] is True
        assert "permissions" in data
    
    async def test_profile_endpoint_unauthorized(self, test_client: AsyncClient):
        """Test getting profile without authentication."""
        
        response = await test_client.get("/auth/profile")
        assert response.status_code == 403  # Missing authentication
    
    async def test_logout_endpoint_success(self, test_client: AsyncClient, sample_user_data):
        """Test successful logout via API."""
        
        # Register and login user
        await test_client.post("/auth/register", json=sample_user_data)
        
        login_data = {
            "email": sample_user_data["email"],
            "password": sample_user_data["password"]
        }
        login_response = await test_client.post("/auth/login", json=login_data)
        tokens = login_response.json()
        
        # Logout
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        response = await test_client.post("/auth/logout", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Logged out successfully"
    
    async def test_refresh_token_endpoint_success(self, test_client: AsyncClient, sample_user_data):
        """Test token refresh via API."""
        
        # Register and login user
        await test_client.post("/auth/register", json=sample_user_data)
        
        login_data = {
            "email": sample_user_data["email"],
            "password": sample_user_data["password"]
        }
        login_response = await test_client.post("/auth/login", json=login_data)
        tokens = login_response.json()
        
        # Refresh token
        refresh_data = {"refresh_token": tokens["refresh_token"]}
        response = await test_client.post("/auth/refresh", json=refresh_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
    
    async def test_health_endpoint(self, test_client: AsyncClient):
        """Test health check endpoint."""
        
        response = await test_client.get("/auth/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "authentication"
        assert "timestamp" in data

@pytest.mark.asyncio
class TestAuthenticationPerformance:
    """Performance tests for authentication service."""
    
    async def test_registration_performance(self, test_client: AsyncClient, sample_user_data, performance_timer):
        """Test registration performance."""
        
        with performance_timer:
            response = await test_client.post("/auth/register", json=sample_user_data)
        
        assert response.status_code == 200
        assert performance_timer.duration < 1.0  # Should complete within 1 second
    
    async def test_login_performance(self, test_client: AsyncClient, sample_user_data, performance_timer):
        """Test login performance."""
        
        # Register user first
        await test_client.post("/auth/register", json=sample_user_data)
        
        login_data = {
            "email": sample_user_data["email"],
            "password": sample_user_data["password"]
        }
        
        with performance_timer:
            response = await test_client.post("/auth/login", json=login_data)
        
        assert response.status_code == 200
        assert performance_timer.duration < 0.5  # Should complete within 0.5 seconds
    
    async def test_concurrent_logins(self, test_client: AsyncClient, sample_user_data):
        """Test concurrent login requests."""
        
        # Register user first
        await test_client.post("/auth/register", json=sample_user_data)
        
        login_data = {
            "email": sample_user_data["email"],
            "password": sample_user_data["password"]
        }
        
        # Perform concurrent logins
        import asyncio
        tasks = [
            test_client.post("/auth/login", json=login_data)
            for _ in range(10)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
