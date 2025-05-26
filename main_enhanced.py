"""
Enhanced Trading Platform - Microservices Architecture
Core Application Factory and Service Discovery
"""
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import structlog
from typing import Dict, Any
import os
from contextlib import asynccontextmanager
from services.auth_service import AuthenticationService
from services.trading_service import TradingService
from services.data_service import DataService
from services.risk_service import RiskService
from services.analytics_service import AnalyticsService
from core.database import Database
from core.redis_client import RedisClient
from core.message_queue import MessageQueue
from core.monitoring import setup_monitoring, setup_tracing


logger = structlog.get_logger()


class ServiceRegistry:
    """Service registry for microservices architecture"""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.health_checks: Dict[str, callable] = {}
    
    def register_service(self, name: str, service: Any, health_check: callable = None):
        """Register a service with optional health check"""
        self.services[name] = service
        if health_check:
            self.health_checks[name] = health_check
        logger.info(f"Service {name} registered")
    
    def get_service(self, name: str):
        """Get a registered service"""
        return self.services.get(name)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health checks on all registered services"""
        results = {}
        for name, check in self.health_checks.items():
            try:
                results[name] = await check()
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = False
        return results


# Global service registry
service_registry = ServiceRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Enhanced Trading Platform")
    
    # Initialize core infrastructure
    await setup_core_infrastructure()
    
    # Initialize services
    await initialize_services()
    
    # Setup monitoring and tracing
    setup_monitoring(app)
    setup_tracing(app)
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced Trading Platform")
    await cleanup_services()


async def setup_core_infrastructure():
    """Initialize core infrastructure components"""
    # Database connection
    database = Database(
        url=os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/trading_db")
    )
    await database.connect()
    service_registry.register_service("database", database, database.health_check)
    
    # Redis client
    redis_client = RedisClient(
        url=os.getenv("REDIS_URL", "redis://localhost:6379")
    )
    await redis_client.connect()
    service_registry.register_service("redis", redis_client, redis_client.health_check)
    
    # Message queue
    message_queue = MessageQueue(
        url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    )
    await message_queue.connect()
    service_registry.register_service("message_queue", message_queue, message_queue.health_check)


async def initialize_services():
    """Initialize all microservices"""
    
    # Authentication Service
    auth_service = AuthenticationService(
        database=service_registry.get_service("database"),
        redis=service_registry.get_service("redis"),
        message_queue=service_registry.get_service("message_queue")
    )
    await auth_service.initialize()
    service_registry.register_service("auth", auth_service, auth_service.health_check)
    
    # Data Service
    data_service = DataService(
        database=service_registry.get_service("database"),
        redis=service_registry.get_service("redis"),
        message_queue=service_registry.get_service("message_queue")
    )
    await data_service.initialize()
    service_registry.register_service("data", data_service, data_service.health_check)
    
    # Trading Service
    trading_service = TradingService(
        database=service_registry.get_service("database"),
        message_queue=service_registry.get_service("message_queue"),
        data_service=data_service
    )
    await trading_service.initialize()
    service_registry.register_service("trading", trading_service, trading_service.health_check)
    
    # Risk Service
    risk_service = RiskService(
        database=service_registry.get_service("database"),
        message_queue=service_registry.get_service("message_queue"),
        trading_service=trading_service
    )
    await risk_service.initialize()
    service_registry.register_service("risk", risk_service, risk_service.health_check)
    
    # Analytics Service
    analytics_service = AnalyticsService(
        database=service_registry.get_service("database"),
        redis=service_registry.get_service("redis"),
        data_service=data_service
    )
    await analytics_service.initialize()
    service_registry.register_service("analytics", analytics_service, analytics_service.health_check)


async def cleanup_services():
    """Cleanup all services on shutdown"""
    for name, service in service_registry.services.items():
        try:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
            logger.info(f"Service {name} cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up service {name}: {e}")


def create_enhanced_app() -> FastAPI:
    """Create the enhanced FastAPI application"""
    
    app = FastAPI(
        title="Enhanced Trading Platform",
        description="Professional-grade cryptocurrency trading platform with microservices architecture",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Global exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
        )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Comprehensive health check"""
        service_health = await service_registry.health_check_all()
        overall_health = all(service_health.values())
        
        return {
            "status": "healthy" if overall_health else "unhealthy",
            "services": service_health,
            "timestamp": os.time()
        }
    
    # Service dependency injection
    def get_auth_service():
        return service_registry.get_service("auth")
    
    def get_data_service():
        return service_registry.get_service("data")
    
    def get_trading_service():
        return service_registry.get_service("trading")
    
    def get_risk_service():
        return service_registry.get_service("risk")
    
    def get_analytics_service():
        return service_registry.get_service("analytics")
    
    # Register dependency providers
    app.dependency_overrides[AuthenticationService] = get_auth_service
    app.dependency_overrides[DataService] = get_data_service
    app.dependency_overrides[TradingService] = get_trading_service
    app.dependency_overrides[RiskService] = get_risk_service
    app.dependency_overrides[AnalyticsService] = get_analytics_service
    
    return app


# Create the application instance
app = create_enhanced_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None  # Use structlog configuration
    )
