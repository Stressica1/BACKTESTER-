"""
Trading Engine Microservice
Handles order execution, portfolio management, and trading strategies
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from decimal import Decimal
import logging

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_
from sqlalchemy.orm import selectinload

from core.database import get_async_session, User, Exchange, Trade, Portfolio
from core.redis_client import get_redis_client
from core.message_queue import get_message_queue_client, QueueMessage, MessageHandlers
from services.auth_service import get_current_user, require_permission

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app for trading service
trading_app = FastAPI(
    title="Trading Platform Trading Engine",
    description="Microservice for trade execution and portfolio management",
    version="1.0.0",
    docs_url="/trading/docs",
    redoc_url="/trading/redoc"
)

# CORS middleware
trading_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enums for trading
class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"

# Pydantic models
class OrderRequest(BaseModel):
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    reduce_only: bool = False
    post_only: bool = False
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v
    
    @validator('price')
    def validate_price(cls, v, values):
        if values.get('order_type') in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError('Price is required for limit orders')
        if v is not None and v <= 0:
            raise ValueError('Price must be positive')
        return v

class OrderResponse(BaseModel):
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal]
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    filled_quantity: Decimal = Decimal('0')
    average_price: Optional[Decimal] = None

class PositionInfo(BaseModel):
    symbol: str
    side: PositionSide
    size: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    margin_used: Decimal
    liquidation_price: Optional[Decimal]

class PortfolioSummary(BaseModel):
    total_equity: Decimal
    available_balance: Decimal
    used_margin: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    total_positions: int
    open_orders: int

class TradeExecution(BaseModel):
    trade_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    commission: Decimal
    timestamp: datetime

# Trading Engine Service
class TradingEngineService:
    """Advanced trading engine with order management and risk controls"""
    
    def __init__(self):
        self.redis_client = None
        self.message_queue = None
        self.active_orders: Dict[str, Dict] = {}
        self.positions: Dict[str, Dict] = {}
        self.risk_limits = {
            "max_position_size": Decimal('100000'),
            "max_daily_loss": Decimal('10000'),
            "max_leverage": Decimal('10'),
            "max_orders_per_minute": 60
        }
        
    async def initialize(self):
        """Initialize service dependencies"""
        self.redis_client = get_redis_client()
        self.message_queue = get_message_queue_client()
        
        # Setup message queue consumers
        await self._setup_message_consumers()
        
    async def _setup_message_consumers(self):
        """Setup message queue consumers for trading operations"""
        await self.message_queue.consume_messages(
            "trade.execution",
            self._handle_trade_execution
        )
        
        await self.message_queue.consume_messages(
            "trade.orders",
            self._handle_order_updates
        )
        
    async def create_order(
        self,
        user: User,
        order_request: OrderRequest,
        db: AsyncSession
    ) -> OrderResponse:
        """Create a new trading order"""
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Validate order request
        await self._validate_order(user, order_request, db)
        
        # Check risk limits
        await self._check_risk_limits(user, order_request, db)
        
        # Create order record
        order_data = {
            "order_id": order_id,
            "user_id": str(user.id),
            "symbol": order_request.symbol,
            "side": order_request.side,
            "order_type": order_request.order_type,
            "quantity": order_request.quantity,
            "price": order_request.price,
            "stop_price": order_request.stop_price,
            "status": OrderStatus.PENDING,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "filled_quantity": Decimal('0'),
            "time_in_force": order_request.time_in_force,
            "reduce_only": order_request.reduce_only,
            "post_only": order_request.post_only
        }
        
        # Store order in cache
        await self.redis_client.hset(
            f"orders:{order_id}",
            mapping={k: str(v) for k, v in order_data.items()}
        )
        
        # Add to active orders
        self.active_orders[order_id] = order_data
        
        # Send order to exchange via message queue
        await self.message_queue.publish_trade_execution({
            "action": "create_order",
            "order_id": order_id,
            "order_data": order_data
        })
        
        logger.info(f"Order created: {order_id} for user {user.id}")
        
        return OrderResponse(**order_data)
    
    async def cancel_order(
        self,
        user: User,
        order_id: str,
        db: AsyncSession
    ) -> bool:
        """Cancel an existing order"""
        
        # Get order from cache
        order_data = await self.redis_client.hgetall(f"orders:{order_id}")
        
        if not order_data or order_data.get("user_id") != str(user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Order not found"
            )
        
        current_status = order_data.get("status")
        if current_status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel order with status: {current_status}"
            )
        
        # Update order status
        await self.redis_client.hset(
            f"orders:{order_id}",
            "status", OrderStatus.CANCELLED
        )
        
        # Send cancellation to exchange
        await self.message_queue.publish_trade_execution({
            "action": "cancel_order",
            "order_id": order_id,
            "user_id": str(user.id)
        })
        
        logger.info(f"Order cancelled: {order_id}")
        return True
    
    async def get_orders(
        self,
        user: User,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        limit: int = 100
    ) -> List[OrderResponse]:
        """Get user's orders with optional filtering"""
        
        # Get all order keys for user
        pattern = f"orders:*"
        order_keys = await self.redis_client.keys(pattern)
        
        orders = []
        for key in order_keys[:limit]:
            order_data = await self.redis_client.hgetall(key)
            
            if order_data.get("user_id") != str(user.id):
                continue
                
            if symbol and order_data.get("symbol") != symbol:
                continue
                
            if status and order_data.get("status") != status:
                continue
            
            # Convert data types
            order_data["quantity"] = Decimal(order_data["quantity"])
            order_data["filled_quantity"] = Decimal(order_data.get("filled_quantity", "0"))
            if order_data.get("price"):
                order_data["price"] = Decimal(order_data["price"])
            if order_data.get("average_price"):
                order_data["average_price"] = Decimal(order_data["average_price"])
            
            orders.append(OrderResponse(**order_data))
        
        return orders
    
    async def get_positions(
        self,
        user: User,
        symbol: Optional[str] = None
    ) -> List[PositionInfo]:
        """Get user's positions"""
        
        # Get positions from cache
        pattern = f"positions:{user.id}:*"
        position_keys = await self.redis_client.keys(pattern)
        
        positions = []
        for key in position_keys:
            position_data = await self.redis_client.hgetall(key)
            
            if symbol and position_data.get("symbol") != symbol:
                continue
            
            # Convert data types
            for field in ["size", "entry_price", "mark_price", "unrealized_pnl", 
                         "realized_pnl", "margin_used"]:
                if position_data.get(field):
                    position_data[field] = Decimal(position_data[field])
            
            if position_data.get("liquidation_price"):
                position_data["liquidation_price"] = Decimal(position_data["liquidation_price"])
            
            positions.append(PositionInfo(**position_data))
        
        return positions
    
    async def get_portfolio_summary(
        self,
        user: User,
        db: AsyncSession
    ) -> PortfolioSummary:
        """Get portfolio summary for user"""
        
        # Get portfolio from database
        result = await db.execute(
            select(Portfolio).where(Portfolio.user_id == user.id)
        )
        portfolio = result.scalar_one_or_none()
        
        if not portfolio:
            # Create default portfolio
            portfolio = Portfolio(
                user_id=user.id,
                total_equity=Decimal('0'),
                available_balance=Decimal('0'),
                used_margin=Decimal('0'),
                unrealized_pnl=Decimal('0'),
                realized_pnl=Decimal('0')
            )
            db.add(portfolio)
            await db.commit()
        
        # Count positions and orders
        positions = await self.get_positions(user)
        orders = await self.get_orders(user, status=OrderStatus.OPEN)
        
        return PortfolioSummary(
            total_equity=portfolio.total_equity,
            available_balance=portfolio.available_balance,
            used_margin=portfolio.used_margin,
            unrealized_pnl=portfolio.unrealized_pnl,
            realized_pnl=portfolio.realized_pnl,
            total_positions=len(positions),
            open_orders=len(orders)
        )
    
    async def _validate_order(
        self,
        user: User,
        order_request: OrderRequest,
        db: AsyncSession
    ) -> None:
        """Validate order parameters"""
        
        # Check if symbol is supported
        result = await db.execute(
            select(Exchange).where(Exchange.symbol == order_request.symbol)
        )
        exchange = result.scalar_one_or_none()
        
        if not exchange:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Symbol {order_request.symbol} not supported"
            )
        
        # Validate minimum order size
        if order_request.quantity < exchange.min_order_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Order quantity below minimum: {exchange.min_order_size}"
            )
        
        # Validate price precision
        if order_request.price and exchange.price_precision:
            decimal_places = str(order_request.price)[::-1].find('.')
            if decimal_places > exchange.price_precision:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Price precision exceeds maximum: {exchange.price_precision}"
                )
    
    async def _check_risk_limits(
        self,
        user: User,
        order_request: OrderRequest,
        db: AsyncSession
    ) -> None:
        """Check risk management limits"""
        
        # Check position size limit
        if order_request.quantity > self.risk_limits["max_position_size"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Order exceeds maximum position size limit"
            )
        
        # Check rate limiting
        rate_key = f"rate_limit:{user.id}:{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        current_count = await self.redis_client.incr(rate_key)
        await self.redis_client.expire(rate_key, 60)  # 1 minute TTL
        
        if current_count > self.risk_limits["max_orders_per_minute"]:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Check daily loss limit
        daily_pnl_key = f"daily_pnl:{user.id}:{datetime.utcnow().strftime('%Y%m%d')}"
        daily_pnl = await self.redis_client.get(daily_pnl_key)
        
        if daily_pnl and Decimal(daily_pnl) < -self.risk_limits["max_daily_loss"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Daily loss limit reached"
            )
    
    async def _handle_trade_execution(self, message: QueueMessage) -> bool:
        """Handle trade execution messages from exchange"""
        try:
            trade_data = message.data
            action = trade_data.get("action")
            
            if action == "order_filled":
                await self._process_order_fill(trade_data)
            elif action == "order_cancelled":
                await self._process_order_cancellation(trade_data)
            elif action == "order_rejected":
                await self._process_order_rejection(trade_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling trade execution: {e}")
            return False
    
    async def _handle_order_updates(self, message: QueueMessage) -> bool:
        """Handle order status updates"""
        try:
            order_data = message.data
            order_id = order_data.get("order_id")
            
            if order_id:
                # Update order in cache
                await self.redis_client.hset(
                    f"orders:{order_id}",
                    mapping={k: str(v) for k, v in order_data.items()}
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling order update: {e}")
            return False
    
    async def _process_order_fill(self, trade_data: Dict[str, Any]) -> None:
        """Process order fill and update positions"""
        
        order_id = trade_data.get("order_id")
        fill_quantity = Decimal(str(trade_data.get("fill_quantity", 0)))
        fill_price = Decimal(str(trade_data.get("fill_price", 0)))
        
        # Update order status
        await self.redis_client.hset(
            f"orders:{order_id}",
            mapping={
                "status": OrderStatus.FILLED,
                "filled_quantity": str(fill_quantity),
                "average_price": str(fill_price),
                "updated_at": datetime.utcnow().isoformat()
            }
        )
        
        # Update position
        await self._update_position(trade_data)
        
        logger.info(f"Order {order_id} filled: {fill_quantity} @ {fill_price}")
    
    async def _process_order_cancellation(self, trade_data: Dict[str, Any]) -> None:
        """Process order cancellation"""
        
        order_id = trade_data.get("order_id")
        
        await self.redis_client.hset(
            f"orders:{order_id}",
            mapping={
                "status": OrderStatus.CANCELLED,
                "updated_at": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Order {order_id} cancelled")
    
    async def _process_order_rejection(self, trade_data: Dict[str, Any]) -> None:
        """Process order rejection"""
        
        order_id = trade_data.get("order_id")
        rejection_reason = trade_data.get("reason", "Unknown")
        
        await self.redis_client.hset(
            f"orders:{order_id}",
            mapping={
                "status": OrderStatus.REJECTED,
                "rejection_reason": rejection_reason,
                "updated_at": datetime.utcnow().isoformat()
            }
        )
        
        logger.warning(f"Order {order_id} rejected: {rejection_reason}")
    
    async def _update_position(self, trade_data: Dict[str, Any]) -> None:
        """Update position based on trade execution"""
        
        user_id = trade_data.get("user_id")
        symbol = trade_data.get("symbol")
        side = trade_data.get("side")
        quantity = Decimal(str(trade_data.get("fill_quantity", 0)))
        price = Decimal(str(trade_data.get("fill_price", 0)))
        
        position_key = f"positions:{user_id}:{symbol}"
        
        # Get current position
        current_position = await self.redis_client.hgetall(position_key)
        
        if current_position:
            # Update existing position
            current_size = Decimal(current_position.get("size", "0"))
            current_entry = Decimal(current_position.get("entry_price", "0"))
            
            if side == OrderSide.BUY:
                new_size = current_size + quantity
            else:
                new_size = current_size - quantity
            
            # Calculate new entry price (weighted average)
            if new_size != 0:
                total_cost = (current_size * current_entry) + (quantity * price)
                new_entry_price = total_cost / abs(new_size)
            else:
                new_entry_price = Decimal('0')
            
            await self.redis_client.hset(
                position_key,
                mapping={
                    "size": str(new_size),
                    "entry_price": str(new_entry_price),
                    "mark_price": str(price),  # Update with latest price
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
        else:
            # Create new position
            position_side = PositionSide.LONG if side == OrderSide.BUY else PositionSide.SHORT
            position_size = quantity if side == OrderSide.BUY else -quantity
            
            await self.redis_client.hset(
                position_key,
                mapping={
                    "symbol": symbol,
                    "side": position_side,
                    "size": str(position_size),
                    "entry_price": str(price),
                    "mark_price": str(price),
                    "unrealized_pnl": "0",
                    "realized_pnl": "0",
                    "margin_used": str(quantity * price * Decimal('0.1')),  # 10% margin
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
            )

# Global service instance
trading_service = TradingEngineService()

# API Routes
@trading_app.post("/trading/orders", response_model=OrderResponse)
async def create_order(
    order_request: OrderRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission("trading")),
    db: AsyncSession = Depends(get_async_session)
):
    """Create a new trading order"""
    
    return await trading_service.create_order(current_user, order_request, db)

@trading_app.delete("/trading/orders/{order_id}")
async def cancel_order(
    order_id: str,
    current_user: User = Depends(require_permission("trading")),
    db: AsyncSession = Depends(get_async_session)
):
    """Cancel an existing order"""
    
    success = await trading_service.cancel_order(current_user, order_id, db)
    
    if success:
        return {"message": "Order cancelled successfully", "order_id": order_id}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to cancel order"
        )

@trading_app.get("/trading/orders", response_model=List[OrderResponse])
async def get_orders(
    symbol: Optional[str] = None,
    status: Optional[OrderStatus] = None,
    limit: int = 100,
    current_user: User = Depends(require_permission("view_portfolio"))
):
    """Get user's orders"""
    
    return await trading_service.get_orders(current_user, symbol, status, limit)

@trading_app.get("/trading/positions", response_model=List[PositionInfo])
async def get_positions(
    symbol: Optional[str] = None,
    current_user: User = Depends(require_permission("view_portfolio"))
):
    """Get user's positions"""
    
    return await trading_service.get_positions(current_user, symbol)

@trading_app.get("/trading/portfolio", response_model=PortfolioSummary)
async def get_portfolio_summary(
    current_user: User = Depends(require_permission("view_portfolio")),
    db: AsyncSession = Depends(get_async_session)
):
    """Get portfolio summary"""
    
    return await trading_service.get_portfolio_summary(current_user, db)

@trading_app.get("/trading/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "trading_engine",
        "timestamp": datetime.utcnow().isoformat()
    }

# Startup event
@trading_app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    await trading_service.initialize()
    logger.info("Trading engine service started successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(trading_app, host="0.0.0.0", port=8002)
