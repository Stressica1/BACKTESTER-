"""
Data Service Microservice
Handles market data ingestion, real-time feeds, and historical data management
"""

import asyncio
import json
import websockets
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from decimal import Decimal
import logging

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from core.database import get_async_session, Exchange
from core.redis_client import get_redis_client
from core.message_queue import get_message_queue_client, QueueMessage
from services.auth_service import get_current_user, require_permission

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app for data service
data_app = FastAPI(
    title="Trading Platform Data Service",
    description="Microservice for market data ingestion and distribution",
    version="1.0.0",
    docs_url="/data/docs",
    redoc_url="/data/redoc"
)

# CORS middleware
data_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class MarketData(BaseModel):
    symbol: str
    timestamp: datetime
    price: Decimal
    volume: Decimal
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    high_24h: Optional[Decimal] = None
    low_24h: Optional[Decimal] = None
    change_24h: Optional[Decimal] = None

class OrderBookLevel(BaseModel):
    price: Decimal
    quantity: Decimal

class OrderBook(BaseModel):
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]

class TradeData(BaseModel):
    symbol: str
    timestamp: datetime
    price: Decimal
    quantity: Decimal
    side: str  # "buy" or "sell"

class Candle(BaseModel):
    symbol: str
    timestamp: datetime
    timeframe: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

class SymbolInfo(BaseModel):
    symbol: str
    base_asset: str
    quote_asset: str
    status: str
    price_precision: int
    quantity_precision: int
    min_order_size: Decimal
    max_order_size: Decimal

class DataSubscription(BaseModel):
    symbol: str
    data_types: List[str]  # ["ticker", "orderbook", "trades", "candles"]

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time data distribution"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # user_id -> set of symbols
        
    async def connect(self, websocket: WebSocket, user_id: str):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.subscriptions[user_id] = set()
        logger.info(f"WebSocket connected for user: {user_id}")
        
    def disconnect(self, user_id: str):
        """Disconnect a WebSocket client"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.subscriptions:
            del self.subscriptions[user_id]
        logger.info(f"WebSocket disconnected for user: {user_id}")
        
    async def send_personal_message(self, message: str, user_id: str):
        """Send message to specific user"""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {e}")
                self.disconnect(user_id)
                
    async def broadcast_market_data(self, symbol: str, data: Dict[str, Any]):
        """Broadcast market data to subscribed users"""
        message = json.dumps({
            "type": "market_data",
            "symbol": symbol,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        for user_id, symbols in self.subscriptions.items():
            if symbol in symbols:
                await self.send_personal_message(message, user_id)
    
    def subscribe(self, user_id: str, symbol: str):
        """Subscribe user to symbol updates"""
        if user_id not in self.subscriptions:
            self.subscriptions[user_id] = set()
        self.subscriptions[user_id].add(symbol)
        
    def unsubscribe(self, user_id: str, symbol: str):
        """Unsubscribe user from symbol updates"""
        if user_id in self.subscriptions:
            self.subscriptions[user_id].discard(symbol)

# Global connection manager
connection_manager = ConnectionManager()

# Data Service class
class DataService:
    """Advanced market data service with real-time feeds and historical data"""
    
    def __init__(self):
        self.redis_client = None
        self.message_queue = None
        self.exchange_feeds: Dict[str, Dict] = {}
        self.supported_exchanges = ["binance", "coinbase", "kraken", "ftx"]
        self.active_symbols: Set[str] = set()
        
    async def initialize(self):
        """Initialize service dependencies"""
        self.redis_client = get_redis_client()
        self.message_queue = get_message_queue_client()
        
        # Start market data feeds
        await self._start_market_data_feeds()
        
    async def _start_market_data_feeds(self):
        """Start real-time market data feeds from exchanges"""
        
        # Start tasks for each exchange
        for exchange in self.supported_exchanges:
            if exchange == "binance":
                asyncio.create_task(self._binance_websocket_feed())
            elif exchange == "coinbase":
                asyncio.create_task(self._coinbase_websocket_feed())
            # Add more exchanges as needed
                
    async def _binance_websocket_feed(self):
        """Connect to Binance WebSocket feed"""
        try:
            uri = "wss://stream.binance.com:9443/ws/!ticker@arr"
            
            async with websockets.connect(uri) as websocket:
                logger.info("Connected to Binance WebSocket feed")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if isinstance(data, list):
                            for ticker in data:
                                await self._process_binance_ticker(ticker)
                        else:
                            await self._process_binance_ticker(data)
                            
                    except Exception as e:
                        logger.error(f"Error processing Binance message: {e}")
                        
        except Exception as e:
            logger.error(f"Binance WebSocket connection error: {e}")
            # Implement reconnection logic
            await asyncio.sleep(5)
            asyncio.create_task(self._binance_websocket_feed())
    
    async def _process_binance_ticker(self, ticker_data: Dict[str, Any]):
        """Process Binance ticker data"""
        try:
            symbol = ticker_data.get("s")  # Symbol
            price = Decimal(ticker_data.get("c", "0"))  # Current price
            volume = Decimal(ticker_data.get("v", "0"))  # 24h volume
            high_24h = Decimal(ticker_data.get("h", "0"))  # 24h high
            low_24h = Decimal(ticker_data.get("l", "0"))  # 24h low
            change_24h = Decimal(ticker_data.get("P", "0"))  # 24h change %
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                price=price,
                volume=volume,
                high_24h=high_24h,
                low_24h=low_24h,
                change_24h=change_24h
            )
            
            # Store in Redis
            await self._store_market_data(market_data)
            
            # Broadcast to WebSocket clients
            await connection_manager.broadcast_market_data(
                symbol,
                market_data.dict()
            )
            
            # Publish to message queue
            await self.message_queue.publish_market_data(
                symbol,
                market_data.dict()
            )
            
        except Exception as e:
            logger.error(f"Error processing Binance ticker: {e}")
    
    async def _coinbase_websocket_feed(self):
        """Connect to Coinbase WebSocket feed"""
        try:
            uri = "wss://ws-feed.pro.coinbase.com"
            
            subscribe_message = {
                "type": "subscribe",
                "product_ids": ["BTC-USD", "ETH-USD", "ADA-USD"],
                "channels": ["ticker"]
            }
            
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps(subscribe_message))
                logger.info("Connected to Coinbase WebSocket feed")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if data.get("type") == "ticker":
                            await self._process_coinbase_ticker(data)
                            
                    except Exception as e:
                        logger.error(f"Error processing Coinbase message: {e}")
                        
        except Exception as e:
            logger.error(f"Coinbase WebSocket connection error: {e}")
            # Implement reconnection logic
            await asyncio.sleep(5)
            asyncio.create_task(self._coinbase_websocket_feed())
    
    async def _process_coinbase_ticker(self, ticker_data: Dict[str, Any]):
        """Process Coinbase ticker data"""
        try:
            symbol = ticker_data.get("product_id", "").replace("-", "")
            price = Decimal(ticker_data.get("price", "0"))
            volume = Decimal(ticker_data.get("volume_24h", "0"))
            high_24h = Decimal(ticker_data.get("high_24h", "0"))
            low_24h = Decimal(ticker_data.get("low_24h", "0"))
            bid = Decimal(ticker_data.get("best_bid", "0"))
            ask = Decimal(ticker_data.get("best_ask", "0"))
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                price=price,
                volume=volume,
                bid=bid,
                ask=ask,
                high_24h=high_24h,
                low_24h=low_24h
            )
            
            # Store in Redis
            await self._store_market_data(market_data)
            
            # Broadcast to WebSocket clients
            await connection_manager.broadcast_market_data(
                symbol,
                market_data.dict()
            )
            
            # Publish to message queue
            await self.message_queue.publish_market_data(
                symbol,
                market_data.dict()
            )
            
        except Exception as e:
            logger.error(f"Error processing Coinbase ticker: {e}")
    
    async def _store_market_data(self, market_data: MarketData):
        """Store market data in Redis with expiration"""
        
        # Store latest price
        await self.redis_client.hset(
            f"market:{market_data.symbol}",
            mapping={
                "price": str(market_data.price),
                "volume": str(market_data.volume),
                "timestamp": market_data.timestamp.isoformat(),
                "bid": str(market_data.bid) if market_data.bid else "",
                "ask": str(market_data.ask) if market_data.ask else "",
                "high_24h": str(market_data.high_24h) if market_data.high_24h else "",
                "low_24h": str(market_data.low_24h) if market_data.low_24h else "",
                "change_24h": str(market_data.change_24h) if market_data.change_24h else ""
            }
        )
        
        # Set expiration
        await self.redis_client.expire(f"market:{market_data.symbol}", 3600)  # 1 hour
        
        # Store in time series for charts
        timestamp = int(market_data.timestamp.timestamp())
        await self.redis_client.zadd(
            f"prices:{market_data.symbol}",
            {str(market_data.price): timestamp}
        )
        
        # Keep only last 1000 prices
        await self.redis_client.zremrangebyrank(f"prices:{market_data.symbol}", 0, -1001)
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for symbol"""
        
        data = await self.redis_client.hgetall(f"market:{symbol}")
        
        if not data:
            return None
        
        return MarketData(
            symbol=symbol,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            price=Decimal(data["price"]),
            volume=Decimal(data["volume"]),
            bid=Decimal(data["bid"]) if data.get("bid") else None,
            ask=Decimal(data["ask"]) if data.get("ask") else None,
            high_24h=Decimal(data["high_24h"]) if data.get("high_24h") else None,
            low_24h=Decimal(data["low_24h"]) if data.get("low_24h") else None,
            change_24h=Decimal(data["change_24h"]) if data.get("change_24h") else None
        )
    
    async def get_price_history(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100
    ) -> List[Candle]:
        """Get historical price data"""
        
        # Get price data from Redis time series
        prices = await self.redis_client.zrevrange(
            f"prices:{symbol}",
            0,
            limit - 1,
            withscores=True
        )
        
        candles = []
        for price_str, timestamp in prices:
            price = Decimal(price_str)
            dt = datetime.fromtimestamp(timestamp)
            
            # For simplicity, create OHLC from price (in real implementation, 
            # you'd aggregate multiple data points)
            candle = Candle(
                symbol=symbol,
                timestamp=dt,
                timeframe=timeframe,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=Decimal("0")  # Would need separate volume tracking
            )
            candles.append(candle)
        
        return candles
    
    async def get_supported_symbols(self, db: AsyncSession) -> List[SymbolInfo]:
        """Get list of supported trading symbols"""
        
        result = await db.execute(select(Exchange))
        exchanges = result.scalars().all()
        
        symbols = []
        for exchange in exchanges:
            symbol_info = SymbolInfo(
                symbol=exchange.symbol,
                base_asset=exchange.base_asset,
                quote_asset=exchange.quote_asset,
                status="TRADING",
                price_precision=exchange.price_precision,
                quantity_precision=exchange.quantity_precision,
                min_order_size=exchange.min_order_size,
                max_order_size=exchange.max_order_size
            )
            symbols.append(symbol_info)
        
        return symbols
    
    async def subscribe_to_symbol(self, user_id: str, symbol: str):
        """Subscribe user to real-time updates for symbol"""
        connection_manager.subscribe(user_id, symbol)
        self.active_symbols.add(symbol)
        
        # Get current market data and send immediately
        market_data = await self.get_market_data(symbol)
        if market_data:
            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "market_data",
                    "symbol": symbol,
                    "data": market_data.dict(),
                    "timestamp": datetime.utcnow().isoformat()
                }),
                user_id
            )

# Global service instance
data_service = DataService()

# API Routes
@data_app.get("/data/market/{symbol}", response_model=MarketData)
async def get_market_data(
    symbol: str,
    current_user = Depends(require_permission("view_market_data"))
):
    """Get latest market data for symbol"""
    
    market_data = await data_service.get_market_data(symbol.upper())
    
    if not market_data:
        raise HTTPException(
            status_code=404,
            detail=f"Market data not found for symbol: {symbol}"
        )
    
    return market_data

@data_app.get("/data/history/{symbol}", response_model=List[Candle])
async def get_price_history(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    current_user = Depends(require_permission("view_market_data"))
):
    """Get historical price data"""
    
    return await data_service.get_price_history(symbol.upper(), timeframe, limit)

@data_app.get("/data/symbols", response_model=List[SymbolInfo])
async def get_supported_symbols(
    db: AsyncSession = Depends(get_async_session),
    current_user = Depends(require_permission("view_market_data"))
):
    """Get list of supported trading symbols"""
    
    return await data_service.get_supported_symbols(db)

@data_app.websocket("/data/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time market data"""
    
    await connection_manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive subscription messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                symbol = message.get("symbol", "").upper()
                await data_service.subscribe_to_symbol(user_id, symbol)
                
                await websocket.send_text(json.dumps({
                    "type": "subscription_confirmed",
                    "symbol": symbol,
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
            elif message.get("type") == "unsubscribe":
                symbol = message.get("symbol", "").upper()
                connection_manager.unsubscribe(user_id, symbol)
                
                await websocket.send_text(json.dumps({
                    "type": "unsubscription_confirmed",
                    "symbol": symbol,
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
    except WebSocketDisconnect:
        connection_manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        connection_manager.disconnect(user_id)

@data_app.get("/data/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "data_service",
        "active_connections": len(connection_manager.active_connections),
        "active_symbols": len(data_service.active_symbols),
        "timestamp": datetime.utcnow().isoformat()
    }

# Startup event
@data_app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    await data_service.initialize()
    logger.info("Data service started successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(data_app, host="0.0.0.0", port=8003)
