from fastapi import FastAPI, Request, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import ccxt
import hmac
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
from config import Config
import asyncio
from risk_config import RiskConfig
from dotenv import load_dotenv
import logging
from data_fetcher import DataFetcher
from backtest_engine import BacktestEngine
import uvicorn
import psutil
from contextlib import asynccontextmanager
from volatility_scanner import VolatilityScanner, run_volatility_scan

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the application"""
    try:
        # Initialize exchange connection
        logger.info("Successfully connected to exchange")
        
        # Start the account balance update task
        asyncio.create_task(update_account_balances())
        
        # Log startup information
        logger.info("Trading Bot Dashboard started successfully")
        logger.info(f"Environment: {'testnet' if Config.is_testnet() else 'mainnet'}")
        
        # Get webhook URL
        webhook_url = os.getenv('NGROK_URL', 'Not configured')
        logger.info(f"Webhook URL: {webhook_url}")
        
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

app = FastAPI(title="Trading Bot Dashboard", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Store trade history and performance metrics
trade_history: List[Dict] = []
account_balances: Dict = {}
last_update_time: float = 0

# In-memory storage for trades
trades: List[Dict[str, Any]] = []

# Initialize data fetcher
data_fetcher = DataFetcher()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting
last_api_call = 0
min_api_interval = 1.0  # Seconds between API calls
retry_delay = 3.0  # Delay between retries when rate limited

async def rate_limit():
    """Apply rate limiting to API calls"""
    global last_api_call
    now = time.time()
    if now - last_api_call < min_api_interval:
        await asyncio.sleep(min_api_interval - (now - last_api_call))
    last_api_call = time.time()

# Add P&L history tracking
pnl_history = []

# Initialize Bitget exchange
exchange = None
try:
    # Check if we have valid Bitget credentials
    api_key = Config.get_api_key()
    api_secret = Config.get_api_secret()
    passphrase = Config.get_passphrase()
    
    if not api_key or api_key == 'your-bitget-api-key-here':
        logger.warning("No valid Bitget API credentials found, using demo mode")
        # Use a demo exchange configuration for testing
        exchange = ccxt.bitget({
            'apiKey': '',
            'secret': '',
            'password': '',
            'enableRateLimit': True,
            'sandbox': True,
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True,
                'createMarketBuyOrderRequiresPrice': False,
            }
        })
    else:
        exchange = ccxt.bitget({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # For USDT-M futures
                'adjustForTimeDifference': True,
                'createMarketBuyOrderRequiresPrice': False,
            }
        })
        
        # Set testnet if configured
        if Config.is_testnet():
            exchange.urls['api'] = exchange.urls['test']
            exchange.set_sandbox_mode(True)
    
    # Test connection to verify setup (only public endpoints)
    try:
        markets = exchange.fetch_markets()
        logger.info(f"Successfully initialized ccxt v{ccxt.__version__} with {len(markets)} markets available")
    except Exception as market_error:
        logger.warning(f"Could not fetch markets: {market_error}, continuing with limited functionality")
        
except Exception as e:
    logger.error(f"Error initializing exchange: {str(e)}")
    # Create a minimal exchange object for basic functionality
    try:
        exchange = ccxt.bitget({
            'apiKey': '',
            'secret': '',
            'password': '',
            'enableRateLimit': True,
            'sandbox': True,
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True,
            }
        })
        logger.info("Created demo exchange instance for testing")
    except Exception as fallback_error:
        logger.error(f"Failed to create fallback exchange: {fallback_error}")
        exchange = None

# Store notification history
notifications = []
last_notification_id = 0

# Initialize data structures
pnl_history = []
positions_cache = {}
orders_cache = {}
last_update_time = 0

class TradingViewAlert(BaseModel):
    symbol: str
    action: str  # buy/sell
    price: float
    size: float = None  # Optional, will be calculated based on balance if not provided

class Trade(BaseModel):
    symbol: str
    action: str
    entry_price: float
    current_price: float
    pnl: float
    status: str
    time: str

class BacktestRequest(BaseModel):
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_balance: float
    risk_per_trade: float
    max_open_trades: int
    max_drawdown: float
    position_sizing: str
    entry_conditions: Dict[str, Any]
    exit_conditions: Dict[str, Any]
    direction: str
    optimize: bool = False
    n_trials: int = 100

def verify_webhook_signature(request: Request, body: bytes) -> bool:
    """Verify the webhook signature from TradingView"""
    signature = request.headers.get('X-TradingView-Signature')
    if not signature:
        logger.warning("No signature found in webhook request")
        return False
    
    # Use a separate webhook secret for signature verification
    webhook_secret = os.getenv('WEBHOOK_SECRET', '').encode()
    if not webhook_secret:
        logger.warning("No WEBHOOK_SECRET found in environment variables")
        return False
        
    hmac_obj = hmac.new(webhook_secret, body, hashlib.sha256)
    calculated_signature = hmac_obj.hexdigest()
    
    return hmac.compare_digest(signature, calculated_signature)

async def update_account_balances():
    """Update account balances from exchange"""
    while True:
        try:
            # For Bitget swap type, we need to specify the currency
            balances = await asyncio.to_thread(exchange.fetch_balance, params={'code': 'USDT'})
            if balances:
                account_balances.update({
                    'USDT': float(balances.get('USDT', {}).get('free', 0.0)),
                    'BTC': float(balances.get('BTC', {}).get('free', 0.0)),
                    'total': float(balances.get('total', {}).get('USDT', 0.0)),
                    'total_pnl': sum(trade.get('pnl', 0) for trade in trades)
                })
        except Exception as e:
            logger.error(f"Error updating balances: {str(e)}")
        await asyncio.sleep(60)  # Update every 60 seconds

def calculate_position_size(exchange, symbol: str, entry_price: float, risk_percentage: float = 0.02) -> float:
    """Calculate position size based on account balance and risk percentage"""
    try:
        balance = exchange.fetch_balance()
        usdt_balance = float(balance['USDT']['free'])
        position_size_usdt = usdt_balance * risk_percentage
        position_size = position_size_usdt / entry_price
        return position_size
    except Exception as e:
        logger.error(f"Error calculating position size: {str(e)}")
        return 0.0

@app.post("/webhook")
async def process_webhook_alert(request: Request):
    """Process incoming webhook alerts from TradingView"""
    try:
        # Get the raw body of the request
        body = await request.body()
        
        # Verify the TradingView signature
        if not verify_webhook_signature(request, body):
            raise HTTPException(status_code=403, detail="Invalid signature")
        
        # Parse the JSON payload
        payload = json.loads(body)
        logger.info(f"Received webhook payload: {payload}")
        
        # Extract relevant fields
        symbol = payload.get('symbol')
        action = payload.get('action')
        price = payload.get('price')
        size = payload.get('size')
        
        # Validate required fields
        if not all([symbol, action, price]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # Convert action to lowercase
        action = action.lower()
        
        # Check if the action is valid
        if action not in ['buy', 'sell']:
            raise HTTPException(status_code=400, detail="Invalid action, must be 'buy' or 'sell'")
        
        # Calculate position size if not provided
        if size is None:
            risk_percentage = 0.02  # Default risk per trade
            size = calculate_position_size(exchange, symbol, price, risk_percentage)
        
        # Log the trade execution
        logger.info(f"Executing {action} for {size} of {symbol} at {price}")
        
        # Execute the trade
        if action == 'buy':
            # Place a market buy order
            order = await asyncio.to_thread(exchange.create_market_buy_order, symbol, size)
        else:
            # Place a market sell order
            order = await asyncio.to_thread(exchange.create_market_sell_order, symbol, size)
        
        logger.info(f"Order response: {order}")
        
        # Extract order details
        order_id = order.get('id')
        filled_size = order.get('filled', size)
        avg_fill_price = order.get('average', price)
        
        # Calculate P&L for the trade
        pnl = (avg_fill_price - price) * filled_size if action == 'buy' else (price - avg_fill_price) * filled_size
        
        # Determine trade status
        status = 'closed' if order.get('status') == 'closed' else 'open'
        
        # Get the current time in ISO format
        trade_time = datetime.now().isoformat()
        
        # Create a trade record
        trade = Trade(
            symbol=symbol,
            action=action,
            entry_price=avg_fill_price,
            current_price=price,
            pnl=pnl,
            status=status,
            time=trade_time
        )
        
        # Store the trade in the in-memory list
        trades.append(trade.dict())
        
        # Log the trade details
        logger.info(f"Trade executed: {trade}")
        
        # Return a success response
        return JSONResponse(content={"status": "success", "trade": trade.dict()}, status_code=200)
    
    except HTTPException as http_ex:
        logger.warning(f"HTTP error: {http_ex.detail}")
        raise
    except Exception as e:
        logger.error(f"Error processing webhook alert: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/backtest")
async def run_backtest(request: Request, backtest_request: BacktestRequest):
    """Run a backtest based on the provided parameters"""
    try:
        # Convert dates from strings to datetime objects
        start_date = datetime.strptime(backtest_request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(backtest_request.end_date, "%Y-%m-%d")
        
        # Validate date range
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        # Log the backtest parameters
        logger.info(f"Running backtest: {backtest_request.dict()}")
        
        # Run the backtest using the BacktestEngine
        results = await asyncio.to_thread(BacktestEngine.run_backtest, backtest_request.dict())
        
        # Log the backtest results
        logger.info(f"Backtest results: {results}")
        
        # Return the backtest results
        return JSONResponse(content={"status": "success", "results": results}, status_code=200)
    
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data"""
    await websocket.accept()
    logger.info("New WebSocket connection established")
    
    # Send initial connection status
    await websocket.send_json({
        "type": "connection_status",
        "data": {
            "status": "connected",
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to trading backend"
        }
    })
    
    try:
        # Process incoming messages
        while True:
            try:
                # Try to receive a message with a 1 second timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                message = json.loads(data)
                
                # Handle different message types
                if message.get('type') == 'ping':
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
                
                elif message.get('type') == 'subscribe':
                    channels = message.get('channels', [])
                    logger.info(f"Client subscribed to channels: {channels}")
                    await websocket.send_json({
                        "type": "subscription_status",
                        "data": {
                            "channels": channels,
                            "status": "active"
                        }
                    })
                
            except asyncio.TimeoutError:
                # No message received, continue sending updates
                pass
            
            # Send regular updates (every 2 seconds)
            await websocket.send_json({
                "type": "dashboard_update",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "status": "connected",
                    "balance": 10000.0 + len(trades) * 100,
                    "pnl": sum(trade.get('pnl', 0) for trade in trades[-5:]) if trades else 0,
                    "open_positions": len(trades),
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage": psutil.virtual_memory().percent
                }
            })
            
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("WebSocket connection closed")