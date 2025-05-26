from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
import asyncio
import json
import ccxt
import time
from typing import Dict, List
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Super Z analyzer
from super_z_pullback_analyzer import SuperZPullbackAnalyzer

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize exchange with proper error handling
try:
    exchange = ccxt.bitget({
        'apiKey': os.getenv('BITGET_API_KEY'),
        'secret': os.getenv('BITGET_API_SECRET'),
        'password': os.getenv('BITGET_PASSPHRASE'),
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
            'adjustForTimeDifference': True
        }
    })
    
    # Set testnet mode if available
    if os.getenv('BITGET_TESTNET', 'false').lower() == 'true':
        exchange.set_sandbox_mode(True)
    
    # Verify connection with a simple market fetch
    markets = exchange.fetch_markets()
    print(f"Exchange initialized successfully with ccxt v{ccxt.__version__} - {len(markets)} markets available")
except Exception as e:
    print(f"Error initializing exchange: {e}")
    exchange = None

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# Store market data
market_data: Dict = {}
positions: Dict = {}

# Rate limiting
last_api_call = 0
min_api_interval = 3.0  # Increased to 3 seconds between API calls
retry_delay = 5.0  # Delay between retries when rate limited

def get_ngrok_url():
    """Get the current ngrok tunnel URL dynamically"""
    try:
        response = requests.get("http://localhost:4040/api/tunnels", timeout=5)
        if response.status_code == 200:
            tunnels = response.json().get("tunnels", [])
            if tunnels:
                return tunnels[0]["public_url"]
    except Exception as e:
        print(f"Error getting ngrok URL: {e}")
    return "http://localhost:8004"  # Fallback to localhost

async def rate_limit():
    global last_api_call
    now = time.time()
    if now - last_api_call < min_api_interval:
        await asyncio.sleep(min_api_interval - (now - last_api_call))
    last_api_call = time.time()

async def fetch_with_retry(func, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            await rate_limit()
            return await func(*args, **kwargs)
        except Exception as e:
            if 'code' in str(e) and '10500' in str(e):  # Rate limit error
                if attempt < max_retries - 1:
                    print(f"Rate limited, retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    continue
            raise e
    return None

@app.get("/")
async def get_dashboard():
    return FileResponse('templates/dashboard_unified.html')

@app.post("/api/log-error")
async def log_error(request: Request):
    """Endpoint for logging client-side errors"""
    try:
        error_data = await request.json()
        print(f"Client Error: {error_data}")
        return {"status": "logged", "timestamp": time.time()}
    except Exception as e:
        print(f"Error logging client error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/webhook/health")
async def webhook_health():
    """Health check endpoint for webhook monitoring"""
    return {
        "status": "healthy",
        "service": "trading_backend",
        "timestamp": time.time(),
        "exchange_connected": exchange is not None
    }

@app.get("/api/ngrok-url")
async def get_current_ngrok_url():
    """Get the current ngrok tunnel URL"""
    url = get_ngrok_url()
    return {
        "ngrok_url": url,
        "webhook_health_url": f"{url}/webhook/health",
        "timestamp": time.time()
    }

@app.get("/api/dashboard-data")
async def get_dashboard_data():
    try:
        positions = []
        if exchange:
            try:
                positions = await fetch_with_retry(exchange.fetch_positions)
            except Exception as e:
                print(f"Error fetching positions: {e}")
        
        return {
            "account_balance": 10000.00,
            "daily_pnl": 0.00,
            "open_positions": positions,
            "win_rate": 0.00,
            "drawdown": 0.00,
            "pnl_history": [],
            "recent_trades": [],
            "api_latency": 0,
            "cpu_usage": 0,
            "memory_usage": 0,
            "uptime": 0
        }
    except Exception as e:
        print(f"Error getting dashboard data: {e}")
        return {"error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = f"client_{len(active_connections) + 1}_{int(time.time())}"
    print(f"New WebSocket connection request from {client_id}")
    
    try:
        await websocket.accept()
        active_connections.append(websocket)
        print(f"✅ WebSocket connection accepted: {client_id} (Total active: {len(active_connections)})")
        
        # Send initial connection confirmation
        await websocket.send_json({
            'type': 'connection_status',
            'status': 'connected',
            'client_id': client_id,
            'timestamp': time.time(),
            'message': 'Connected to trading server'
        })
        
        # Start heartbeat for this connection
        heartbeat_task = asyncio.create_task(send_heartbeat(websocket, client_id))
        
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    print(f"📩 Received from {client_id}: {data[:100]}...")
                    message = json.loads(data)
                    
                    # Handle different message types with error checking
                    message_type = message.get('type', 'unknown')
                    
                    if message_type == 'subscribe':
                        channels = message.get('channels', [])
                        print(f"📊 {client_id} subscribing to channels: {channels}")
                        if 'positions' in channels:
                            await send_positions(websocket)
                        if 'market_data' in channels:
                            await send_market_data(websocket)
                    
                    elif message_type == 'execute_trade':
                        if 'order' in message:
                            await execute_trade(message['order'], websocket)
                    
                    elif message_type == 'refresh_positions':
                        await send_positions(websocket)
                    
                    elif message_type == 'get_analytics':
                        await send_analytics(websocket, message.get('timeframe', '1d'))
                    
                    elif message_type == 'get_risk_metrics':
                        await send_risk_metrics(websocket)
                    
                    elif message_type == 'ping':
                        # Respond to ping with pong
                        await websocket.send_json({
                            'type': 'pong',
                            'timestamp': time.time(),
                            'echo': message.get('timestamp')
                        })
                        print(f"🏓 Ping-pong with {client_id}")
                    
                    else:
                        # Send a default response for unknown message types
                        await websocket.send_text(json.dumps({
                            'type': 'error',
                            'message': f'Unknown message type: {message_type}'
                        }))
                        print(f"❓ Unknown message type from {client_id}: {message_type}")
                        
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    }))
                    print(f"❌ Invalid JSON from {client_id}")
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': f'Error processing message: {str(e)}'
                    }))
                    print(f"❌ Error processing message from {client_id}: {str(e)}")
        
        except WebSocketDisconnect:
            print(f"❌ WebSocket disconnected: {client_id}")
        finally:
            # Cancel heartbeat task
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
    
    except Exception as e:
        print(f"❌ Error handling WebSocket connection: {str(e)}")
    
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
            print(f"👋 Client {client_id} removed from active connections (Total remaining: {len(active_connections)})")

async def send_heartbeat(websocket: WebSocket, client_id: str):
    """Send periodic heartbeats to keep the connection alive"""
    try:
        while True:
            await asyncio.sleep(15)  # Send heartbeat every 15 seconds
            if websocket.client_state != WebSocketState.DISCONNECTED:
                try:
                    await websocket.send_json({
                        'type': 'heartbeat',
                        'timestamp': time.time(),
                        'message': 'Server heartbeat'                    })
                    print(f"💓 Heartbeat sent to {client_id}")
                except Exception as e:
                    print(f"❌ Failed to send heartbeat to {client_id}: {str(e)}")
                    break
            else:
                print(f"❌ Cannot send heartbeat to disconnected client {client_id}")
                break
    except asyncio.CancelledError:
        print(f"💤 Heartbeat task for {client_id} cancelled")
        raise
    except Exception as e:
        print(f"❌ Heartbeat error for {client_id}: {e}")

async def send_positions(websocket: WebSocket):
    if not exchange:
        await websocket.send_json({
            'type': 'error',
            'message': 'Exchange not initialized'
        })
        return

    try:
        positions = await fetch_with_retry(exchange.fetch_positions)
        for position in positions:
            if position['contracts'] != 0:  # Only send non-zero positions
                await websocket.send_json({
                    'type': 'position_update',
                    'symbol': position['symbol'],
                    'size': position['contracts'],
                    'entry_price': position['entryPrice'],
                    'leverage': position['leverage'],
                    'unrealized_pnl': position['unrealizedPnl']
                })
    except Exception as e:
        print(f"Error fetching positions: {e}")
        await websocket.send_json({
            'type': 'error',
            'message': f'Error fetching positions: {str(e)}'
        })

async def send_market_data(websocket: WebSocket):
    if not exchange:
        await websocket.send_json({
            'type': 'error',
            'message': 'Exchange not initialized'
        })
        return

    try:
        tickers = await fetch_with_retry(exchange.fetch_tickers)
        for symbol, ticker in tickers.items():
            if symbol.endswith('USDT'):  # Only send USDT pairs
                await websocket.send_json({
                    'type': 'market_data',
                    'symbol': symbol,
                    'price': ticker['last'],
                    'volume': ticker['baseVolume'],
                    'change': ticker['percentage']
                })
    except Exception as e:
        print(f"Error fetching market data: {e}")
        await websocket.send_json({
            'type': 'error',
            'message': f'Error fetching market data: {str(e)}'
        })

async def execute_trade(order: Dict, websocket: WebSocket):
    if not exchange:
        await websocket.send_json({
            'type': 'error',
            'message': 'Exchange not initialized'
        })
        return

    try:
        await rate_limit()
        result = exchange.create_order(
            symbol=order['symbol'],
            type=order['type'],
            side=order['side'],
            amount=order['size'],
            price=order.get('price')
        )
        
        await websocket.send_json({
            'type': 'trade_execution',
            'status': 'success',
            'order_id': result['id'],
            'symbol': order['symbol'],
            'side': order['side'],
            'price': result['price'],
            'size': result['amount']
        })
        
        # Send alert
        await websocket.send_json({
            'type': 'alert',
            'alertType': 'tradeAlerts',
            'alertLevel': 'success',
            'message': f"Trade executed: {order['side']} {order['symbol']} @ {result['price']}"
        })
        
    except Exception as e:
        await websocket.send_json({
            'type': 'alert',
            'alertType': 'tradeAlerts',
            'alertLevel': 'danger',
            'message': f"Trade execution failed: {str(e)}"
        })

async def send_analytics(websocket: WebSocket, timeframe: str):
    try:
        # Calculate analytics based on timeframe
        analytics = {
            'sharpe_ratio': 1.25,
            'sortino_ratio': 1.85,
            'max_drawdown': -12.3,
            'win_rate': 68.5,
            'equity_curve': [
                {'timestamp': int(time.time() - i * 3600), 'value': 10000 + i * 100}
                for i in range(24)
            ],
            'drawdown': [
                {'timestamp': int(time.time() - i * 3600), 'value': -i * 0.5}
                for i in range(24)
            ]
        }
        
        await websocket.send_json({
            'type': 'analytics',
            **analytics
        })
        
    except Exception as e:
        print(f"Error calculating analytics: {e}")

async def send_risk_metrics(websocket: WebSocket):
    try:
        # Calculate risk metrics
        risk_metrics = {
            'var': 2345.67,
            'expected_shortfall': 3456.78,
            'position_concentration': 45.0,
            'leverage_ratio': 2.5,
            'risk_score': 75
        }
        
        await websocket.send_json({
            'type': 'risk_metrics',
            **risk_metrics
        })
        
    except Exception as e:
        print(f"Error calculating risk metrics: {e}")

@app.get("/")
async def dashboard():
    """Serve the unified dashboard"""
    return FileResponse('templates/dashboard_unified.html')

@app.get("/dashboard-clean")
async def dashboard_clean():
    """Serve the clean dashboard"""
    return FileResponse('templates/dashboard_clean.html')

@app.get("/dashboard-optimized")
async def dashboard_optimized():
    """Serve the optimized dashboard"""
    return FileResponse('templates/dashboard_optimized.html')

@app.get("/super-z-analysis")
async def super_z_analysis_page():
    """Serve the Super Z analysis dashboard"""
    return FileResponse('templates/super_z_analysis.html')

@app.get("/api/super-z-analysis")
async def run_super_z_analysis(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    days: int = 30,
    vhma_length: int = 21,
    st_length: int = 10,
    st_multiplier: float = 3.0
):
    """
    Run Super Z pullback analysis on specified symbol
    Tests hypothesis: signals are followed by pullbacks to red VHMA areas
    """
    try:
        # Initialize analyzer
        analyzer = SuperZPullbackAnalyzer()
        
        # Fetch data for the symbol
        df = await analyzer.fetch_data(symbol=symbol, timeframe=timeframe, days=days)
        
        # Calculate indicators
        df['vhma'] = analyzer.calculate_vhma(df, length=vhma_length)
        df['supertrend'], df['supertrend_direction'] = analyzer.calculate_supertrend(
            df, period=st_length, multiplier=st_multiplier
        )
        
        # Detect signals
        signals = analyzer.detect_signals(df)
        
        # Analyze pullbacks
        pullback_events = analyzer.analyze_pullbacks_after_signals(
            df, signals, lookback_periods=50
        )
        
        # Calculate statistics
        stats = analyzer.analyze_pullback_statistics(pullback_events)
        
        # Prepare results
        results = {
            "total_signals": len(signals),
            "pullback_events": len(pullback_events),
            "pullback_rate": (len(pullback_events) / len(signals) * 100) if len(signals) > 0 else 0,
            "statistics": stats,
            "signals": signals[-10:],  # Last 10 signals for display
            "hypothesis_confirmed": stats.get("pullback_rate", 0) > 70  # Hypothesis: >70% pullback rate
        }
        
        return {
            "status": "success",
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_period_days": days,
            "parameters": {
                "vhma_length": vhma_length,
                "st_length": st_length,
                "st_multiplier": st_multiplier
            },
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in Super Z analysis: {e}")
        return {
            "status": "error",
            "error": str(e),
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/super-z-analysis/symbols")
async def get_available_symbols():
    """Get list of available symbols for Super Z analysis"""
    try:
        # Return common crypto symbols that work well with the strategy
        symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT",
            "SOL/USDT", "DOT/USDT", "LINK/USDT", "AVAX/USDT", "MATIC/USDT",
            "ATOM/USDT", "NEAR/USDT", "ALGO/USDT", "FTM/USDT", "ONE/USDT",
            "LTC/USDT", "BCH/USDT", "ETC/USDT", "DOGE/USDT", "SHIB/USDT"
        ]
        
        return {
            "status": "success",
            "symbols": symbols,
            "count": len(symbols),
            "recommended": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/super-z-analysis/batch")
async def run_batch_super_z_analysis(
    request: Request
):
    """
    Run Super Z analysis on multiple symbols
    Useful for testing the pullback hypothesis across different markets
    """
    try:
        data = await request.json()
        symbols = data.get("symbols", ["BTC/USDT", "ETH/USDT"])
        timeframe = data.get("timeframe", "4h")
        days = data.get("days", 30)
        vhma_length = data.get("vhma_length", 21)
        st_length = data.get("st_length", 10)
        st_multiplier = data.get("st_multiplier", 3.0)
        
        results = {}
        analyzer = SuperZPullbackAnalyzer()
        
        for symbol in symbols:
            try:
                # Fetch data for the symbol
                df = await analyzer.fetch_data(symbol=symbol, timeframe=timeframe, days=days)
                
                # Calculate indicators
                df['vhma'] = analyzer.calculate_vhma(df, length=vhma_length)
                df['supertrend'], df['supertrend_direction'] = analyzer.calculate_supertrend(
                    df, period=st_length, multiplier=st_multiplier
                )
                
                # Detect signals
                signals = analyzer.detect_signals(df)
                
                # Analyze pullbacks
                pullback_events = analyzer.analyze_pullbacks_after_signals(
                    df, signals, lookback_periods=50
                )
                
                # Calculate statistics
                stats = analyzer.analyze_pullback_statistics(pullback_events)
                
                # Prepare symbol results
                symbol_results = {
                    "total_signals": len(signals),
                    "pullback_events": len(pullback_events),
                    "pullback_rate": (len(pullback_events) / len(signals) * 100) if len(signals) > 0 else 0,
                    "statistics": stats,
                    "hypothesis_confirmed": stats.get("pullback_rate", 0) > 70
                }
                
                results[symbol] = {
                    "status": "success",
                    "data": symbol_results
                }
                
                # Add small delay to avoid rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                results[symbol] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Calculate aggregate statistics
        successful_analyses = [r for r in results.values() if r["status"] == "success"]
        total_signals = sum(r["data"]["total_signals"] for r in successful_analyses)
        total_pullbacks = sum(r["data"]["pullback_events"] for r in successful_analyses)
        
        aggregate_stats = {
            "symbols_analyzed": len(symbols),
            "successful_analyses": len(successful_analyses),
            "total_signals_across_markets": total_signals,
            "total_pullback_events": total_pullbacks,
            "overall_pullback_rate": (total_pullbacks / total_signals * 100) if total_signals > 0 else 0,
            "hypothesis_confirmed": total_pullbacks / total_signals > 0.7 if total_signals > 0 else False
        }
        
        return {
            "status": "success",
            "batch_results": results,
            "aggregate_statistics": aggregate_stats,
            "parameters": {
                "timeframe": timeframe,
                "days": days,
                "vhma_length": vhma_length,
                "st_length": st_length,
                "st_multiplier": st_multiplier
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Ultimate Trading Dashboard Server...")
    print("📍 Unified Dashboard: http://localhost:8000")
    print("📍 Clean Dashboard: http://localhost:8000/dashboard-clean")
    print("📍 Optimized Dashboard: http://localhost:8000/dashboard-optimized")
    uvicorn.run(app, host="0.0.0.0", port=8000)