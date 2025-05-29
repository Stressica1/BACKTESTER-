import json
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import websockets
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning) 
import concurrent.futures
from tqdm import tqdm
import prettytable
import multiprocessing
import sys
import glob
import psutil
import itertools
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from ratelimit import limits, sleep_and_retry
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import importlib

# Bitget API Rate Limiting
try:
    from aiolimiter import AsyncLimiter
except ImportError:
    print("Installing aiolimiter for proper rate limiting...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiolimiter"])
    from aiolimiter import AsyncLimiter

# Import rate monitoring
try:
    from bitget_rate_monitor import record_api_request, get_rate_status, generate_rate_report
except ImportError:
    # Create dummy functions if monitor not available
    def record_api_request(*args, **kwargs): pass
    def get_rate_status(): return {}
    def generate_rate_report(*args, **kwargs): return "Rate monitoring not available"

# Configure logging with unified error logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_unified.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SuperTrend")

# Rate limit error logger
rate_limit_logger = logging.getLogger("BitgetRateLimit")
rate_limit_handler = logging.FileHandler('bitget_rate_limits.log')
rate_limit_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
rate_limit_logger.addHandler(rate_limit_handler)
rate_limit_logger.setLevel(logging.INFO)

# 24-bit color helpers
RESET = '\033[0m'
BOLD = '\033[1m'
HEADER_BG = '\033[48;2;40;40;80m'
HEADER_FG = '\033[38;2;255;255;0m'
BUY_FG = '\033[38;2;0;255;0m'
SELL_FG = '\033[38;2;255;0;0m'
HOLD_FG = '\033[38;2;200;200;200m'
TABLE_BG = '\033[48;2;30;30;40m'
TABLE_FG = '\033[38;2;180;220;255m'
ACCENT = '\033[38;2;255;0;255m'

class BitgetRateLimiter:
    """Comprehensive Bitget API rate limiting system"""
    
    def __init__(self):
        # Public endpoints rate limiters (20 req/s)
        self.public_limiters = {
            'time': AsyncLimiter(20, 1),
            'currencies': AsyncLimiter(20, 1),
            'products': AsyncLimiter(20, 1),
            'product': AsyncLimiter(20, 1),
            'ticker': AsyncLimiter(20, 1),
            'tickers': AsyncLimiter(20, 1),
            'fills': AsyncLimiter(20, 1),
            'candles': AsyncLimiter(20, 1),
            'depth': AsyncLimiter(20, 1),
            'transferRecords': AsyncLimiter(20, 1),
            'orderInfo': AsyncLimiter(20, 1),
            'open-orders': AsyncLimiter(20, 1),
            'history': AsyncLimiter(20, 1),
            'trade_fills': AsyncLimiter(20, 1)
        }
        
        # Private endpoints rate limiters (10 req/s for most)
        self.private_limiters = {
            'assets': AsyncLimiter(10, 1),
            'bills': AsyncLimiter(10, 1),
            'orders': AsyncLimiter(10, 1),
            'cancel-order': AsyncLimiter(10, 1),
            'cancel-batch-orders': AsyncLimiter(10, 1),
            'batch-orders': AsyncLimiter(5, 1)  # 5 req/s for batch orders
        }
        
        # Global rate limiter for safety
        self.global_limiter = AsyncLimiter(50, 1)  # Conservative global limit
        
        # WebSocket limits
        self.ws_connections = 0
        self.max_ws_connections = 20
        self.max_subscriptions_per_connection = 240
        
        # Rate limit tracking
        self.rate_limit_hits = 0
        self.last_rate_limit_time = 0
        self.consecutive_rate_limits = 0
        
    async def acquire_public(self, endpoint_type='ticker'):
        """Acquire rate limit for public endpoint"""
        try:
            async with self.global_limiter:
                if endpoint_type in self.public_limiters:
                    async with self.public_limiters[endpoint_type]:
                        return True
                else:
                    # Default to 20 req/s for unknown public endpoints
                    async with self.public_limiters['ticker']:
                        return True
        except Exception as e:
            rate_limit_logger.error(f"Error acquiring public rate limit for {endpoint_type}: {e}")
            return False
    
    async def acquire_private(self, endpoint_type='orders'):
        """Acquire rate limit for private endpoint"""
        try:
            async with self.global_limiter:
                if endpoint_type in self.private_limiters:
                    async with self.private_limiters[endpoint_type]:
                        return True
                else:
                    # Default to 10 req/s for unknown private endpoints
                    async with self.private_limiters['orders']:
                        return True
        except Exception as e:
            rate_limit_logger.error(f"Error acquiring private rate limit for {endpoint_type}: {e}")
            return False
    
    def handle_rate_limit_error(self, error_code, endpoint, retry_count=0):
        """Handle rate limit errors with exponential backoff"""
        self.rate_limit_hits += 1
        self.last_rate_limit_time = time.time()
        self.consecutive_rate_limits += 1
        
        # Calculate backoff time
        base_delay = 2 ** min(retry_count, 6)  # Max 64 seconds
        jitter = np.random.uniform(0.1, 0.5)
        backoff_time = base_delay + jitter
        
        rate_limit_logger.warning(
            f"Rate limit hit - Endpoint: {endpoint}, Error: {error_code}, "
            f"Consecutive hits: {self.consecutive_rate_limits}, "
            f"Backing off for {backoff_time:.2f}s"
        )
        
        # If too many consecutive rate limits, extend backoff
        if self.consecutive_rate_limits >= 5:
            backoff_time *= 2
            rate_limit_logger.error(
                f"Excessive rate limiting detected! Extended backoff: {backoff_time:.2f}s"
            )
        
        return backoff_time
    
    def reset_rate_limit_tracking(self):
        """Reset rate limit tracking after successful requests"""
        if time.time() - self.last_rate_limit_time > 60:  # Reset after 1 minute of no rate limits
            self.consecutive_rate_limits = 0
    
    def get_suggested_wait_time(self, endpoint_type):
        """Get suggested wait time for an endpoint"""
        if endpoint_type in self.public_limiters:
            return 1.0 / 20  # 50ms for 20 req/s
        elif endpoint_type in self.private_limiters:
            if endpoint_type == 'batch-orders':
                return 1.0 / 5  # 200ms for 5 req/s
            return 1.0 / 10  # 100ms for 10 req/s
        return 1.0 / 10  # Default to conservative 100ms

def fetch_ohlcv_data(args):
    symbol, tf, exchange_config = args
    try:
        # Create a new exchange instance for each process
        exchange_id = exchange_config['exchange']
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'apiKey': exchange_config['api_key'],
            'secret': exchange_config['api_secret'],
            'enableRateLimit': True,
        })
        if exchange_config.get('use_sandbox', False):
            exchange.set_sandbox_mode(True)
            
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=200)
        if not ohlcv or len(ohlcv) == 0:
            return (symbol, tf), None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return (symbol, tf), df
    except Exception:
        logger.exception(f"Error fetching {symbol} {tf}")
        return (symbol, tf), None

class EnhancedSuperTrend:
    def __init__(self, config_file='test_batch.json'):
        """Initialize the Enhanced SuperTrend strategy"""
        self.config = self.load_config(config_file)
        print(f"\nLoaded config: {json.dumps(self.config, indent=2)}\n")
        
        # Initialize Bitget rate limiter
        self.rate_limiter = BitgetRateLimiter()
        logger.info("✅ Bitget rate limiter initialized")
        
        self.exchange = self.setup_exchange()

        # --- TOP 20 COINS DYNAMIC SELECTION ---
        symbols = []
        try:
            # Try to use VolatilityScanner for top 20 coins by volume
            vs_mod = importlib.import_module('volatility_scanner')
            VolatilityScanner = getattr(vs_mod, 'VolatilityScanner')
            scanner = VolatilityScanner(
                api_key=self.config['testnet']['api_key'],
                api_secret=self.config['testnet']['api_secret'],
                passphrase=self.config['testnet'].get('password'),
                testnet=self.config['testnet'].get('use_sandbox', True)
            )
            # Use asyncio to run the scan synchronously
            import asyncio
            loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            top_coins = loop.run_until_complete(scanner.scan_all_markets_ultra_fast(
                timeframe='4h', top_n=20, min_volume=self.config.get('execution', {}).get('min_volume', 1000000)
            ))
            symbols = [c['symbol'].replace(':USDT', '/USDT').replace('USDT/USDT', 'USDT') for c in top_coins]
            print(f"{HEADER_BG}{HEADER_FG}{BOLD}Using TOP 20 COINS by volume (Bitget USDT-margined swaps):{RESET}")
            print(symbols)
            except Exception as e:
            print(f"{SELL_FG}{BOLD}Could not fetch top 20 coins dynamically, falling back to hardcoded list. Reason: {e}{RESET}")
            # Fallback: Hardcoded top 20 by market cap (as of 2024)
            symbols = [
                "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT", "TRX/USDT:USDT", "LINK/USDT:USDT",
                "MATIC/USDT:USDT", "DOT/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT", "SHIB/USDT:USDT", "UNI/USDT:USDT", "ICP/USDT:USDT", "NEAR/USDT:USDT", "FIL/USDT:USDT", "ETC/USDT:USDT"
            ]
            print(f"{HEADER_BG}{HEADER_FG}{BOLD}Using fallback TOP 20 COINS (hardcoded):{RESET}")
            print(symbols)
        
            self.config['symbols'] = symbols
        # --- END TOP 20 COINS SELECTION ---

        self.active_trades = {}
        self.position_status = {}
        self.dashboard_data = {
            'strategy_status': 'idle',
            'active_symbols': [],
            'signals': [],
            'positions': [],
            'performance': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_profit': 0.0
            }
        }
        
        # Initialize rate limit tracking
        self.request_timestamps = []
        self.last_request_time = 0
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as file:
                config = json.load(file)
                
            # Verify testnet configuration
            if 'testnet' not in config:
                raise ValueError("Missing testnet configuration")
            if 'api_key' not in config['testnet'] or 'api_secret' not in config['testnet']:
                raise ValueError("Missing API credentials in testnet configuration")
                
            logger.info(f"Configuration loaded successfully from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
        
    def setup_exchange(self):
        """Initialize exchange connection with testnet"""
        try:
            # Always use Bitget and testnet
            exchange_id = "bitget"
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'apiKey': self.config['testnet']['api_key'],
                'secret': self.config['testnet']['api_secret'],
                'password': self.config['testnet']['password'],  # Added password
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',  # Use USDT-margined futures
                    'testnet': True,  # Force testnet mode
                }
            })
                exchange.set_sandbox_mode(True)
            logger.info(f"🔵 Connected to {exchange_id} TESTNET (sandbox mode enabled)")

            # Verify testnet connection
            try:
                balance = exchange.fetch_balance()
                logger.info(f"✅ Testnet balance: {balance['total']['USDT']} USDT")
            except Exception as e:
                logger.error(f"❌ Failed to fetch testnet balance: {e}")
                raise

            return exchange
        except Exception as e:
            logger.error(f"❌ Error setting up exchange: {e}")
            raise
        
    def calculate_supertrend(self, df, length=None, multiplier=None):
        """Calculate SuperTrend indicator"""
        length = length or self.config['st_length']
        multiplier = multiplier or self.config['st_multiplier']
        
        # Calculate ATR
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        df['atr'] = df['tr'].rolling(length).mean()
        
        # Calculate SuperTrend
        df['upperband'] = ((df['high'] + df['low']) / 2) + (multiplier * df['atr'])
        df['lowerband'] = ((df['high'] + df['low']) / 2) - (multiplier * df['atr'])
        df['in_uptrend'] = True
        
        for i in range(1, len(df)):
            if df['close'].loc[df.index[i]] > df['upperband'].loc[df.index[i-1]]:
                df['in_uptrend'].loc[df.index[i]] = True
            elif df['close'].loc[df.index[i]] < df['lowerband'].loc[df.index[i-1]]:
                df['in_uptrend'].loc[df.index[i]] = False
            else:
                df['in_uptrend'].loc[df.index[i]] = df['in_uptrend'].loc[df.index[i-1]]
                
                if df['in_uptrend'].loc[df.index[i]] and df['lowerband'].loc[df.index[i]] < df['lowerband'].loc[df.index[i-1]]:
                    df['lowerband'].loc[df.index[i]] = df['lowerband'].loc[df.index[i-1]]
                if not df['in_uptrend'].loc[df.index[i]] and df['upperband'].loc[df.index[i]] > df['upperband'].loc[df.index[i-1]]:
                    df['upperband'].loc[df.index[i]] = df['upperband'].loc[df.index[i-1]]
        
        # Add SuperTrend values
        df['supertrend'] = np.nan
        for i in range(len(df)):
            if df['in_uptrend'].loc[df.index[i]]:
                df['supertrend'].loc[df.index[i]] = df['lowerband'].loc[df.index[i]]
            else:
                df['supertrend'].loc[df.index[i]] = df['upperband'].loc[df.index[i]]
                
        return df
    
    def calculate_rsi(self, df, period=None):
        """Calculate RSI indicator"""
        period = period or self.config['enhanced_st']['rsi_period']
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def apply_volume_filter(self, df):
        """Apply volume filter to signals"""
        if self.config['enhanced_st']['volume_filter']:
            # Calculate average volume
            df['avg_volume'] = df['volume'].rolling(20).mean()
            # Only consider signals when volume is above threshold
            df['volume_signal'] = df['volume'] > (df['avg_volume'] * self.config['enhanced_st']['volume_threshold'])
            return df['volume_signal']
        return pd.Series([True] * len(df), index=df.index)
    
    def apply_rsi_filter(self, df):
        """Apply RSI filter to signals"""
        if self.config['enhanced_st']['rsi_filter']:
            overbought = self.config['enhanced_st']['rsi_overbought']
            oversold = self.config['enhanced_st']['rsi_oversold']
            
            # Create RSI signals
            df['rsi_buy_signal'] = df['rsi'] < oversold
            df['rsi_sell_signal'] = df['rsi'] > overbought
            
            return df['rsi_buy_signal'], df['rsi_sell_signal']
        return pd.Series([True] * len(df), index=df.index), pd.Series([True] * len(df), index=df.index)
    
    async def get_ohlcv_data(self, symbol, timeframe):
        """Fetch OHLCV data from exchange with rate limiting"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Acquire rate limit for candles endpoint
                if not await self.rate_limiter.acquire_public('candles'):
                    rate_limit_logger.warning(f"Failed to acquire rate limit for candles: {symbol}")
                    await asyncio.sleep(1)
                    continue
                
            print(f"Requesting OHLCV for {symbol} {timeframe}...")
                start_time = time.time()
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=200)
                response_time = time.time() - start_time
                
                # Record successful API request
                record_api_request('candles', 'public', True, None, response_time)
                
                # Reset rate limit tracking on success
                self.rate_limiter.reset_rate_limit_tracking()
                
            if not ohlcv or len(ohlcv) == 0:
                print(f"[ERROR] No OHLCV data returned for {symbol} {timeframe}!")
                return None
                    
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            print(f"First rows for {symbol} {timeframe}:\n{df.head()}\n")
            return df
                
        except Exception as e:
                error_str = str(e)
                
                # Check for Bitget rate limit error codes
                if '429' in error_str or '30007' in error_str or '30001' in error_str:
                    backoff_time = self.rate_limiter.handle_rate_limit_error(error_str, 'candles', retry_count)
                    
                    # Record rate limit error
                    record_api_request('candles', 'public', False, error_str, 0.0, retry_count, backoff_time)
                    
                    rate_limit_logger.error(
                        f"Rate limit error fetching {symbol} {timeframe}: {error_str}. "
                        f"Backing off for {backoff_time:.2f}s"
                    )
                    await asyncio.sleep(backoff_time)
                    retry_count += 1
                    continue
                
            print(f"[ERROR] Exception fetching data for {symbol} {timeframe}: {e}")
            logger.error(f"Error fetching data for {symbol}: {e}")
                return None
        
        logger.error(f"Failed to fetch OHLCV data for {symbol} after {max_retries} retries")
            return None
    
    def calculate_position_size(self, symbol, entry_price):
        """Calculate position size based on risk parameters"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['total']['USDT']
            risk_amount = usdt_balance * (self.config['risk_management']['position_size_percent'] / 100)
            position_size = risk_amount / entry_price
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def place_order(self, symbol, side, amount, price=None):
        """Place order on exchange with robust testnet logic and Bitget rate limiting"""
        max_retries = 3
        attempt = 0
        last_error = None
        order = None
        
        while attempt < max_retries:
            try:
                # Acquire rate limit for ticker endpoint first
                if not await self.rate_limiter.acquire_public('ticker'):
                    rate_limit_logger.warning(f"Failed to acquire rate limit for ticker: {symbol}")
                    await asyncio.sleep(1)
                    continue
                
                # Acquire rate limit for orders endpoint
                if not await self.rate_limiter.acquire_private('orders'):
                    rate_limit_logger.warning(f"Failed to acquire rate limit for orders: {symbol}")
                    await asyncio.sleep(1)
                    continue
                
                order_type = 'market' if attempt == 0 else 'limit'
                params = {
                    'testnet': True,
                    'reduceOnly': False,
                    'marginMode': 'cross',
                    'positionSide': 'long' if side == 'buy' else 'short',
                }
                
                # Always fetch latest ticker and mark price
                start_time = time.time()
                ticker = self.exchange.fetch_ticker(symbol)
                response_time = time.time() - start_time
                
                # Record ticker API request
                record_api_request('ticker', 'public', True, None, response_time)
                
                mark_price = ticker.get('markPrice', ticker.get('last'))
                if mark_price is None:
                    raise ValueError(f"Could not fetch mark/last price for {symbol}")
                if price is None or attempt > 0:
                    offset = 1 + (0.0001 * (attempt + 1)) if side == 'buy' else 1 - (0.0001 * (attempt + 1))
                    price = mark_price * offset
                if amount is None:
                    amount = self.calculate_position_size(symbol, price)
                
                logger.info(f"Attempt {attempt+1}: Placing {order_type} order for {side} {amount} {symbol} @ {price}")
                
                start_time = time.time()
                if order_type == 'market':
                    # For Bitget, market orders need amount not cost
                    logger.info(f"Market {side}: amount={amount}, price={price}")
                    order = self.exchange.create_order(
                        symbol, order_type, side, amount, None, params
                    )
                else:
            order = self.exchange.create_order(
                symbol, order_type, side, amount, price, params
            )
            
                order_response_time = time.time() - start_time
                
                # Record successful order placement
                record_api_request('orders', 'private', True, None, order_response_time)
                
                # Reset rate limit tracking on success
                self.rate_limiter.reset_rate_limit_tracking()
                
                if order and order.get('id'):
                    logger.info(f"💫 TESTNET Order placed: {side} {amount} {symbol} @ {price} (order: {order})")
                    rate_limit_logger.info(f"Successful order placement: {symbol} {side} {amount}")
            self.update_dashboard({
                'type': 'order',
                'data': {
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'status': 'filled',
                    'timestamp': datetime.now().isoformat()
                }
            })
            return order
            
        except Exception as e:
                last_error = str(e)
                error_str = str(e)
                
                # Check for Bitget rate limit error codes
                if '429' in error_str or '30007' in error_str or '30001' in error_str:
                    backoff_time = self.rate_limiter.handle_rate_limit_error(error_str, 'orders', attempt)
                    
                    # Record rate limit error for orders
                    record_api_request('orders', 'private', False, error_str, 0.0, attempt, backoff_time)
                    
                    rate_limit_logger.error(
                        f"Rate limit error placing order {symbol} {side}: {error_str}. "
                        f"Backing off for {backoff_time:.2f}s"
                    )
                    await asyncio.sleep(backoff_time)
                    attempt += 1
                    continue
                
                # Check for Bitget order error codes
                elif '50067' in error_str:
                    # Price deviation error - adjust price and retry
                    logger.warning(f"Price deviation error for {symbol}, adjusting price...")
                    attempt += 1
                    await asyncio.sleep(1)
                    continue
                elif '50001' in error_str:
                    # Insufficient balance
                    logger.error(f"Insufficient balance for {symbol} {side} {amount}")
                    break
                elif '50002' in error_str or '50003' in error_str:
                    # Order size issues
                    logger.error(f"Order size issue for {symbol}: {error_str}")
                    break
                else:
                    logger.error(f"❌ Error placing order (attempt {attempt+1}): {e}")
                    break
                    
        logger.error(f"❌ All attempts to place order failed: {last_error}")
        rate_limit_logger.error(f"Failed order placement after {max_retries} attempts: {symbol} {side} - {last_error}")
        
            self.update_dashboard({
                'type': 'error',
                'data': {
                'message': f"Error placing order after {max_retries} attempts: {last_error}",
                    'symbol': symbol,
                    'side': side,
                    'timestamp': datetime.now().isoformat()
                }
            })
            return None
    
    async def analyze_symbol(self, symbol, timeframe=None):
        """Analyze a symbol for SuperTrend signals with rate limiting"""
        timeframe = timeframe or self.config['timeframes'][0]
        
        # Fetch data with rate limiting
        df = await self.get_ohlcv_data(symbol, timeframe)
        if df is None or len(df) < 50:
            logger.warning(f"Not enough data for {symbol}")
            return None
        
        # Calculate indicators
        df = self.calculate_supertrend(df)
        df = self.calculate_rsi(df)
        
        # Apply filters
        volume_signal = self.apply_volume_filter(df)
        rsi_buy_signal, rsi_sell_signal = self.apply_rsi_filter(df)
        
        # Get latest data point
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Detect SuperTrend changes (signals)
        signal = None
        if not previous['in_uptrend'] and current['in_uptrend'] and volume_signal.loc[df.index[-1]] and rsi_buy_signal.loc[df.index[-1]]:
            signal = {
                'type': 'buy',
                'symbol': symbol,
                'price': current['close'],
                'supertrend': current['supertrend'],
                'rsi': current['rsi'],
                'timestamp': current.name.isoformat(),
                'volume_ratio': current['volume'] / current['avg_volume'] if 'avg_volume' in current else 1.0
            }
            logger.info(f"BUY Signal: {symbol} @ {current['close']} (SuperTrend: {current['supertrend']:.2f}, RSI: {current['rsi']:.2f})")
        
            # Execute buy order with rate limiting
            amount = self.calculate_position_size(symbol, current['close'])
            order = await self.place_order(symbol, 'buy', amount, current['close'])
            if order:
                logger.info(f"BUY Order executed: {order}")
                rate_limit_logger.info(f"BUY Order executed for {symbol} at {current['close']}")
        
        elif previous['in_uptrend'] and not current['in_uptrend'] and rsi_sell_signal.loc[df.index[-1]]:
            signal = {
                'type': 'sell',
                'symbol': symbol,
                'price': current['close'],
                'supertrend': current['supertrend'],
                'rsi': current['rsi'],
                'timestamp': current.name.isoformat(),
                'volume_ratio': current['volume'] / current['avg_volume'] if 'avg_volume' in current else 1.0
            }
            logger.info(f"SELL Signal: {symbol} @ {current['close']} (SuperTrend: {current['supertrend']:.2f}, RSI: {current['rsi']:.2f})")
            
            # Execute sell order with rate limiting
            amount = self.calculate_position_size(symbol, current['close'])
            order = await self.place_order(symbol, 'sell', amount, current['close'])
            if order:
                logger.info(f"SELL Order executed: {order}")
                rate_limit_logger.info(f"SELL Order executed for {symbol} at {current['close']}")
        
        # Update dashboard with signal
        if signal:
            self.update_dashboard({
                'type': 'signal',
                'data': signal
            })
            
        return {
            'symbol': symbol,
            'last_close': current['close'],
            'supertrend': current['supertrend'],
            'in_uptrend': current['in_uptrend'],
            'rsi': current['rsi'],
            'signal': signal,
            'timestamp': current.name.isoformat()
        }
    
    def update_dashboard(self, update):
        """Update dashboard data for UI"""
        if update['type'] == 'signal':
            self.dashboard_data['signals'].append(update['data'])
            # Keep only the last 20 signals
            self.dashboard_data['signals'] = self.dashboard_data['signals'][-20:]
            
        elif update['type'] == 'order':
            # Update positions list
            symbol = update['data']['symbol']
            side = update['data']['side']
            price = update['data']['price']
            amount = update['data']['amount']
            
            # Find existing position or create new one
            position_exists = False
            for i, pos in enumerate(self.dashboard_data['positions']):
                if pos['symbol'] == symbol:
                    position_exists = True
                    if side == 'sell':
                        # Close position
                        entry_price = pos['entry_price']
                        pnl = (price - entry_price) * pos['amount'] if pos['side'] == 'buy' else (entry_price - price) * pos['amount']
                        self.dashboard_data['positions'].pop(i)
                        
                        # Update performance metrics
                        self.dashboard_data['performance']['total_trades'] += 1
                        if pnl > 0:
                            self.dashboard_data['performance']['winning_trades'] += 1
                        else:
                            self.dashboard_data['performance']['losing_trades'] += 1
                        self.dashboard_data['performance']['total_profit'] += pnl
                        
                        # Calculate win rate and profit factor
                        if self.dashboard_data['performance']['total_trades'] > 0:
                            self.dashboard_data['performance']['win_rate'] = (
                                self.dashboard_data['performance']['winning_trades'] / 
                                self.dashboard_data['performance']['total_trades']
                            ) * 100
                    break
            
            if not position_exists and side == 'buy':
                # Add new position
                self.dashboard_data['positions'].append({
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'entry_price': price,
                    'current_price': price,
                    'pnl': 0.0,
                    'pnl_percent': 0.0,
                    'timestamp': update['data']['timestamp']
                })
            
            # Update active symbols list
            active_symbols = [pos['symbol'] for pos in self.dashboard_data['positions']]
            self.dashboard_data['active_symbols'] = active_symbols
            
        elif update['type'] == 'price':
            # Update price for a symbol
            symbol = update['data']['symbol']
            price = update['data']['price']
            
            # Update position current price and pnl
            for pos in self.dashboard_data['positions']:
                if pos['symbol'] == symbol:
                    pos['current_price'] = price
                    if pos['side'] == 'buy':
                        pos['pnl'] = (price - pos['entry_price']) * pos['amount']
                        pos['pnl_percent'] = ((price / pos['entry_price']) - 1) * 100
                    else:
                        pos['pnl'] = (pos['entry_price'] - price) * pos['amount']
                        pos['pnl_percent'] = ((pos['entry_price'] / price) - 1) * 100
                    break
        
        elif update['type'] == 'status':
            # Update strategy status
            self.dashboard_data['strategy_status'] = update['data']['status']
        
        elif update['type'] == 'error':
            # Add error to signals list with special type
            error_signal = {
                'type': 'error',
                'message': update['data']['message'],
                'symbol': update['data'].get('symbol', 'system'),
                'timestamp': update['data']['timestamp']
            }
            self.dashboard_data['signals'].append(error_signal)
            # Keep only the last 20 signals
            self.dashboard_data['signals'] = self.dashboard_data['signals'][-20:]
    
    def get_dashboard_data(self):
        """Get current dashboard data for UI"""
        # Add timestamp to the data
        dashboard_copy = self.dashboard_data.copy()
        dashboard_copy['timestamp'] = datetime.now().isoformat()
        return dashboard_copy
    
    async def start_dashboard_server(self, host='0.0.0.0', port=8765):
        """Start a WebSocket server for dashboard updates"""
        async def handler(websocket, path):
            """Handle WebSocket connections"""
            logger.info(f"New dashboard connection established")
            
            # Send initial data
            await websocket.send(json.dumps({
                'type': 'dashboard_update',
                'data': self.get_dashboard_data()
            }))
            
            try:
                # Keep connection alive and send updates
                while True:
                    try:
                        # Handle incoming messages (e.g., for commands)
                        message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        data = json.loads(message)
                        
                        # Handle commands
                        if data.get('type') == 'command':
                            cmd = data.get('command')
                            if cmd == 'start':
                                self.update_dashboard({
                                    'type': 'status',
                                    'data': {'status': 'running'}
                                })
                            elif cmd == 'stop':
                                self.update_dashboard({
                                    'type': 'status',
                                    'data': {'status': 'stopped'}
                                })
                            elif cmd == 'refresh':
                                # Force refresh all data
                                pass
                    
                    except asyncio.TimeoutError:
                        # No message received, continue
                        pass
                    
                    # Send regular updates
                    await websocket.send(json.dumps({
                        'type': 'dashboard_update',
                        'data': self.get_dashboard_data()
                    }))
                    
                    await asyncio.sleep(1)
            
            except websockets.exceptions.ConnectionClosed:
                logger.info("Dashboard connection closed")
            except Exception as e:
                logger.error(f"Error in dashboard connection: {e}")
        
        # Start WebSocket server
        logger.info(f"Starting dashboard server on ws://{host}:{port}")
        server = await websockets.serve(handler, host, port)
        
        # Keep server running
        await server.wait_closed()
    
    def print_signal_table(self, results):
        print(f"\n{HEADER_BG}{HEADER_FG}{BOLD}{'='*80}{RESET}")
        print(f"{HEADER_FG}{BOLD}{'SuperTrend Signal Summary':^80}{RESET}")
        print(f"{HEADER_BG}{HEADER_FG}{BOLD}{'='*80}{RESET}")
        print(f"{BUY_FG}{BOLD}{'Symbol':<12}{'TF':<6}{'Signal':<10}{'Price':<12}{'Time':<20}{'Volume':<8}{RESET}")
        print(f"{HEADER_BG}{HEADER_FG}{BOLD}{'-'*80}{RESET}")
        for r in results:
            color = BUY_FG + BOLD if r['signal']=='BUY' else SELL_FG + BOLD if r['signal']=='SELL' else HOLD_FG + BOLD
            print(f"{color}{r['symbol']:<12}{r['timeframe']:<6}{r['signal']:<10}{r['price']:<12.2f}{r['time']:<20}{r['volume']:<8.2f}{RESET}")
        print(f"{HEADER_BG}{HEADER_FG}{BOLD}{'='*80}{RESET}\n")

    def print_performance_summary(self):
        perf = self.dashboard_data['performance']
        print(f"{HEADER_FG}{BOLD}{'🏆 Performance Summary 🏆':^80}{RESET}")
        print(f"{HEADER_BG}{HEADER_FG}{BOLD}{'━'*80}{RESET}")
        print(f"{HEADER_FG}{BOLD}Total Trades: {perf['total_trades']}  |  Win Rate: {perf['win_rate']:.2f}%  |  Profit Factor: {perf['profit_factor']:.2f}{RESET}")
        print(f"{BUY_FG}{BOLD}Total Profit: {perf['total_profit']:.2f}  |  Wins: {perf['winning_trades']}  |  Losses: {perf['losing_trades']}{RESET}")
        print(f"{HEADER_BG}{HEADER_FG}{BOLD}{'━'*80}{RESET}")
        print(f"{HEADER_FG}{BOLD}{'✨🚀💰🔥🤑💎👑🌈🎉🎊🎯🥇🥈🥉🏅🏆':^80}{RESET}\n")

    def detect_signal(self, last, vol_signal, rsi_buy, rsi_sell, method='strict'):
        """Detect signal with different strictness methods"""
        # Strict: All filters must be true
        if method == 'strict':
            buy = last['in_uptrend'] and vol_signal and rsi_buy
            sell = not last['in_uptrend'] and vol_signal and rsi_sell
        # Moderate: At least one filter (SuperTrend + (vol or rsi))
        elif method == 'moderate':
            buy = last['in_uptrend'] and (vol_signal or rsi_buy)
            sell = not last['in_uptrend'] and (vol_signal or rsi_sell)
        # Loose: Only SuperTrend
        elif method == 'loose':
            buy = last['in_uptrend']
            sell = not last['in_uptrend']
        return buy, sell

    def batch_fetch_ohlcv(self, tasks):
        """Optimized parallel OHLCV fetching with rate limiting"""
        max_workers = min(multiprocessing.cpu_count() * 2, 32)
        chunk_size = 10  # Process in smaller chunks to respect rate limits
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Split tasks into chunks
            for i in range(0, len(tasks), chunk_size):
                chunk = tasks[i:i + chunk_size]
                futures = []
                
                for task in chunk:
                    # Add rate limiting delay
                    current_time = time.time()
                    if self.request_timestamps:
                        time_since_last = current_time - self.last_request_time
                        if time_since_last < (1.0 / self.rate_limiter.get_suggested_wait_time('ticker')):
                            time.sleep((1.0 / self.rate_limiter.get_suggested_wait_time('ticker')) - time_since_last)
                    
                    # Create new exchange instance for each task
                    exchange_config = self.config['testnet']
                    future = executor.submit(self._fetch_ohlcv_with_retry, task[0], task[1], exchange_config)
                    futures.append(future)
                    self.request_timestamps.append(current_time)
                    self.last_request_time = current_time
                
                # Process chunk results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        key, data = future.result()
                        if data is not None:
                            results[key] = data
                    except Exception as e:
                        logger.error(f"Error in batch fetch: {e}")
                        self.rate_limiter.handle_rate_limit_error(e, 'ticker')
                        
                        # If we hit too many rate limits, pause and reset
                        if self.rate_limiter.rate_limit_hits >= 5:
                            logger.warning("Too many rate limit hits, pausing for 60 seconds...")
                            time.sleep(60)
                            self.rate_limiter.reset_rate_limit_tracking()
        
        return results

    def _fetch_ohlcv_with_retry(self, task, max_retries=3):
        """Fetch OHLCV data with exponential backoff retry"""
        symbol, tf, exchange_config = task
        retry_count = 0
        while retry_count < max_retries:
            try:
                exchange = self._get_exchange_instance(exchange_config)
                ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=200)
                
                if not ohlcv or len(ohlcv) == 0:
                    return (symbol, tf), None
                    
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return (symbol, tf), df
                
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    logger.error(f"Failed to fetch {symbol} {tf} after {max_retries} retries: {e}")
                    return (symbol, tf), None
                    
                # Exponential backoff
                time.sleep(2 ** retry_count)

    def _get_exchange_instance(self, exchange_config):
        """Create a new exchange instance for parallel processing"""
        try:
            exchange_id = exchange_config['exchange']
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'apiKey': exchange_config['api_key'],
                'secret': exchange_config['api_secret'],
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',
                    'testnet': True,
                }
            })
            exchange.set_sandbox_mode(True)
            return exchange
        except Exception as e:
            logger.error(f"Error creating exchange instance: {e}")
            raise

    def analyze_symbol_timeframe_fast(self, args_df):
        (symbol, tf), df = args_df
        if df is None or df.empty:
            return None
        df = self.calculate_supertrend(df)
        df = self.calculate_rsi(df)
        vol_signal = self.apply_volume_filter(df)
        rsi_buy, rsi_sell = self.apply_rsi_filter(df)
        last = df.iloc[-1]
        results = {}
        for method in ['strict', 'moderate', 'loose']:
            buy, sell = self.detect_signal(last, vol_signal.loc[df.index[-1]], rsi_buy.loc[df.index[-1]], rsi_sell.loc[df.index[-1]], method)
            if buy:
                signal = 'BUY'
            elif sell:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            filters = {
                'vol': bool(vol_signal.loc[df.index[-1]]),
                'rsi': bool(rsi_buy.loc[df.index[-1]]) if buy else bool(rsi_sell.loc[df.index[-1]]) if sell else False,
                'st': bool(last['in_uptrend']) if buy else not last['in_uptrend'] if sell else False
            }
            results[method] = {
                                'symbol': symbol,
                'timeframe': tf,
                'signal': signal,
                'price': last['close'],
                'time': last.name.strftime('%Y-%m-%d %H:%M'),
                'volume': last['volume'],
                'filters': filters
            }
        return results

    def print_cyberpunk_table(self, results, method):
        table = prettytable.PrettyTable()
        table.field_names = [
            f"{HEADER_BG}{HEADER_FG}{BOLD}Symbol{RESET}",
            f"{HEADER_BG}{HEADER_FG}{BOLD}TF{RESET}",
            f"{HEADER_BG}{HEADER_FG}{BOLD}Signal{RESET}",
            f"{HEADER_BG}{HEADER_FG}{BOLD}Price{RESET}",
            f"{HEADER_BG}{HEADER_FG}{BOLD}Time{RESET}",
            f"{HEADER_BG}{HEADER_FG}{BOLD}Volume{RESET}",
            f"{HEADER_BG}{HEADER_FG}{BOLD}Filters{RESET}",
            f"{HEADER_BG}{HEADER_FG}{BOLD}Score{RESET}"
        ]
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        for r in results:
            if not r: continue
            r = r[method]
            if r['signal'] == 'BUY':
                color = BUY_FG + BOLD
                emoji = '🟢'
                signal_counts['BUY'] += 1
            elif r['signal'] == 'SELL':
                color = SELL_FG + BOLD
                emoji = '🔴'
                signal_counts['SELL'] += 1
            else:
                color = HOLD_FG + BOLD
                emoji = '⚪️'
                signal_counts['HOLD'] += 1
            filters = r.get('filters', {})
            fstr = ''
            fstr += f"{'✔️' if filters.get('vol') else '❌'}Vol "
            fstr += f"{'✔️' if filters.get('rsi') else '❌'}RSI "
            fstr += f"{'✔️' if filters.get('st') else '❌'}ST "
            score = 0
            score += 1 if filters.get('vol') else 0
            score += 1 if filters.get('rsi') else 0
            score += 1 if filters.get('st') else 0
            table.add_row([
                f"{color}{r['symbol']}{RESET}",
                f"{color}{r['timeframe']}{RESET}",
                f"{color}{emoji} {r['signal']}{RESET}",
                f"{r['price']:.2f}",
                f"{r['time']}",
                f"{r['volume']:.2f}",
                fstr.strip(),
                score
            ])
        print(f"\n{HEADER_BG}{HEADER_FG}{BOLD}=== SIGNALS ({method.title()}) ==={RESET}")
        print(table)
        print(f"{BOLD}Summary: {BUY_FG}BUY: {signal_counts['BUY']}  {SELL_FG}SELL: {signal_counts['SELL']}  {HOLD_FG}HOLD: {signal_counts['HOLD']}{RESET}\n")

    async def run_strategy(self):
        """Run the trading strategy"""
        try:
            # First verify testnet trading
            logger.info("🧪 Testing testnet trading...")
            if await self.test_trade():
                logger.info("✅ Testnet trading verified")
            else:
                logger.error("❌ Testnet trading verification failed")
                return
            
            # Continue with normal strategy execution
        methods = ['strict', 'moderate', 'loose']
        timeframes = self.config.get('timeframes', [self.config.get('timeframe', '1m')])
        symbols = self.config['symbols']
            
            logger.info(f"🚀 Starting strategy with {len(symbols)} symbols")
            
            iteration_count = 0
            while True:  # Main trading loop
                try:
                    iteration_count += 1
                    logger.info(f"🔄 Starting iteration {iteration_count}")
                    
                    # Show rate limit status every 10 iterations
                    if iteration_count % 10 == 1:
                        self.print_rate_limit_status()
                        if iteration_count > 1:
                            self.generate_and_log_rate_report(hours=1)
                    
                    # Fetch and analyze data
        tasks = list(itertools.product(symbols, timeframes))
        ohlcv_data = self.batch_fetch_ohlcv(tasks)
                    
                    # Process signals with rate limiting
                    for symbol in symbols:
                        for tf in timeframes:
                            key = (symbol, tf)
                            if key in ohlcv_data and ohlcv_data[key] is not None:
                                signal = await self.analyze_symbol(symbol, tf)
                                if signal:
                                    logger.info(f"Signal for {symbol}: {signal}")
                                    rate_limit_logger.info(f"Signal processed for {symbol}: {signal.get('signal', {}).get('type', 'no signal')}")
                            
                            # Add small delay between symbols to respect rate limits
                            await asyncio.sleep(0.1)
                    
                    # Quick rate limit status check every iteration
                    if iteration_count % 5 == 0:
                        status = get_rate_status()
                        if 'alerts' in status and status['alerts'].get('active_rate_limits', 0) > 0:
                            print(f"{SELL_FG}{BOLD}⚠️ Rate limit alerts detected! Iteration {iteration_count}{RESET}")
                    
                    # Wait before next iteration
                    logger.info(f"✅ Iteration {iteration_count} completed. Waiting 60 seconds...")
                    await asyncio.sleep(60)  # 1 minute delay
                    
                except Exception as e:
                    logger.error(f"❌ Error in strategy loop: {e}")
                    await asyncio.sleep(5)  # Wait before retry
            
        except Exception as e:
            logger.error(f"❌ Strategy execution failed: {e}")
            raise

    async def setup_websocket_connections(self):
        """Setup optimized WebSocket connections for real-time data"""
        symbols = self.config['symbols']
        max_subscriptions = self.rate_limiter.max_subscriptions_per_connection
        
        # Split symbols into chunks for multiple connections
        symbol_chunks = [symbols[i:i + max_subscriptions] 
                        for i in range(0, len(symbols), max_subscriptions)]
        
        self.ws_connections = []
        for chunk in symbol_chunks:
            try:
                ws = await websockets.connect('wss://api.bitget.com/ws/v1')
                
                # Subscribe to multiple symbols in one message
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [f"ticker.{symbol}" for symbol in chunk]
                }
                await ws.send(json.dumps(subscribe_msg))
                
                # Start message handler and ping/pong
                asyncio.create_task(self._handle_websocket_messages(ws, chunk))
                asyncio.create_task(self._maintain_websocket_connection(ws))
                
                self.ws_connections.append(ws)
                
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")

    async def _maintain_websocket_connection(self, ws):
        """Maintain WebSocket connection with ping/pong"""
        while True:
            try:
                await ws.send(json.dumps({"op": "ping"}))
                await asyncio.sleep(self.rate_limiter.get_suggested_wait_time('ticker'))
            except Exception as e:
                logger.error(f"WebSocket ping error: {e}")
                # Attempt reconnection
                await self._reconnect_websocket(ws)

    def validate_ohlcv_data(self, df):
        """Validate OHLCV data quality"""
        if df is None or df.empty:
            return False
        
        # Check for missing values
        if df.isnull().any().any():
            logger.warning("OHLCV data contains missing values")
            return False
        
        # Check for price anomalies
        price_changes = df['close'].pct_change().abs()
        if (price_changes > 0.5).any():  # 50% price change threshold
            logger.warning("Detected potential price anomalies")
            return False
        
        # Check for volume anomalies
        volume_mean = df['volume'].mean()
        volume_std = df['volume'].std()
        if (df['volume'] > volume_mean + 3 * volume_std).any():
            logger.warning("Detected volume anomalies")
            return False
        
        # Check for timestamp continuity
        time_diff = df.index.to_series().diff()
        if (time_diff > pd.Timedelta(minutes=5)).any():
            logger.warning("Detected gaps in OHLCV data")
            return False
        
        return True

    def monitor_performance(self):
        """Monitor system performance metrics"""
        metrics = {
            'api_latency': [],
            'processing_time': [],
            'memory_usage': [],
            'error_rate': 0,
            'rate_limit_hits': 0,
            'requests_per_second': []
        }
        
        while True:
            try:
                # Calculate requests per second
                current_time = time.time()
                if self.request_timestamps:
                    recent_requests = [t for t in self.request_timestamps if current_time - t < 1.0]
                    metrics['requests_per_second'].append(len(recent_requests))
                    
                    # Clean up old timestamps
                    self.request_timestamps = [t for t in self.request_timestamps if current_time - t < 60.0]
                
                # Measure API latency
                start_time = time.time()
                self.exchange.fetch_time()
                latency = time.time() - start_time
                metrics['api_latency'].append(latency)
                
                # Monitor memory usage
                process = psutil.Process()
                metrics['memory_usage'].append(process.memory_info().rss / 1024 / 1024)
                
                # Calculate statistics
                if metrics['api_latency']:
                    avg_latency = sum(metrics['api_latency'][-100:]) / min(len(metrics['api_latency']), 100)
                    if avg_latency > 1.0:  # If average latency > 1 second
                        logger.warning(f"High API latency detected: {avg_latency:.2f}s")
                
                # Alert if approaching rate limits
                if metrics['requests_per_second'] and metrics['requests_per_second'][-1] > 15:
                    logger.warning(f"High request rate: {metrics['requests_per_second'][-1]} req/s")
                
                # Keep only last 1000 measurements
                for key in metrics:
                    if isinstance(metrics[key], list):
                        metrics[key] = metrics[key][-1000:]
                
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                metrics['error_rate'] += 1

    async def test_trade(self):
        """Place a test trade on testnet with rate limiting"""
        try:
            symbol = self.config['symbols'][0]  # Use first symbol
            amount = 0.01  # Minimum test amount for Bitget
            
            # Acquire rate limit for ticker
            if not await self.rate_limiter.acquire_public('ticker'):
                logger.error("Failed to acquire rate limit for ticker in test trade")
                return False
            
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']
            
            logger.info(f"🧪 Starting test trade for {symbol} at {price}")
            rate_limit_logger.info(f"Test trade initiated for {symbol}")
            
            # Place test buy order
            order = await self.place_order(symbol, 'buy', amount, price)
            if order:
                logger.info(f"✅ Test buy order placed: {order}")
                rate_limit_logger.info(f"Test buy order successful: {order.get('id', 'unknown')}")
                
                # Wait 5 seconds
                await asyncio.sleep(5)
                
                # Place test sell order
                sell_order = await self.place_order(symbol, 'sell', amount, price)
                if sell_order:
                    logger.info(f"✅ Test sell order placed: {sell_order}")
                    rate_limit_logger.info(f"Test sell order successful: {sell_order.get('id', 'unknown')}")
                    logger.info(f"🎉 Test trade completed successfully!")
                    return True
                
            return False
        except Exception as e:
            logger.error(f"❌ Test trade failed: {e}")
            rate_limit_logger.error(f"Test trade failed: {str(e)}")
            return False

    def print_rate_limit_status(self):
        """Print comprehensive rate limiting status"""
        try:
            status = get_rate_status()
            
            print(f"\n{HEADER_BG}{HEADER_FG}{BOLD}{'='*80}{RESET}")
            print(f"{HEADER_FG}{BOLD}{'🚦 BITGET API RATE LIMIT STATUS':^80}{RESET}")
            print(f"{HEADER_BG}{HEADER_FG}{BOLD}{'='*80}{RESET}")
            
            # Rate limit windows
            if 'windows' in status:
                print(f"\n{BUY_FG}{BOLD}📊 Current API Usage:{RESET}")
                for window_type, data in status['windows'].items():
                    limit = data.get('limit', 10)
                    current = data.get('requests_last_second', 0)
                    utilization = data.get('utilization', 0) * 100
                    
                    color = BUY_FG if utilization < 60 else (ACCENT if utilization < 80 else SELL_FG)
                    status_emoji = "🟢" if utilization < 60 else ("🟡" if utilization < 80 else "🔴")
                    
                    print(f"  {status_emoji} {window_type.title():<10} {current:>2}/{limit} req/s ({color}{utilization:>5.1f}%{RESET})")
            
            # Top endpoints
            if 'top_endpoints' in status and status['top_endpoints']:
                print(f"\n{BUY_FG}{BOLD}🔥 Most Active Endpoints:{RESET}")
                for i, endpoint in enumerate(status['top_endpoints'][:5], 1):
                    compliance = endpoint.get('compliance_score', 100)
                    success_rate = endpoint.get('success_rate', 100)
                    rps = endpoint.get('requests_per_second', 0)
                    
                    compliance_color = BUY_FG if compliance >= 90 else (ACCENT if compliance >= 80 else SELL_FG)
                    success_color = BUY_FG if success_rate >= 95 else (ACCENT if success_rate >= 90 else SELL_FG)
                    
                    print(f"  {i}. {endpoint['endpoint']:<15} "
                          f"RPS: {rps:>4.1f} | "
                          f"Success: {success_color}{success_rate:>5.1f}%{RESET} | "
                          f"Compliance: {compliance_color}{compliance:>5.1f}%{RESET}")
            
            # Alerts
            if 'alerts' in status:
                alerts = status['alerts']
                active_limits = alerts.get('active_rate_limits', 0)
                low_compliance = alerts.get('low_compliance_endpoints', [])
                
                if active_limits > 0 or low_compliance:
                    print(f"\n{SELL_FG}{BOLD}🚨 ACTIVE ALERTS:{RESET}")
                    
                    if active_limits > 0:
                        print(f"  🔴 Rate limit hits in last minute: {active_limits}")
                    
                    for endpoint_alert in low_compliance[:3]:
                        print(f"  ⚠️  Low compliance: {endpoint_alert['endpoint']} ({endpoint_alert['score']:.1f}%)")
            else:
                    print(f"\n{BUY_FG}{BOLD}✅ No active alerts - All systems operating normally{RESET}")
            
            # Rate limiter internal status
            print(f"\n{BUY_FG}{BOLD}⚙️ Rate Limiter Status:{RESET}")
            print(f"  🔄 Total rate limit hits: {self.rate_limiter.rate_limit_hits}")
            print(f"  ⏰ Last rate limit: {datetime.fromtimestamp(self.rate_limiter.last_rate_limit_time).strftime('%H:%M:%S') if self.rate_limiter.last_rate_limit_time > 0 else 'Never'}")
            print(f"  🔁 Consecutive rate limits: {self.rate_limiter.consecutive_rate_limits}")
            
            print(f"{HEADER_BG}{HEADER_FG}{BOLD}{'='*80}{RESET}\n")
            
        except Exception as e:
            logger.error(f"Error printing rate limit status: {e}")
            print(f"{SELL_FG}{BOLD}❌ Could not retrieve rate limit status: {e}{RESET}")

    def generate_and_log_rate_report(self, hours: int = 1):
        """Generate and log a rate limiting report"""
        try:
            report = generate_rate_report(hours)
            
            # Log to file
            with open('bitget_rate_report.txt', 'w') as f:
                f.write(report)
            
            logger.info(f"📊 Rate limit report generated for last {hours} hour(s)")
            
            # Print summary
            lines = report.split('\n')
            summary_lines = [line for line in lines if 'Total Requests:' in line or 'Rate Limit Hits:' in line or 'Success Rate:' in line]
            
            if summary_lines:
                print(f"\n{HEADER_FG}{BOLD}📈 Rate Limit Report Summary (Last {hours}h):{RESET}")
                for line in summary_lines[:3]:
                    print(f"  {line.strip()}")
                print(f"  📄 Full report saved to: bitget_rate_report.txt\n")
                
        except Exception as e:
            logger.error(f"Error generating rate report: {e}")

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, strategy):
        self.strategy = strategy
        self.last_reload = time.time()
        
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            current_time = time.time()
            if current_time - self.last_reload > 5:  # Prevent multiple reloads
                self.last_reload = current_time
                logger.info(f"🔄 Code change detected in {event.src_path}")
                self.reload_strategy()
    
    def reload_strategy(self):
        try:
            # Preserve existing connections
            old_exchange = self.strategy.exchange
            old_ws_connections = getattr(self.strategy, 'ws_connections', [])
            
            # Reload the strategy
            self.strategy = EnhancedSuperTrend()
            
            # Restore connections
            self.strategy.exchange = old_exchange
            self.strategy.ws_connections = old_ws_connections
            
            logger.info("✅ Strategy reloaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to reload strategy: {e}")

def setup_watchdog(strategy):
    """Setup file system monitoring for code changes"""
    event_handler = CodeChangeHandler(strategy)
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()
    return observer

def tail_log_files(n=30):
    """Tail the last n lines of all log files and fix common issues"""
    log_files = glob.glob("*.log")
    if not log_files:
        print(f"\n{'='*50}")
        print("⚠️ No log files found. Creating new log file...")
        print(f"{'='*50}")
        logging.basicConfig(
            filename='trading.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.info("🔄 New log file created")
        log_files = ['trading.log']

    for log_file in log_files:
        print(f"\n{'='*50}")
        print(f"📋 Last {n} lines of {log_file}:")
        print(f"{'='*50}")
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if not lines:
                    print(f"⚠️ {log_file} is empty")
                continue
                    
                # Show last n lines
                for line in lines[-n:]:
                    # Fix common log issues
                    line = line.strip()
                    if "error" in line.lower():
                        print(f"❌ {line}")
                    elif "warning" in line.lower():
                        print(f"⚠️ {line}")
                    elif "info" in line.lower():
                        print(f"ℹ️ {line}")
                    elif "debug" in line.lower():
                        print(f"🔍 {line}")
                    else:
                        print(line)
                        
                # Check for common issues
                error_count = sum(1 for line in lines if "error" in line.lower())
                warning_count = sum(1 for line in lines if "warning" in line.lower())
                
                if error_count > 0 or warning_count > 0:
                    print(f"\n{'='*50}")
                    print(f"📊 Log Analysis:")
                    print(f"❌ Errors: {error_count}")
                    print(f"⚠️ Warnings: {warning_count}")
                    
                    # Fix common issues
                    if "connection" in ' '.join(lines[-n:]).lower():
                        print("\n🔄 Attempting to fix connection issues...")
                        # Add reconnection logic here
                        
                    if "api" in ' '.join(lines[-n:]).lower():
                        print("\n🔑 Checking API configuration...")
                        # Add API validation logic here
                        
        except Exception as e:
            print(f"❌ Could not read {log_file}: {e}")
            # Try to fix the log file
            try:
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"Log file reset at {datetime.now()}\n")
                print(f"✅ Reset {log_file}")
            except:
                print(f"❌ Could not reset {log_file}")
                
        print(f"{'='*50}\n")

def check_dependencies():
    """Check and install required packages"""
    required_packages = {
        'ccxt': '4.0.0',
        'pandas': '1.5.0',
        'numpy': '1.21.0',
        'websockets': '10.0',
        'tqdm': '4.65.0',
        'prettytable': '3.0.0',
        'psutil': '5.9.0',
        'ratelimit': '2.2.1',
        'watchdog': '3.0.0'
    }
    
    import pkg_resources
    import subprocess
    import sys
    
    for package, version in required_packages.items():
        try:
            pkg_resources.require(f"{package}>={version}")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            print(f"Installing {package}>={version}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}>={version}"])
            print(f"✅ {package} installed successfully")

async def main():
    """Main function to run the strategy"""
    try:
        # Check and install dependencies
        check_dependencies()
        
        # Tail and fix logs first
        tail_log_files(30)
        
        # Initialize strategy
        strategy = EnhancedSuperTrend()
        logger.info("✅ Strategy initialized successfully")
        
        # Setup watchdog
        observer = setup_watchdog(strategy)
        logger.info("👀 Watchdog monitoring enabled")
        
        # Start performance monitoring in background
        monitoring_thread = threading.Thread(target=strategy.monitor_performance, daemon=True)
        monitoring_thread.start()
        
        # Setup WebSocket connections
        await strategy.setup_websocket_connections()
        logger.info("🔌 WebSocket connections established")
        
        # Run strategy with monitoring
        logger.info("🚀 Starting strategy execution...")
        await strategy.run_strategy()
        
    except Exception as e:
        logger.exception(f"❌ Main error: {e}")
    finally:
        if 'observer' in locals():
            observer.stop()
            observer.join()

if __name__ == "__main__":
    asyncio.run(main())
 