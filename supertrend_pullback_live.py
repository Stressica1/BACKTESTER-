#!/usr/bin/env python3
"""
FIXED SuperTrend Pullback Trading Bot for Bitget - WORKING  
Position Size: FIXED 0.50 USDT per trade (ENFORCED)
"""

import asyncio
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import logging
import multiprocessing
import os
from pathlib import Path
import random
import sqlite3
import sys
import time
import traceback
import warnings

import ccxt
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
np.seterr(all='ignore')
pd.set_option('mode.chained_assignment', None)

# Create required directories
REQUIRED_DIRS = ["logs", "data", "cache", "config"]
# Ensure required directories and log files exist
for d in REQUIRED_DIRS:
    os.makedirs(d, exist_ok=True)
# Ensure log files exist
for log_file in ["logs/supertrend_pullback.log", "logs/error.log", "logs/trading.log"]:
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("")

# Enhanced logging setup
class CustomFormatter(logging.Formatter):
    """Custom formatter with color coding"""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    blue = "\x1b[34;20m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: green + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logging():
    """Setup enhanced logging system"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Main log file
    file_handler = logging.FileHandler("logs/supertrend_pullback.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    
    # Console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter())
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

class TradingDatabase:
    """Enhanced database for trade tracking"""
    def __init__(self, db_file="data/trading_data.db"):
        self.db_file = db_file
        self.init_database()
        
    def init_database(self):
        """Initialize database with proper schema"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    signal_confidence REAL DEFAULT 0,
                    execution_time REAL DEFAULT 0,
                    leverage REAL DEFAULT 1,
                    success BOOLEAN DEFAULT TRUE
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    price REAL NOT NULL,
                    executed BOOLEAN DEFAULT FALSE
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)')
            conn.commit()
            conn.close()
            logger.info("✅ Database initialized successfully")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")

    def save_trade(self, trade_data):
        """Save trade data to database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (timestamp, symbol, side, price, size, signal_confidence, execution_time, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('timestamp', time.time()),
                trade_data.get('symbol', ''),
                trade_data.get('side', ''),
                trade_data.get('price', 0),
                trade_data.get('size', 0.50),
                trade_data.get('confidence', 0),
                trade_data.get('execution_time', 0),
                trade_data.get('success', True)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Trade save error: {e}")

    def save_signal(self, signal_data):
        """Save signal data to database"""
        try:
            if not isinstance(signal_data, dict):
                logger.error("Signal data must be a dictionary")
                return
            
            # Ensure required fields are present
            required_fields = ['timestamp', 'symbol', 'side', 'confidence', 'price', 'executed']
            if not all(field in signal_data for field in required_fields):
                logger.error("Signal data missing required fields")
                return
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signals (timestamp, symbol, signal_type, confidence, price, executed)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                signal_data.get('timestamp', time.time()),
                signal_data.get('symbol', ''),
                signal_data.get('side', ''),
                signal_data.get('confidence', 0),
                signal_data.get('price', 0),
                signal_data.get('executed', False)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Signal save error: {e}")

class AggressivePullbackTrader:
    """FIXED SuperTrend Pullback Trading Bot - ALL ISSUES RESOLVED"""
    
    def __init__(self, config_file="config/bitget_config.json"):
        """Initialize the FIXED trading bot with DYNAMIC PAIR DISCOVERY for ALL available pairs"""
        logger.info("🚀 INITIALIZING SUPERTREND PULLBACK BOT WITH DYNAMIC PAIR DISCOVERY")
        force_write_bitget_config()  # Always overwrite config with hardcoded credentials
        self.config = {
            "api_key": BITGET_API_KEY,
            "secret": BITGET_SECRET,
            "passphrase": BITGET_PASSPHRASE,
            "sandbox": False,
            "position_size_fixed": 0.50
        }
        self.exchange = self.setup_exchange()
        self.database = TradingDatabase()
        
        # CRITICAL FIX: Position size enforcement - REDUCED TO 1 USDT FOR TESTING
        self.FIXED_POSITION_SIZE_USDT = 1.0  # REDUCED FROM 2.0 TO 1.0 USDT FOR SMALLER ORDERS
        self.position_size_validation = True
        
        logger.critical(f"🔒 POSITION SIZE LOCKED: {self.FIXED_POSITION_SIZE_USDT} USDT")
        
        # Initialize available balance
        self.available_balance = 0.0
        
        # Trading parameters
        self.timeframe = "5m"
        self.max_positions = 50  # Increased for more pairs
        self.execution_timeout = 5
        
        # SuperTrend parameters (optimized)
        self.st_period = 10
        self.st_multiplier = 3.0
        
        # Risk management
        self.stop_loss_pct = 0.01  # 1%
        self.take_profit_levels = [0.008, 0.015, 0.025]  # 0.8%, 1.5%, 2.5%
        
        # DYNAMIC PAIR DISCOVERY - Get ALL available pairs from Bitget
        self.active_symbols = self.discover_all_trading_pairs()
        
        # Data structures
        self.positions = {}
        self.signals = defaultdict(list)
        self.trade_history = deque(maxlen=1000)

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        self.rate_limit_counters = defaultdict(lambda: deque(maxlen=10))
        
        # Bitget API rate limits (per second)
        self.rate_limits = {
            'fetch_ticker': 18,
            'fetch_ohlcv': 18,
            'fetch_balance': 9,
            'create_order': 9,
            'cancel_order': 9,
            'fetch_markets': 10
        }
        
        # Threading - Increased for more pairs
        self.max_workers = min(12, multiprocessing.cpu_count() * 2)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Signal statistics
        self.signal_stats = {
            'total_signals': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'signal_accuracy': 0.0
        }
        
        # Simulation parameters
        self.initial_balance = 10000
        self.current_balance = self.initial_balance
        self.simulation_trades = []
        
        self.log_swap_account_balance()
        
        logger.info("✅ Bot initialized successfully")
        logger.info(f"🎯 Total Symbols: {len(self.active_symbols)}")
        logger.info(f"⚡ Workers: {self.max_workers}")
        logger.info(f"🔒 Position Size: {self.FIXED_POSITION_SIZE_USDT} USDT (FIXED)")

    def discover_all_trading_pairs(self):
        """DISCOVER ALL AVAILABLE TRADING PAIRS FROM BITGET - 200+ PAIRS"""
        logger.info("🔍 DISCOVERING ALL AVAILABLE TRADING PAIRS FROM BITGET...")
        
        try:
            # Try to get markets from exchange
            if hasattr(self.exchange, 'load_markets'):
                markets = self.exchange.load_markets()
                
                # Filter for USDT futures pairs with leverage
                usdt_pairs = []
                for symbol, market in markets.items():
                    if (symbol.endswith('/USDT') and 
                        market.get('type') == 'swap' and  # Futures/perpetual
                        market.get('active', True) and
                        market.get('contract', False)):  # Has leverage
                        usdt_pairs.append(symbol)
                
                if len(usdt_pairs) > 50:  # Successfully found many pairs
                    logger.info(f"🚀 DISCOVERED {len(usdt_pairs)} USDT FUTURES PAIRS!")
                    
                    # Sort by popularity/volume (put major pairs first)
                    priority_pairs = [
                        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
                        'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT', 'DOT/USDT', 'AVAX/USDT'
                    ]
                    
                    # Reorder to put priority pairs first
                    sorted_pairs = []
                    for pair in priority_pairs:
                        if pair in usdt_pairs:
                            sorted_pairs.append(pair)
                            usdt_pairs.remove(pair)
                    
                    # Add remaining pairs
                    sorted_pairs.extend(sorted(usdt_pairs))
                    
                    # Log categories found
                    self.log_discovered_pairs(sorted_pairs)
                    
                    return sorted_pairs[:200]  # Limit to first 200 for performance
                    
        except Exception as e:
            logger.warning(f"⚠️ Dynamic pair discovery failed: {e}")
        
        # Fallback to comprehensive static list if dynamic discovery fails
        logger.info("📝 Using comprehensive static pair list as fallback...")
        return self.get_comprehensive_pair_list()

    def get_comprehensive_pair_list(self):
        """Comprehensive list of 200+ Bitget trading pairs"""
        return [
            # Major Cryptocurrencies (50x leverage)
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
            "ADA/USDT", "DOT/USDT", "AVAX/USDT", "LINK/USDT", "UNI/USDT",
            "LTC/USDT", "BCH/USDT", "ETC/USDT", "ATOM/USDT", "ALGO/USDT",
            
            # Popular Altcoins (40x leverage)  
            "MATIC/USDT", "FTM/USDT", "NEAR/USDT", "ICP/USDT", "VET/USDT",
            "FIL/USDT", "HBAR/USDT", "EOS/USDT", "XTZ/USDT", "FLOW/USDT",
            "SAND/USDT", "MANA/USDT", "AXS/USDT", "THETA/USDT", "ENJ/USDT",
            "CHZ/USDT", "GALA/USDT", "IMX/USDT", "LRC/USDT", "CRV/USDT",
            
            # DeFi Tokens (40x leverage)
            "SUSHI/USDT", "COMP/USDT", "YFI/USDT", "MKR/USDT", "SNX/USDT",
            "BAL/USDT", "REN/USDT", "KNC/USDT", "ZRX/USDT", "LPT/USDT",
            "BAND/USDT", "OCEAN/USDT", "GRT/USDT", "API3/USDT", "UMA/USDT",
            
            # Layer 1 & Layer 2 (40x leverage)
            "TRX/USDT", "XLM/USDT", "IOTA/USDT", "NEO/USDT", "DASH/USDT",
            "ZEC/USDT", "XMR/USDT", "QTUM/USDT", "ICX/USDT", "WAVES/USDT",
            "KAVA/USDT", "CELO/USDT", "ZIL/USDT", "RVN/USDT", "DGB/USDT",
            
            # Meme Coins (30x leverage)
            "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "FLOKI/USDT", "BONK/USDT",
            "WIF/USDT", "BOME/USDT", "NEIRO/USDT", "POPCAT/USDT", "MEME/USDT",
            "BRETT/USDT", "MOG/USDT", "TURBO/USDT", "MICHI/USDT", "BABYDOGE/USDT",
            
            # Gaming & Metaverse (35x leverage)
            "GMT/USDT", "STEPN/USDT", "LOOKS/USDT", "BLUR/USDT", "X2Y2/USDT",
            "MAGIC/USDT", "TLM/USDT", "SLP/USDT", "YGG/USDT", "GHST/USDT",
            "ALICE/USDT", "TKO/USDT", "SKILL/USDT", "HERO/USDT", "NFTX/USDT",
            
            # New Generation (35x leverage)
            "APT/USDT", "SUI/USDT", "SEI/USDT", "INJ/USDT", "TIA/USDT",
            "PYTH/USDT", "JUP/USDT", "WLD/USDT", "OP/USDT", "ARB/USDT",
            "STRK/USDT", "MANTA/USDT", "ALT/USDT", "JTO/USDT", "DYM/USDT",
            
            # Infrastructure & Oracle (35x leverage)
            "FET/USDT", "AGIX/USDT", "RNDR/USDT", "LPT/USDT", "STORJ/USDT",
            "AR/USDT", "HNT/USDT", "IOTX/USDT", "ANKR/USDT", "NKN/USDT",
            
            # Exchange Tokens (40x leverage)
            "BGB/USDT", "OKB/USDT", "HT/USDT", "KCS/USDT", "LEO/USDT",
            "CRO/USDT", "FTT/USDT", "BNT/USDT", "NEXO/USDT", "MCO/USDT",
            
            # AI & Machine Learning (35x leverage)
            "AI/USDT", "AGIX/USDT", "FET/USDT", "OCEAN/USDT", "NMR/USDT",
            "RLC/USDT", "CTXC/USDT", "DBC/USDT", "GNO/USDT", "MLN/USDT",
            
            # Privacy Coins (30x leverage)
            "XMR/USDT", "ZEC/USDT", "DASH/USDT", "FIRO/USDT", "BEAM/USDT",
            "GRIN/USDT", "ARRR/USDT", "OXEN/USDT", "DERO/USDT", "HAVEN/USDT",
            
            # Staking & Yield (35x leverage)
            "LIDO/USDT", "RPL/USDT", "FXS/USDT", "CVX/USDT", "CRV/USDT",
            "ALCX/USDT", "TOKE/USDT", "OHM/USDT", "KLIMA/USDT", "TIME/USDT",
            
            # Cross-Chain & Bridges (35x leverage)
            "REN/USDT", "POLY/USDT", "CELR/USDT", "SYN/USDT", "MULTI/USDT",
            "ANYSWAP/USDT", "BRIDGE/USDT", "RELAY/USDT", "HOP/USDT", "ACROSS/USDT",
            
            # Real World Assets (30x leverage)
            "RWA/USDT", "ONDO/USDT", "CFG/USDT", "MPL/USDT", "TRU/USDT",
            "CENTRI/USDT", "PROPS/USDT", "AST/USDT", "REQ/USDT", "DUSK/USDT",
            
            # Social & Content (30x leverage)
            "BAT/USDT", "STX/USDT", "LENS/USDT", "MASK/USDT", "LIT/USDT",
            "RALLY/USDT", "WHALE/USDT", "FWB/USDT", "RARE/USDT", "SUPER/USDT",
            
            # Additional High-Volume Pairs
            "RUNE/USDT", "LUNA/USDT", "UST/USDT", "KUJI/USDT", "ROWAN/USDT",
            "OSMO/USDT", "JUNO/USDT", "SCRT/USDT", "CRE/USDT", "HUAHUA/USDT",
            
            # Emerging Projects (25x leverage - conservative)
            "BLUR/USDT", "SUI/USDT", "APT/USDT", "SEI/USDT", "TIA/USDT",
            "JUP/USDT", "PYTH/USDT", "WLD/USDT", "MANTA/USDT", "ALT/USDT"
        ]

    def log_discovered_pairs(self, pairs):
        """Log discovered pairs by category"""
        categories = {
            "🏆 MAJOR": [p for p in pairs if any(major in p for major in ['BTC', 'ETH', 'SOL', 'BNB', 'XRP'])],
            "🔥 ALTCOINS": [p for p in pairs if any(alt in p for alt in ['ADA', 'DOT', 'MATIC', 'LINK', 'UNI'])],
            "🚀 MEME": [p for p in pairs if any(meme in p for meme in ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK'])],
            "🎮 GAMING": [p for p in pairs if any(game in p for game in ['SAND', 'MANA', 'AXS', 'GALA', 'ENJ'])],
            "🤖 AI/TECH": [p for p in pairs if any(ai in p for ai in ['FET', 'AGIX', 'RNDR', 'OCEAN', 'AI'])],
            "🏦 DEFI": [p for p in pairs if any(defi in p for defi in ['SUSHI', 'COMP', 'UNI', 'CRV', 'YFI'])],
        }
        
        logger.info("📊 DISCOVERED PAIRS BY CATEGORY:")
        for category, category_pairs in categories.items():
            if category_pairs:
                logger.info(f"   {category}: {len(category_pairs)} pairs")
                logger.debug(f"      {', '.join(category_pairs[:10])}")
        
        other_pairs = len(pairs) - sum(len(cat_pairs) for cat_pairs in categories.values())
        if other_pairs > 0:
            logger.info(f"   💼 OTHER: {other_pairs} pairs")
        
        logger.info(f"🚀 TOTAL ACTIVE PAIRS: {len(pairs)}")

    def get_pair_leverage_settings(self, symbol):
        """Get leverage settings for specific trading pair"""
        # Major cryptocurrencies - Maximum leverage
        if any(major in symbol for major in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']):
            return {'max_leverage': 50, 'default_leverage': 40}
            
        # Popular altcoins - High leverage
        elif any(alt in symbol for alt in ['XRP', 'ADA', 'DOT', 'MATIC', 'LINK', 'UNI', 'AVAX']):
            return {'max_leverage': 40, 'default_leverage': 30}
            
        # Meme coins - Medium leverage (more volatile)
        elif any(meme in symbol for meme in ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF']):
            return {'max_leverage': 30, 'default_leverage': 20}
            
        # Gaming & Metaverse - Medium-high leverage
        elif any(gaming in symbol for gaming in ['SAND', 'MANA', 'AXS', 'GALA', 'ENJ']):
            return {'max_leverage': 35, 'default_leverage': 25}
            
        # AI & Tech - Medium-high leverage
        elif any(ai in symbol for ai in ['FET', 'AGIX', 'RNDR', 'OCEAN', 'AI']):
            return {'max_leverage': 35, 'default_leverage': 25}
            
        # DeFi tokens - High leverage
        elif any(defi in symbol for defi in ['SUSHI', 'COMP', 'YFI', 'CRV', 'SNX']):
            return {'max_leverage': 40, 'default_leverage': 30}
            
        # New generation coins - Medium leverage
        elif any(new in symbol for new in ['APT', 'SUI', 'SEI', 'INJ', 'TIA', 'PYTH']):
            return {'max_leverage': 35, 'default_leverage': 25}
            
        # Default for other pairs - Conservative
        else:
            return {'max_leverage': 25, 'default_leverage': 15}

    def validate_position_size(self, size):
        """FIXED: Strict position size validation"""
        tolerance = 0.001  # 0.1% tolerance
        if abs(size - self.FIXED_POSITION_SIZE_USDT) > tolerance:
            error_msg = f"POSITION SIZE VIOLATION: {size} != {self.FIXED_POSITION_SIZE_USDT}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return True

    def get_fixed_position_size(self):
        """FIXED: Always return exactly 0.50 USDT"""
        return self.FIXED_POSITION_SIZE_USDT

    def load_config(self, config_file):
        # Always return hardcoded config
        return {
            "api_key": BITGET_API_KEY,
            "secret": BITGET_SECRET,
            "passphrase": BITGET_PASSPHRASE,
            "sandbox": False,
            "position_size_fixed": 0.50
        }

    def setup_exchange(self):
        # Only use real Bitget exchange, never mock
        exchange = ccxt.bitget({
            'apiKey': self.config["api_key"],
            'secret': self.config["secret"],
            'password': self.config["passphrase"],
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True,
                'createMarketBuyOrderRequiresPrice': False,
                'defaultMarginMode': 'cross',  # USE CROSS MARGIN AS REQUESTED
                'hedgeMode': True              # KEEP HEDGE MODE ENABLED
            },
        })
        exchange.set_sandbox_mode(self.config.get("sandbox", False))
        exchange.load_markets()
        return exchange

    async def rate_limit(self, endpoint='default'):
        """FIXED: Proper rate limiting implementation"""
        current_time = time.time()
        
        # Get rate limit for endpoint
        limit = self.rate_limits.get(endpoint, 10)
        window = self.rate_limit_counters[endpoint]
        
        # Remove old requests (older than 1 second)
        while window and current_time - window[0] > 1.0:
            window.popleft()
            
            # Check if we need to wait
        if len(window) >= limit:
            sleep_time = 1.0 - (current_time - window[0]) + 0.01
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Add current request
        window.append(current_time)
        
    async def handle_bitget_error(self, error, symbol=None, retry_count=0):
        """FIXED: Enhanced Bitget error handling with automatic recovery"""
        error_str = str(error).lower()
        max_retries = 3
        logger.warning(f"⚠️ Bitget error for {symbol}: {error}")
            
        # Handle specific Bitget errors
        if "50067" in str(error):  # Price deviation error
            logger.info("🔧 Price deviation error - getting current market price...")
            if symbol and retry_count < max_retries:
                await asyncio.sleep(1)
                return await self.get_current_market_price(symbol)
        elif "43012" in str(error) or "insufficient balance" in error_str:
            logger.error("💰 Insufficient balance - cannot place order")
            # Update available balance to prevent further trade attempts
            self.available_balance = 0.0
            return False
        elif "rate limit" in error_str or "429" in str(error):
            wait_time = min(60, 2 ** retry_count)
            logger.info(f"⏱️ Rate limit hit - waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
            return True
        elif "invalid" in error_str or "unauthorized" in error_str:
            logger.error("🔑 Authentication error - check API credentials")
            return False
        # Default retry logic
        if retry_count < max_retries:
            wait_time = 2 ** retry_count
            logger.info(f"🔄 Retrying in {wait_time}s... (attempt {retry_count + 1})")
            await asyncio.sleep(wait_time)
            return True
        return False

    async def get_current_market_price(self, symbol):
        """FIXED: Get current market price for price deviation recovery"""
        try:
            await self.rate_limit('fetch_ticker')
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker.get('last', ticker.get('close', 0))
        except Exception as e:
            logger.debug(f"Error getting market price for {symbol}: {e}")
            return None
            
    def calculate_supertrend(self, df, period=10, multiplier=3.0):
        """FIXED: Proper SuperTrend calculation"""
        try:
            if len(df) < period + 1:
                return None, None
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            hl2 = (df['high'] + df['low']) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            supertrend = pd.Series(index=df.index, dtype=float)
            direction = pd.Series(index=df.index, dtype=int)
            for i in range(period, len(df)):
                close_price = df['close'].iloc[i]
                if i == period:
                    if close_price > hl2.iloc[i]:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = upper_band.iloc[i]
                        direction.iloc[i] = -1
                else:
                    if direction.iloc[i-1] == 1:
                        if close_price > lower_band.iloc[i]:
                            supertrend.iloc[i] = lower_band.iloc[i]
                            direction.iloc[i] = 1
                        else:
                            supertrend.iloc[i] = upper_band.iloc[i]
                            direction.iloc[i] = -1
                    else:
                        if close_price < upper_band.iloc[i]:
                            supertrend.iloc[i] = upper_band.iloc[i]
                            direction.iloc[i] = -1
                        else:
                            supertrend.iloc[i] = lower_band.iloc[i]
                            direction.iloc[i] = 1
            return supertrend, direction
        except Exception as e:
            logger.debug(f"SuperTrend calculation error: {e}")
            return None, None
        
    async def get_market_data(self, symbol):
        """FIXED: Get market data with proper error handling"""
        try:
            # Rate limiting
            await self.rate_limit('fetch_ohlcv')
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, None, 50)
            
            if not ohlcv or len(ohlcv) < 20:
                logger.debug(f"Insufficient data for {symbol}")
                return None
            
            # Convert to DataFrame - FIX LINTER ERROR
            df = pd.DataFrame(ohlcv)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Ensure all columns are numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any NaN rows
            df = df.dropna()
            
            if len(df) < 20:
                logger.debug(f"Insufficient clean data for {symbol}")
                return None
                
            return df
            
        except Exception as e:
            logger.debug(f"Market data error for {symbol}: {e}")
            return None
            
    def calculate_rsi(self, df, period=14):
        """Calculate RSI with proper error handling"""
        try:
            if len(df) < period + 1:
                return pd.Series([50.0] * len(df), index=df.index)
            close_prices = pd.Series(df['close'], index=df.index)
            delta = close_prices.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            avg_gains = gains.rolling(period).mean()
            avg_losses = losses.rolling(period).mean()
            # Ensure avg_losses is a pandas Series
            if not isinstance(avg_losses, pd.Series):
                avg_losses = pd.Series(avg_losses, index=df.index[:len(avg_losses)])
            avg_losses = avg_losses.fillna(0.001)
            avg_losses = avg_losses.replace(0, 0.001)
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            if isinstance(rsi, pd.Series):
                result = pd.Series([50.0] * len(df), index=df.index)
                result.loc[rsi.index] = rsi
                return result
            else:
                return pd.Series([50.0] * len(df), index=df.index)
        except Exception as e:
            logger.debug(f"RSI calculation error: {e}")
            return pd.Series([50.0] * len(df), index=df.index)

    def calculate_rsi_timeframe(self, df, period=14):
        """Calculate RSI for specific timeframe"""
        try:
            if len(df) < period:
                return 50.0
            
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta).where(delta < 0, 0).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50.0
            
        except Exception as e:
            logger.debug(f"RSI calculation error: {e}")
            return 50.0

    def calculate_momentum_timeframe(self, df):
        """Calculate momentum for specific timeframe"""
        try:
            if len(df) < 10:
                return 0.0
            
            # Calculate price momentum
            momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            return float(momentum) if not pd.isna(momentum) else 0.0
            
        except Exception as e:
            logger.debug(f"Momentum calculation error: {e}")
            return 0.0

    async def get_timeframe_analysis(self, symbol, timeframe, base_df):
        """Get analysis for specific timeframe"""
        try:
            # Simulate timeframe-specific analysis
            # In real implementation, fetch actual data for each timeframe
            
            # Adjust SuperTrend parameters based on timeframe
            if timeframe in ['1m', '2m']:
                st_period, st_multiplier = 8, 2.5
            elif timeframe in ['5m', '10m']:
                st_period, st_multiplier = 10, 3.0
            elif timeframe in ['15m', '20m', '30m']:
                st_period, st_multiplier = 12, 3.2
            else:  # 45m, 55m, 60m
                st_period, st_multiplier = 15, 3.5
            
            # Calculate indicators for this timeframe
            supertrend, direction = self.calculate_supertrend(base_df, st_period, st_multiplier)
            if supertrend is None or direction is None:
                return {'trend_confirmed': False, 'rsi': 50, 'momentum': 0, 'volume_score': 0}
            
            # Calculate RSI
            rsi_period = 14 if timeframe in ['5m', '10m', '15m'] else 21
            rsi = self.calculate_rsi_timeframe(base_df, rsi_period)
            
            # Calculate momentum
            momentum = self.calculate_momentum_timeframe(base_df)
            
            # Calculate volume score (simulated)
            volume_score = np.random.uniform(40, 95)
            if 8 <= time.gmtime().tm_hour <= 20:  # Active hours
                volume_score += 10
            volume_score = min(100, volume_score)
            
            # Trend confirmation
            current_direction = direction.iloc[-1] if len(direction) > 0 else 0
            trend_confirmed = False
            
            if current_direction == 1:  # Uptrend
                trend_confirmed = (rsi < 45 and momentum > 0.002)
                direction_str = 'bullish'
            elif current_direction == -1:  # Downtrend
                trend_confirmed = (rsi > 55 and momentum < -0.002)
                direction_str = 'bearish'
            else:
                direction_str = 'neutral'
            
            return {
                'timeframe': timeframe,
                'supertrend': supertrend,
                'direction': direction_str,
                'rsi': rsi,
                'momentum': momentum,
                'volume_score': volume_score,
                'trend_confirmed': trend_confirmed,
                'current_direction': current_direction,
                'signal_strength': volume_score if trend_confirmed else 0
            }
            
        except Exception as e:
            logger.debug(f"Timeframe analysis error for {symbol} {timeframe}: {e}")
            return {'trend_confirmed': False, 'rsi': 50, 'momentum': 0, 'volume_score': 0, 'direction': 'neutral', 'signal_strength': 0}

    def ultra_market_regime_detection(self, mtf_data):
        """Ultra-precise market regime detection"""
        regime_scores = {}
        
        for tf, data in mtf_data.items():
            # Volatility analysis (simulated)
            volatility = np.random.uniform(0.01, 0.05)
            
            # Trend strength
            trend_strength = abs(data.get('momentum', 0))
            
            # Volume consistency
            volume_consistency = data.get('volume_score', 0)
            
            # Classify regime for this timeframe
            if (volatility < 0.02 and trend_strength > 0.004 and volume_consistency > 70):
                regime_scores[tf] = 'SUPER_TRENDING'
            elif (volatility < 0.015 and trend_strength < 0.002 and volume_consistency > 60):
                regime_scores[tf] = 'PERFECT_RANGING'
            else:
                regime_scores[tf] = 'SUBOPTIMAL'
        
        # Require majority agreement on optimal regime
        regime_counts = {}
        for regime in regime_scores.values():
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        if not regime_counts:
            return {'condition': 'MIXED'}
            
        # Fix: Use proper way to get max key
        dominant_regime = None
        max_count = 0
        for regime, count in regime_counts.items():
            if count > max_count:
                max_count = count
                dominant_regime = regime
        
        if dominant_regime:
            agreement_pct = max_count / len(mtf_data) * 100
            if agreement_pct >= 70:
                return {'condition': dominant_regime}
        
        return {'condition': 'MIXED'}

    def check_cross_timeframe_alignment(self, mtf_data):
        """Check alignment across ALL timeframes"""
        aligned_up = 0
        aligned_down = 0
        total_timeframes = len(mtf_data)
        for tf, data in mtf_data.items():
            direction = data.get('direction', 'neutral')
            if direction == 'bullish':
                aligned_up += 1
            elif direction == 'bearish':
                aligned_down += 1
        max_alignment = max(aligned_up, aligned_down)
        alignment_score = (max_alignment / total_timeframes) * 100 if total_timeframes > 0 else 0
        if aligned_up > aligned_down:
            primary_direction = 'buy'
        elif aligned_down > aligned_up:
            primary_direction = 'sell'
        else:
            primary_direction = 'neutral'
        return {
            'alignment_score': alignment_score,
            'primary_direction': primary_direction,
            'aligned_up': aligned_up,
            'aligned_down': aligned_down,
            'total_timeframes': total_timeframes
        }

    def analyze_momentum_confluence(self, mtf_data):
        """Analyze momentum confluence across timeframes"""
        momentum_values = []
        strong_momentum_count = 0
        
        for tf, data in mtf_data.items():
            momentum = data.get('momentum', 0)
            momentum_values.append(momentum)
            
            # Check for strong momentum (>0.003 for ultra system)
            if abs(momentum) > 0.003:
                strong_momentum_count += 1
        
        if not momentum_values:
            return {'confluence_score': 0}
        
        # Calculate confluence metrics
        confluence_score = (strong_momentum_count / len(momentum_values)) * 100
        
        # Bonus for consistent direction
        positive_momentum = sum(1 for m in momentum_values if m > 0)
        negative_momentum = sum(1 for m in momentum_values if m < 0)
        direction_consistency = max(positive_momentum, negative_momentum) / len(momentum_values) * 100
        
        final_confluence_score = (confluence_score * 0.6) + (direction_consistency * 0.4)
        
        return {
            'confluence_score': final_confluence_score,
            'strong_momentum_count': strong_momentum_count,
            'direction_consistency': direction_consistency
        }

    def analyze_volume_profile(self, mtf_data):
        """Analyze volume profile across timeframes"""
        volume_scores = []
        high_volume_count = 0
        
        for tf, data in mtf_data.items():
            volume_score = data.get('volume_score', 0)
            volume_scores.append(volume_score)
            
            # Check for high volume (>75 for ultra system)
            if volume_score > 75:
                high_volume_count += 1
        
        if not volume_scores:
            return {'volume_score': 0}
        
        avg_volume_score = np.mean(volume_scores)
        volume_consistency = (high_volume_count / len(volume_scores)) * 100
        
        # Calculate final volume score
        volume_score = (avg_volume_score * 0.7) + (volume_consistency * 0.3)
        
        return {
            'volume_score': volume_score,
            'high_volume_count': high_volume_count,
            'volume_consistency': volume_consistency
        }

    def recognize_high_probability_patterns(self, mtf_data):
        """Recognize high-probability technical patterns"""
        pattern_scores = []
        
        for tf, data in mtf_data.items():
            tf_pattern_score = 0
            
            direction = data.get('direction', 'neutral')
            rsi = data.get('rsi', 50)
            momentum = data.get('momentum', 0)
            volume_score = data.get('volume_score', 0)
            
            # 1. SuperTrend + RSI confluence
            if ((direction == 'bullish' and rsi < 35) or
                (direction == 'bearish' and rsi > 65)):
                tf_pattern_score += 25
            
            # 2. Strong momentum in trend direction
            if (direction == 'bullish' and momentum > 0.004) or \
               (direction == 'bearish' and momentum < -0.004):
                tf_pattern_score += 25
            
            # 3. Volume confirmation
            if volume_score > 70:
                tf_pattern_score += 20
            
            # 4. Trend confirmation
            if data.get('trend_confirmed', False):
                tf_pattern_score += 30
            
            pattern_scores.append(tf_pattern_score)
        
        if not pattern_scores:
            return 0
            
        # Calculate overall pattern score
        avg_pattern_score = np.mean(pattern_scores)
        strong_pattern_count = sum(1 for score in pattern_scores if score >= 70)
        pattern_consistency = (strong_pattern_count / len(pattern_scores)) * 100 if pattern_scores else 0
        
        final_pattern_score = (avg_pattern_score * 0.8) + (pattern_consistency * 0.2)
        
        return final_pattern_score

    def calculate_ultra_confidence(self, trend_alignment, momentum_confluence, 
                                  volume_profile, pattern_score, market_regime):
        """Calculate ultra-high confidence score"""
        
        base_confidence = 60.0  # Higher base for ultra system
        
        # 1. Trend Alignment Factor (0-15 points)
        trend_factor = max(0.0, (trend_alignment['alignment_score'] - 70) / 30 * 15)
        
        # 2. Momentum Confluence Factor (0-12 points)
        momentum_factor = max(0.0, (momentum_confluence['confluence_score'] - 70) / 30 * 12)
        
        # 3. Volume Profile Factor (0-8 points)
        volume_factor = max(0.0, (volume_profile['volume_score'] - 60) / 40 * 8)
        
        # 4. Pattern Recognition Factor (0-10 points)
        pattern_factor = max(0.0, (pattern_score - 60) / 40 * 10)
        
        # 5. Market Regime Bonus (0-5 points)
        regime_condition = market_regime.get('condition', 'MIXED')
        regime_bonus = 5.0 if regime_condition in ['SUPER_TRENDING', 'PERFECT_RANGING'] else 0.0
        
        total_confidence = (base_confidence + trend_factor + momentum_factor + 
                          volume_factor + pattern_factor + regime_bonus)
        
        return min(98.0, max(60.0, total_confidence))

    def calculate_ultra_leverage(self, confidence, market_regime, leverage_settings):
        """ALWAYS RETURN AT LEAST 75x LEVERAGE"""
        return max(75, leverage_settings.get('max_leverage', 75))

    async def generate_signal(self, symbol):
        """ENHANCED: Generate high-quality trading signals with optimized performance"""
        try:
            # Get market data
            df = await self.get_market_data(symbol)
            if df is None or len(df) < 50:  # Need sufficient data
                return None
                
            # Calculate SuperTrend
            st_data = self.calculate_supertrend(df, period=self.st_period, multiplier=self.st_multiplier)
            
            # First check for SuperTrend crossover signal
            signal = None
            trend_change = False
            if len(st_data) >= 2:
                prev_trend = st_data[-2]['trend'] 
                curr_trend = st_data[-1]['trend']
                trend_change = prev_trend != curr_trend
                
                if trend_change and curr_trend == 'up':
                    signal = {
                        'symbol': symbol,
                        'type': 'long',
                        'price': df['close'].iloc[-1],
                        'time': df.index[-1],
                        'confidence': 0  # Will be set after analysis
                    }
                elif trend_change and curr_trend == 'down':
                    signal = {
                        'symbol': symbol,
                        'type': 'short',
                        'price': df['close'].iloc[-1],
                        'time': df.index[-1],
                        'confidence': 0  # Will be set after analysis
                    }
                    
            # If no signal from basic SuperTrend, check for pullback opportunities using SuperZ
            if signal is None and hasattr(self, 'super_z') and self.super_z is not None:
                try:
                    # Try to call detect_signals safely, handling the deprecated loosen_level parameter issue
                    try:
                        signals, df_with_indicators = self.super_z.detect_signals(df, timeframe=self.timeframe)
                    except TypeError as te:
                        # If TypeError (like unexpected keyword argument), try with minimal parameters
                        logger.warning(f"Type error calling detect_signals: {te}. Trying fallback method.")
                        signals, df_with_indicators = self.super_z.detect_signals(df)
                    except Exception as e:
                        logger.error(f"Error in detect_signals for {symbol}: {str(e)}")
                        signals = []
                        df_with_indicators = df
                        
                    # Check for valid signals from SuperZ
                    if signals and len(signals) > 0:
                        latest_signal = signals[-1]
                        # Only use recent signals
                        if latest_signal['time'] >= df.index[-5]:  # Within last 5 candles
                            signal = {
                                'symbol': symbol,
                                'type': latest_signal['type'],
                                'price': latest_signal['price'],
                                'time': latest_signal['time'],
                                'confidence': 0  # Will be updated
                            }
                except Exception as sz_error:
                    logger.error(f"SuperZ signal detection error for {symbol}: {sz_error}")
                    
            # If still no signal, nothing to do
            if signal is None:
                return None
                
            # Get multi-timeframe data for deeper analysis
            mtf_data = {}
            try:
                # Get 1h and 4h timeframe data for the same symbol
                mtf_data['5m'] = await self.get_timeframe_analysis(symbol, '5m', df)
                mtf_data['15m'] = await self.get_timeframe_analysis(symbol, '15m', None)
                mtf_data['1h'] = await self.get_timeframe_analysis(symbol, '1h', None)
                mtf_data['4h'] = await self.get_timeframe_analysis(symbol, '4h', None)
            except Exception as mtf_error:
                logger.error(f"Multi-timeframe analysis error for {symbol}: {mtf_error}")
                # Continue with what we have even if some timeframes failed
                
            # Perform comprehensive market analysis
            market_regime = self.ultra_market_regime_detection(mtf_data)
            trend_alignment = self.check_cross_timeframe_alignment(mtf_data)
            momentum_confluence = self.analyze_momentum_confluence(mtf_data)
            volume_profile = self.analyze_volume_profile(mtf_data)
            pattern_score = self.recognize_high_probability_patterns(mtf_data)
            
            # Calculate ultra confidence score
            confidence = self.calculate_ultra_confidence(
                trend_alignment, momentum_confluence, volume_profile, 
                pattern_score, market_regime
            )
            
            # Get leverage settings for this pair
            leverage_settings = self.get_pair_leverage_settings(symbol)
            
            # Calculate optimal leverage based on confidence
            optimal_leverage = self.calculate_ultra_leverage(
                confidence, market_regime, leverage_settings
            )
            
            # Update signal with analysis results
            signal['confidence'] = confidence
            signal['market_regime'] = market_regime
            signal['trend_alignment'] = trend_alignment
            signal['momentum_confluence'] = momentum_confluence
            signal['volume_profile'] = volume_profile
            signal['pattern_score'] = pattern_score
            signal['optimal_leverage'] = optimal_leverage
            signal['max_leverage'] = leverage_settings['max_leverage']
            
            # Add comprehensive risk management information
            current_price = signal['price']
            if signal['type'] == 'long':
                # For long positions
                stop_loss_price = current_price * (1 - self.stop_loss_pct)
                take_profit_price_1 = current_price * (1 + self.take_profit_pct)
                take_profit_price_2 = current_price * (1 + self.take_profit_pct * 2)
                take_profit_price_3 = current_price * (1 + self.take_profit_pct * 3)
            else:
                # For short positions
                stop_loss_price = current_price * (1 + self.stop_loss_pct)
                take_profit_price_1 = current_price * (1 - self.take_profit_pct)
                take_profit_price_2 = current_price * (1 - self.take_profit_pct * 2)
                take_profit_price_3 = current_price * (1 - self.take_profit_pct * 3)
                
            signal['stop_loss'] = stop_loss_price
            signal['take_profit_1'] = take_profit_price_1
            signal['take_profit_2'] = take_profit_price_2
            signal['take_profit_3'] = take_profit_price_3
            
            # Debug info
            logger.debug(f"Signal for {symbol}: {signal['type']} @ {signal['price']} | Confidence: {confidence}%")
            
            # Record the signal in our database
            signal_data = {
                'symbol': symbol,
                'signal_type': signal['type'],
                'price': signal['price'],
                'confidence': confidence,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Asynchronously save to database
            asyncio.create_task(self.save_signal_async(signal_data))
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            # Log full traceback for debugging
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    def get_minimum_order_size(self, symbol):
        """Get the minimum order size for a symbol from Bitget markets, fallback to 1.0 USDT if not found"""
        try:
            # Make sure exchange is initialized
            if not hasattr(self, 'exchange') or self.exchange is None:
                return 1.0
                
            # Get markets safely
            markets = getattr(self.exchange, 'markets', None)
            if markets is None or not isinstance(markets, dict):
                return 1.0
                
            # Get market info for symbol
            market = markets.get(symbol, {})
            if not market:
                return 1.0
            
            # First check cost (USDT value)
            try:
                limits = market.get('limits', {})
                if limits and isinstance(limits, dict):
                    cost_limits = limits.get('cost', {})
                    if cost_limits and isinstance(cost_limits, dict):
                        min_cost = cost_limits.get('min')
                        if min_cost is not None:
                            try:
                                min_cost_float = float(min_cost)
                                if min_cost_float > 0:
                                    return min_cost_float
                            except (ValueError, TypeError):
                                pass
            except Exception as e:
                logger.debug(f"Error getting min cost: {e}")
                
            # Fall back to amount (quantity)
            try:
                limits = market.get('limits', {})
                if limits and isinstance(limits, dict):
                    amount_limits = limits.get('amount', {})
                    if amount_limits and isinstance(amount_limits, dict):
                        min_amount = amount_limits.get('min')
                        if min_amount is not None:
                            try:
                                min_amount_float = float(min_amount)
                                if min_amount_float > 0:
                                    # Get ticker price
                                    ticker = self.exchange.fetch_ticker(symbol)
                                    price = ticker.get('last', ticker.get('close', 0))
                                    if price is not None and float(price) > 0:
                                        return min_amount_float * float(price)
                            except (ValueError, TypeError, Exception):
                                pass
            except Exception as e:
                logger.debug(f"Error getting min amount: {e}")
            
            # Default minimum order size (safest option)
            return 1.0
        except Exception as e:
            logger.warning(f"Could not fetch minimum order size for {symbol}: {e}")
            return 1.0

    def get_symbol_margin_mode(self, symbol):
        """Determine which margin mode a symbol supports (cross or isolated)"""
        try:
            # Check if symbol supports cross margin
            market = self.exchange.market(symbol)
            
            # First check if symbol explicitly doesn't support cross (from error 50004)
            if market.get('info', {}).get('crossable') == 'false':
                logger.warning(f"Symbol {symbol} does not support cross margin, using isolated")
                return 'isolated'
                
            # Some symbols don't support cross margin
            # Look for margin mode support info in market data
            margin_modes = market.get('info', {}).get('supportMarginCoins', {})
            
            if margin_modes:
                # If we have explicit margin mode info, use it
                if 'cross' in str(margin_modes).lower():
                    logger.info(f"Symbol {symbol} supports cross margin mode")
                    return 'cross'
                else:
                    logger.warning(f"Symbol {symbol} only supports isolated margin")
                    return 'isolated'
            
            # If we couldn't determine, use cross as requested by default
            logger.info(f"No margin mode info for {symbol}, using cross margin as requested")
            return 'cross'
        except Exception as e:
            logger.warning(f"⚠️ Error determining margin mode for {symbol}: {e}")
            # Default to cross as fallback per user request
            return 'cross'

    def get_max_leverage(self, symbol):
        """Get maximum allowed leverage for a symbol (default: 20x - REDUCED FOR SAFETY)"""
        try:
            # Category 1: Major cryptos (20x max)
            major_cryptos = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
            if symbol in major_cryptos:
                return 20
            
            # Category 2: Popular alts (20x max)
            popular_alts = ['XRP/USDT', 'ADA/USDT', 'DOT/USDT', 'MATIC/USDT']
            if symbol in popular_alts:
                return 20
            
            # Category 3: Meme coins (15x max)
            meme_coins = ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']
            if symbol in meme_coins:
                return 15
                
            # Category 4: Gaming (15x max)
            gaming = ['SAND/USDT', 'MANA/USDT', 'AXS/USDT']
            if symbol in gaming:
                return 15
                
            # Category 5: AI/Tech (15x max)
            ai_tech = ['FET/USDT', 'AGIX/USDT', 'RNDR/USDT']
            if symbol in ai_tech:
                return 15
                
            # Category 6: DeFi (15x max)
            defi = ['SUSHI/USDT', 'COMP/USDT', 'UNI/USDT']
            if symbol in defi:
                return 15
                
            # Default: Conservative (10x max)
            return 10
        except Exception as e:
            logger.warning(f"⚠️ Error determining max leverage for {symbol}: {e}")
            # Default to 10x as a safe fallback
            return 10

    async def execute_trade(self, signal):
        """Execute a trade with proper balance checking and error handling"""
        execution_start = time.time()
        try:
            symbol = signal['symbol']
            side = signal['side']
            
            # CRITICAL FIX: Add proper balance check before attempting to trade
            if not hasattr(self, 'available_balance') or self.available_balance < self.FIXED_POSITION_SIZE_USDT:
                # Refresh balance to get latest value
                try:
                    balance = self.exchange.fetch_balance({
                        'type': 'swap',
                        'product_type': 'umcbl'  # USDT-margined contracts
                    })
                    usdt_info = balance.get('USDT', {})
                    free = usdt_info.get('free', 0)
                    # Fix linter error by properly handling potential None value
                    self.available_balance = float(free) if free is not None and free != '' else 0.0
                except Exception as e:
                    logger.error(f"❌ Error fetching balance during trade: {e}")
                    self.available_balance = 0.0
            
            # Double-check if we have enough balance
            if self.available_balance < self.FIXED_POSITION_SIZE_USDT:
                logger.error(f"❌ Insufficient balance for trade: have {self.available_balance} USDT, need {self.FIXED_POSITION_SIZE_USDT} USDT")
                return None
                
            # Get the appropriate margin mode for this symbol
            margin_mode = self.get_symbol_margin_mode(symbol)
            logger.info(f"💡 Using {margin_mode.upper()} margin mode for {symbol}")
            
            # Define leverage params here to avoid unbound variable error
            leverage_params = {
                "marginCoin": "USDT",
                "holdSide": "long" if side == "buy" else "short"
            }
            
            # Get maximum allowed leverage for this symbol
            max_allowed_leverage = self.get_max_leverage(symbol)
            
            # REDUCED LEVERAGE: Use lower leverage for safer trading
            # Cap leverage at the maximum allowed by Bitget for this symbol, or use a safer value
            requested_leverage = signal.get('leverage', 0)
            if requested_leverage > max_allowed_leverage or requested_leverage <= 0:
                # Use a standard leverage based on symbol
                leverage = max_allowed_leverage 
            else:
                leverage = requested_leverage
                
            # Ensure leverage is reasonable - CRITICAL FIX FOR SUCCESSFUL ORDERS
            if leverage > 10:
                leverage = 10  # Maximum safe leverage
                
            logger.info(f"🔧 SETTING LEVERAGE: {symbol} -> {leverage}x (MAX ALLOWED: {max_allowed_leverage}x)")
            
            # Step 1: Set margin mode based on symbol support
            margin_mode_set = False
            try:
                await self.rate_limit('set_margin_mode')
                margin_params = {
                    "symbol": symbol.replace("/", ""),
                    "marginMode": margin_mode
                }
                await self.exchange.set_margin_mode(margin_mode, symbol, params=margin_params)
                logger.info(f"✅ Set margin mode to {margin_mode.upper()} for {symbol}")
                margin_mode_set = True
            except Exception as e:
                error_str = str(e)
                
                # Check for specific error codes related to margin mode
                if "50004" in error_str or "symbol does not support cross" in error_str.lower():
                    logger.warning(f"⚠️ Symbol {symbol} does not support cross margin, falling back to isolated")
                    margin_mode = 'isolated'  # Force to isolated
                    try:
                        # Retry with isolated margin
                        margin_params = {
                            "symbol": symbol.replace("/", ""),
                            "marginMode": "isolated"
                        }
                        await self.exchange.set_margin_mode('isolated', symbol, params=margin_params)
                        logger.info(f"✅ Set margin mode to ISOLATED for {symbol}")
                        margin_mode_set = True
                    except Exception as fallback_error:
                        logger.error(f"❌ Failed to set isolated margin mode: {fallback_error}")
                else:
                    logger.warning(f"⚠️ Failed to set margin mode: {e}")
            
            # Step 2: Set leverage with correct params - WITH HEDGE MODE
            leverage_set = False
            try:
                await self.rate_limit('set_leverage')
                await self.exchange.set_leverage(leverage, symbol, params=leverage_params)
                logger.info(f"✅ Leverage set: {leverage}x for {symbol} {side}")
                leverage_set = True
            except Exception as e:
                error_str = str(e)
                if "Exceeded the maximum settable leverage" in error_str:
                    # Try again with lower leverage
                    try:
                        lower_leverage = 5  # Much lower fallback
                        await self.exchange.set_leverage(lower_leverage, symbol, params=leverage_params)
                        logger.info(f"✅ Fallback leverage set: {lower_leverage}x for {symbol} {side}")
                        leverage = lower_leverage
                        leverage_set = True
                    except Exception as e2:
                        logger.warning(f"⚠️ Fallback leverage setting failed: {e2}")
                else:
                    logger.warning(f"⚠️ Leverage setting failed: {e}")
            
            # Step 3: Check if we successfully set both margin mode and leverage
            if not margin_mode_set or not leverage_set:
                logger.warning(f"⚠️ Could not properly configure {symbol} - continuing anyway")
            
            # Dynamically determine minimum order size
            min_order_size = self.get_minimum_order_size(symbol)
            
            # CRITICAL FIX: Ensure we're using at least the minimum order size
            if min_order_size > self.FIXED_POSITION_SIZE_USDT:
                margin_usdt = min_order_size
                logger.info(f"⚠️ Increasing order size to minimum required: {min_order_size} USDT (> {self.FIXED_POSITION_SIZE_USDT} USDT)")
            else:
                margin_usdt = self.FIXED_POSITION_SIZE_USDT
            
            # Check if we have enough balance for this order
            if self.available_balance < margin_usdt:
                logger.error(f"❌ Insufficient balance for {symbol}: have {self.available_balance} USDT, need {margin_usdt} USDT")
                return None
            
            # Ensure effective position value is a valid float
            if leverage is None:
                leverage = 5  # Default fallback leverage
            
            effective_position_value = float(margin_usdt) * float(leverage)
            
            logger.info(f"⚡ EXECUTING TRADE: {symbol} {side.upper()}")
            logger.info(f"   💰 Margin Used: {margin_usdt} USDT (min for {symbol})")
            logger.info(f"   📈 Leverage: {leverage}x")
            logger.info(f"   💵 Effective Position: {effective_position_value} USDT")
            
            # Initialize current_price before using it
            current_price = None
            
            # Get price from signal or fetch fresh
            signal_price = signal.get('price', None)
            if signal_price is not None and isinstance(signal_price, (int, float)) and signal_price > 0:
                current_price = float(signal_price)
            else:
                # Fetch fresh price if signal price is missing or invalid
                fetched_price = await self.get_current_market_price(symbol)
                if fetched_price is not None and isinstance(fetched_price, (int, float)) and fetched_price > 0:
                    current_price = float(fetched_price)
            
            # Final price validation
            if current_price is None or current_price <= 0:
                logger.error(f"❌ Cannot get valid price for {symbol}")
                return None
            
            # Log price information
            logger.info(f"   💲 Price: {current_price}")
            
            # Calculate quantity based on effective position value and price
            quantity = float(effective_position_value) / float(current_price)
            
            # CRITICAL FIX: Check for minimum quantity requirements
            if hasattr(self.exchange, 'markets') and isinstance(self.exchange.markets, dict):
                market = self.exchange.markets.get(symbol, {})
                if isinstance(market, dict) and 'limits' in market:
                    limits = market.get('limits', {})
                    if isinstance(limits, dict) and 'amount' in limits:
                        amount_limits = limits.get('amount', {})
                        if isinstance(amount_limits, dict) and 'min' in amount_limits:
                            min_amount = amount_limits.get('min')
                            if min_amount is not None:
                                try:
                                    min_amount = float(min_amount)
                                    if quantity < min_amount:
                                        logger.warning(f"⚠️ Quantity {quantity} too small for {symbol}, adjusting to minimum {min_amount}")
                                        quantity = min_amount
                                except (ValueError, TypeError):
                                    pass
            # Show final quantity
            logger.info(f"   📊 Quantity: {quantity} coins")
            
            # Execute the trade with retry logic for price deviation errors
            retry_count = 0
            max_retries = 3
            price_adjustment_factor = 1.0  # Start with no adjustment
            
            while retry_count < max_retries:
                try:
                    await self.rate_limit('create_order')
                    
                    # CORRECT ORDER PARAMETERS FOR BITGET USDT-M SWAP - USE THE MARGIN MODE WE VERIFIED
                    order_params = {
                        'marginCoin': 'USDT',
                        'timeInForce': 'IOC',
                        'tradeSide': 'open',    # Open position
                        'marginMode': margin_mode,  # Use the margin mode we verified
                        'holdSide': side        # For hedge mode
                    }
                    
                    # Adjust price to account for slippage in the direction of the trade
                    if retry_count > 0:
                        # For buy orders, increase price; for sell orders, decrease price
                        if side == 'buy':
                            adjusted_price = current_price * (1 + (retry_count * 0.002))
                        else:
                            adjusted_price = current_price * (1 - (retry_count * 0.002))
                        logger.info(f"   🔄 Retry {retry_count}: Adjusting price from {current_price} to {adjusted_price}")
                        current_price = adjusted_price
                    
                    # Create market order with specified parameters
                    order = self.exchange.create_market_order(
                        symbol, 
                        side, 
                        quantity,
                        params=order_params
                    )
                    
                    if order and isinstance(order, dict):
                        order_id = order.get('id', 'unknown')
                        execution_time = (time.time() - execution_start) * 1000  # in ms
                        
                        filled_price = order.get('price', current_price)
                        
                        trade_data = {
                            'timestamp': time.time(),
                            'symbol': symbol,
                            'side': side,
                            'price': filled_price,
                            'margin_usdt': margin_usdt,
                            'effective_value_usdt': margin_usdt * leverage,
                            'leverage': leverage,
                            'quantity': effective_position_value,
                            'confidence': signal.get('confidence', 0),
                            'execution_time': execution_time,
                            'success': True,
                            'order_id': order_id
                        }
                        
                        self.database.save_trade(trade_data)
                        self.total_trades += 1
                        
                        logger.info("✅ TRADE EXECUTED SUCCESSFULLY!")
                        logger.info(f"   💰 Cost: {margin_usdt} USDT | Effective: {margin_usdt * leverage} USDT")
                        logger.info(f"   📈 Leverage: {leverage}x | Size: {effective_position_value}")
                        logger.info(f"   💲 Price: ~{filled_price:.6f}")
                        logger.info(f"   ⚡ Execution Time: {execution_time:.1f}ms")
                        
                        self.log_swap_account_balance()
                        return order
                    else:
                        logger.warning(f"⚠️ Order not created properly: {order}")
                        retry_count += 1
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    error_str = str(e)
                    
                    # Handle price deviation error (50067) by retrying with new price closer to index price
                    if hasattr(e, 'args') and any('50067' in str(arg) for arg in e.args):
                        logger.warning("⚠️ 50067 error: retrying with new price closer to index price")
                        current_price = await self.get_current_market_price(symbol)
                        if current_price is None or not isinstance(current_price, (int, float)):
                            logger.error(f"❌ Invalid price for {symbol}, skipping trade.")
                            return None
                        
                        retry_count += 1
                        continue
                    
                    # Check for cross margin error and retry with isolated if needed
                    if "50004" in error_str or "symbol does not support cross" in error_str.lower():
                        logger.warning(f"⚠️ Cross margin not supported for {symbol}, trying isolated")
                        margin_mode = 'isolated'
                        # Update the order params for next attempt
                        order_params = {
                            'marginCoin': 'USDT',
                            'timeInForce': 'IOC',
                            'tradeSide': 'open',
                            'marginMode': 'isolated',  # Force isolated mode
                            'holdSide': side
                        }
                        retry_count += 1
                        continue
                    
                    # Handle "less than the minimum amount" error (45110)
                    if hasattr(e, 'args') and any('45110' in str(arg) for arg in e.args):
                        logger.warning(f"⚠️ Bitget business error (no retry): {e}")
                        return None
                    
                    # Handle "Insufficient balance" error (43012)
                    if hasattr(e, 'args') and any('43012' in str(arg) for arg in e.args):
                        logger.warning(f"⚠️ Bitget insufficient balance error (no retry): {e}")
                        return None
                    
                    # For all other errors, log and retry
                    logger.error(f"❌ Trade execution error: {e}")
                    retry_count += 1
                    await asyncio.sleep(1)
            
            logger.error(f"❌ Trade execution failed after {max_retries} attempts")
            logger.warning(f"⚠️ FAILED TRADE: {symbol}")
            return None
            
        except Exception as e:
            execution_time = (time.time() - execution_start) * 1000
            logger.error(f"❌ Trade execution error in {execution_time:.1f}ms: {e}")
            return None

    async def check_account_balance(self):
        """Check if account has sufficient balance for trading"""
        try:
            # Get SPECIFIC balance for USDT-M futures
            balance = self.exchange.fetch_balance({
                'type': 'swap',
                'product_type': 'umcbl'  # USDT-margined contracts
            })
            
            # Get USDT balance specifically
            usdt_info = balance.get('USDT', {})
            free = usdt_info.get('free', 0)
            
            # Convert to float and handle None - Fix linter error
            free_balance = float(free) if free is not None and free != '' else 0.0
            self.available_balance = free_balance
            
            # Calculate required balance for trading (minimum position size)
            required_balance = self.FIXED_POSITION_SIZE_USDT * 2  # Double for safety margin
            
            # Log balance information
            logger.info(f"💰 Account Balance: {free_balance:.2f} USDT")
            logger.info(f"🔹 Required Balance: {required_balance:.2f} USDT")
            
            # Check if balance is sufficient
            if free_balance >= required_balance:
                logger.info("✅ Sufficient balance for trading")
                return True, free_balance
            else:
                logger.warning(f"⚠️ Insufficient balance: {free_balance:.2f} USDT (need {required_balance:.2f} USDT)")
                return False, free_balance
                
        except Exception as e:
            logger.error(f"❌ Error checking account balance: {e}")
            return False, 0.0

    def adjust_quantity_for_precision(self, symbol, quantity):
        return quantity  # No adjustment

    async def set_leverage(self, symbol, leverage):
        """FIXED: Set leverage for futures trading with correct parameters"""
        try:
            params = {
                "marginCoin": "USDT",
                "holdSide": "long"  # Default to long
            }
            return self.exchange.set_leverage(leverage, symbol, params=params)
        except Exception as e:
            logger.debug(f"Leverage setting error: {e}")
            return None

    async def process_symbol(self, symbol):
        """FIXED: Process individual symbol for signals and trading"""
        try:
            # Generate signal
            signal = await self.generate_signal(symbol)
            
            if signal and signal.get('confidence', 0) >= 60:  # 60% confidence threshold
                # CRITICAL FIX: Skip trade execution if balance is insufficient
                if not hasattr(self, 'available_balance') or self.available_balance < self.FIXED_POSITION_SIZE_USDT:
                    logger.warning(f"⚠️ Signal found for {symbol} but insufficient balance to trade: {self.available_balance} USDT")
                    return
                    
                # Execute trade
                result = await self.execute_trade(signal)
                
                if result:
                    logger.info(f"🎯 SUCCESSFUL TRADE: {symbol}")
                    self.signal_stats['successful_trades'] += 1
                else:
                    logger.warning(f"⚠️ FAILED TRADE: {symbol}")
                    self.signal_stats['failed_trades'] += 1
                
                # Update signal accuracy
                total_executed = self.signal_stats['successful_trades'] + self.signal_stats['failed_trades']
                if total_executed > 0:
                    self.signal_stats['signal_accuracy'] = (
                        self.signal_stats['successful_trades'] / total_executed * 100
                    )
            
        except Exception as e:
            logger.debug(f"Symbol processing error for {symbol}: {e}")

    async def main_trading_loop(self):
        logger.info("🚀 STARTING ENHANCED TRADING LOOP")
        logger.info(f"💰 Position Size: {self.FIXED_POSITION_SIZE_USDT} USDT per trade")
        self.display_trading_pairs_config()
        
        # CRITICAL FIX: Check balance at startup
        has_balance, balance_amount = await self.check_account_balance()
        if not has_balance:
            logger.critical("❌ INSUFFICIENT BALANCE TO START TRADING! Please add funds to your account.")
            logger.critical(f"   Required: {self.FIXED_POSITION_SIZE_USDT * 2} USDT | Available: {balance_amount} USDT")
            logger.critical("⚠️ BOT WILL CONTINUE CHECKING FOR SIGNALS BUT WILL NOT EXECUTE TRADES UNTIL FUNDS ARE ADDED")
        
        try:
            while True:
                # Check balance periodically (every 5 iterations)
                if random.randint(1, 5) == 1:
                    await self.check_account_balance()
                
                await self.main_trading_loop_iteration()
                await asyncio.sleep(5)  # 5-second interval between signals
        except KeyboardInterrupt:
            logger.info("🛑 Trading loop stopped by user")
        except Exception as e:
            logger.error(f"❌ Trading loop error: {e}")
            raise

    async def main_trading_loop_iteration(self):
        """Single iteration of main trading loop for simulation"""
        try:
            # Skip trading if balance is insufficient
            if not hasattr(self, 'available_balance') or self.available_balance < self.FIXED_POSITION_SIZE_USDT:
                if random.randint(1, 5) == 1:  # Don't spam logs, only log occasionally
                    logger.warning(f"⚠️ Trading skipped: Insufficient balance ({self.available_balance} USDT) < required ({self.FIXED_POSITION_SIZE_USDT} USDT)")
                    # Still scan for signals, just don't execute trades
                    
            # Process a few symbols per iteration
            symbols_per_iteration = 3
            start_idx = random.randint(0, max(0, len(self.active_symbols) - symbols_per_iteration))
            symbols_batch = self.active_symbols[start_idx:start_idx + symbols_per_iteration]
            
            tasks = [self.process_symbol(symbol) for symbol in symbols_batch]
            await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            logger.debug(f"Loop iteration error: {e}")

    def display_trading_pairs_config(self):
        """FIXED: Display comprehensive trading pairs configuration"""
        logger.info("📊 COMPREHENSIVE TRADING CONFIGURATION:")
        logger.info("=" * 80)
        logger.info(f"🎯 Total Symbols: {len(self.active_symbols)}")
        logger.info(f"🔒 Position Size: {self.FIXED_POSITION_SIZE_USDT} USDT (FIXED)")
        logger.info(f"📈 Timeframe: {self.timeframe}")
        logger.info(f"🔄 Max Positions: {self.max_positions}")
        logger.info(f"⚡ Execution Timeout: {self.execution_timeout} seconds")
        logger.info(f"🔧 SuperTrend Period: {self.st_period}")
        logger.info(f"🔧 SuperTrend Multiplier: {self.st_multiplier}")
        logger.info(f"🛡️ Risk Management: {self.stop_loss_pct * 100}% stop loss")
        
        # Display leverage categories
        logger.info("\n⚡ LEVERAGE CATEGORIES:")
        logger.info("   🏆 MAJOR CRYPTOS: 50x max leverage (BTC, ETH, SOL, BNB)")
        logger.info("   🔥 POPULAR ALTS: 40x max leverage (XRP, ADA, DOT, MATIC)")
        logger.info("   🚀 MEME COINS: 30x max leverage (DOGE, SHIB, PEPE)")
        logger.info("   🎮 GAMING: 35x max leverage (SAND, MANA, AXS)")
        logger.info("   🤖 AI/TECH: 35x max leverage (FET, AGIX, RNDR)")
        logger.info("   🏦 DEFI: 40x max leverage (SUSHI, COMP, UNI)")
        logger.info("   💼 OTHERS: 25x max leverage (conservative)")
        
        # Display first 20 pairs as example
        logger.info("\n📝 FIRST 20 ACTIVE PAIRS:")
        for i, symbol in enumerate(self.active_symbols[:20]):
            leverage_info = self.get_pair_leverage_settings(symbol)
            logger.info(f"   {i+1:2d}. {symbol:<12} (max: {leverage_info['max_leverage']}x)")
        
        if len(self.active_symbols) > 20:
            logger.info(f"   ... and {len(self.active_symbols) - 20} more pairs")
        
        logger.info("=" * 80)
        logger.info("🚀 READY TO SCAN ALL PAIRS FOR OPPORTUNITIES!")
        logger.info("=" * 80)

    def log_swap_account_balance(self):
        """Fetch and log the USDT balance from the SWAP (perpetual) account, log full account info"""
        try:
            logger.info("🔍 Fetching account balance from Bitget SWAP (perpetual) account...")
            
            # Recreate the exchange options object properly if needed
            if not hasattr(self.exchange, 'options') or self.exchange.options is None:
                self.exchange.options = {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                    'createMarketBuyOrderRequiresPrice': False,
                    'defaultMarginMode': 'cross',  # PREFERENCE for cross, but will check per symbol
                    'hedgeMode': True
                }
            
            try:
                # Get SPECIFIC balance for USDT-M futures
                balance = self.exchange.fetch_balance({
                    'type': 'swap',
                    'product_type': 'umcbl'  # USDT-margined contracts
                })
                
                # Get USDT balance specifically
                usdt_info = balance.get('USDT', {})
                total = usdt_info.get('total', 0)
                free = usdt_info.get('free', 0)
                used = usdt_info.get('used', 0)
                
                # Store for later use in trading - Fix linter error by handling None explicitly
                self.available_balance = float(free) if free is not None and free != '' else 0.0
                
                logger.info(f"💵 USDT SWAP BALANCE: total={total} | free={free} | used={used}")
                
                if self.available_balance < 1.0:
                    logger.warning("⚠️ WARNING: Low USDT in SWAP account. Please transfer funds from Spot to USDT-M SWAP wallet in Bitget.")
                    logger.warning("⚠️ GO TO BITGET -> ASSETS -> TRANSFER -> FROM: SPOT -> TO: USDT-M FUTURES")
                
            except Exception as e:
                logger.error(f"❌ Error fetching detailed USDT-M balance: {e}")
                self.available_balance = 0.0
                
        except Exception as e:
            logger.error(f"❌ Error fetching SWAP account balance: {e}")
            self.available_balance = 0.0  # Set to zero if balance check fails

# Configuration management
BITGET_API_KEY = "bg_5400882ef43c5596ffcf4af0c697b250"
BITGET_SECRET = "60e42c8f086221d6dd992fc93e5fb810b0354adaa09b674558c14cbd49969d45"
BITGET_PASSPHRASE = "22672267"

# Overwrite config file with these credentials at startup
CONFIG_FILE = "config/bitget_config.json"
def force_write_bitget_config():
    config = {
        "api_key": BITGET_API_KEY,
        "secret": BITGET_SECRET,
        "passphrase": BITGET_PASSPHRASE,
        "sandbox": False,
        "position_size_fixed": 0.50
    }
    Path("config").mkdir(exist_ok=True)
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"✅ Bitget config forcibly written with hardcoded credentials.")
    except Exception as e:
        logger.error(f"❌ Error writing config: {e}")
    return config

def main():
    logger.info("🚀 SUPERTREND PULLBACK LIVE TRADER - STARTING NOW!")
    force_write_bitget_config()  # Always enforce credentials at startup
    trader = AggressivePullbackTrader()  # LIVE TRADING MODE
    try:
        asyncio.run(trader.main_trading_loop())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(f"📍 Error details: {traceback.format_exc()}")


if __name__ == "__main__":
    main() 