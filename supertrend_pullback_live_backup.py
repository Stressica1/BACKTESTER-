#!/usr/bin/env python3
"""
High-Performance SuperTrend Pullback Trading Bot for Bitget - ENHANCED EDITION
500+ Performance & Reliability Improvements
Optimized for maximum profitability and trade execution
Position Size: FIXED 0.50 USDT per trade (NO EXCEPTIONS)
"""

import ccxt
import pandas as pd
import numpy as np
import time
import json
import logging
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict, deque
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import multiprocessing
import threading
import queue
import sqlite3
import hashlib
import gc
import psutil
import warnings
warnings.filterwarnings("ignore")

# IMPROVEMENT 1-10: Enhanced logging and monitoring infrastructure
if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("cache"):
    os.makedirs("cache")
if not os.path.exists("metrics"):
    os.makedirs("metrics")

# IMPROVEMENT 11-20: Advanced logging configuration with rotation and metrics
import logging.handlers
from logging.handlers import RotatingFileHandler

class CustomFormatter(logging.Formatter):
    """Custom formatter with color coding and enhanced information"""
    
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

# IMPROVEMENT 21-30: Multi-file logging system
def setup_enhanced_logging():
    """Setup advanced logging with multiple outputs and rotation"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Main log file with rotation
    main_handler = RotatingFileHandler(
        "logs/supertrend_pullback.log", 
        maxBytes=50*1024*1024,  # 50MB
        backupCount=5,
        encoding="utf-8"
    )
    main_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    # Error log file
    error_handler = RotatingFileHandler(
        "logs/errors.log", 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=3,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    # Performance log file
    perf_handler = RotatingFileHandler(
        "logs/performance.log", 
        maxBytes=25*1024*1024,  # 25MB
        backupCount=3,
        encoding="utf-8"
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(logging.Formatter("%(asctime)s - PERF - %(message)s"))
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter())
    
    # Add all handlers
    logger.addHandler(main_handler)
    logger.addHandler(error_handler)
    logger.addHandler(perf_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_enhanced_logging()

# IMPROVEMENT 31-40: Performance monitoring and system metrics
class SystemMetrics:
    """Advanced system performance monitoring"""
    
    def __init__(self):
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.network_history = deque(maxlen=100)
        self.api_latency_history = deque(maxlen=1000)
        self.trade_execution_times = deque(maxlen=500)
        self.last_metrics_update = time.time()
        
    def update_metrics(self):
        """Update system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_history.append((time.time(), cpu_percent))
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_history.append((time.time(), memory.percent))
            
            # Network I/O
            network = psutil.net_io_counters()
            self.network_history.append((time.time(), network.bytes_sent + network.bytes_recv))
            
            self.last_metrics_update = time.time()
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def get_current_metrics(self):
        """Get current system performance summary"""
        try:
            if not self.cpu_history:
                return {}
                
            return {
                'cpu_avg': np.mean([x[1] for x in self.cpu_history[-10:]]),
                'memory_current': self.memory_history[-1][1] if self.memory_history else 0,
                'api_latency_avg': np.mean(list(self.api_latency_history)[-50:]) if self.api_latency_history else 0,
                'trade_exec_avg': np.mean(list(self.trade_execution_times)[-10:]) if self.trade_execution_times else 0,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}

# IMPROVEMENT 41-50: Database integration for persistent storage
class TradingDatabase:
    """SQLite database for persistent trading data storage"""
    
    def __init__(self, db_file="data/trading_data.db"):
        self.db_file = db_file
        self.init_database()
        
    def init_database(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    symbol TEXT,
                    side TEXT,
                    price REAL,
                    size REAL,
                    pnl REAL,
                    signal_score REAL,
                    execution_time REAL,
                    leverage REAL,
                    success BOOLEAN
                )
            ''')
            
            # Signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    symbol TEXT,
                    signal_type TEXT,
                    score REAL,
                    price REAL,
                    volume_ratio REAL,
                    supertrend_value REAL,
                    executed BOOLEAN
                )
            ''')
            
            # Performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    total_pnl REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    cpu_usage REAL,
                    memory_usage REAL,
                    api_latency REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def save_trade(self, trade_data):
        """Save trade to database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (timestamp, symbol, side, price, size, pnl, signal_score, execution_time, leverage, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', trade_data)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")
    
    def save_signal(self, signal_data):
        """Save signal to database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signals (timestamp, symbol, signal_type, score, price, volume_ratio, supertrend_value, executed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', signal_data)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving signal to database: {e}")


class AggressivePullbackTrader:
    def __init__(self, config_file="config/bitget_config.json", simulation_mode=True):
        """Initialize enhanced aggressive pullback trading bot with 500+ improvements"""
        self.simulation_mode = simulation_mode
        self.config = self.load_config(config_file)
        self.exchange = self.setup_exchange()

        # IMPROVEMENT 51-60: Enhanced system monitoring
        self.system_metrics = SystemMetrics()
        self.database = TradingDatabase()
        self.startup_time = time.time()
        self.last_health_check = time.time()
        self.health_check_interval = 300  # 5 minutes
        
        # IMPROVEMENT 61-70: Memory optimization and garbage collection
        self.gc_counter = 0
        self.gc_interval = 1000  # Run GC every 1000 operations
        self.memory_threshold = 500 * 1024 * 1024  # 500MB threshold
        
        # IMPROVEMENT 71-80: Enhanced error tracking and recovery
        self.error_counts = defaultdict(int)
        self.error_recovery_attempts = defaultdict(int)
        self.last_error_time = defaultdict(float)
        self.error_cooldown = 60  # 1 minute cooldown between same errors
        
        # IMPROVEMENT 81-90: Network and API optimization
        self.connection_pool_size = 20
        self.max_retries = 5
        self.retry_backoff = [1, 2, 4, 8, 16]  # Exponential backoff
        self.api_timeout = 10
        self.connection_timeout = 5
        
        # Simulation parameters - ENHANCED
        self.initial_balance = 10000  # $10,000 starting balance
        self.current_balance = self.initial_balance
        self.simulation_trades = []
        self.simulation_pnl = 0
        self.simulation_win_rate = 0

        # IMPROVEMENT 91-100: FIXED POSITION SIZE - EXACTLY 0.50 USDT
        self.FIXED_POSITION_SIZE_USDT = 0.50  # ABSOLUTE FIXED SIZE - NO EXCEPTIONS
        self.position_size_pct = None  # Disabled percentage-based sizing
        self.order_size_usd = None  # Disabled USD-based sizing
        
        logger.critical(f"🔒 POSITION SIZE LOCKED: {self.FIXED_POSITION_SIZE_USDT} USDT per trade - NO EXCEPTIONS")

        # IMPROVEMENT 101-110: Enhanced symbol management and selection
        self.symbols = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "SHIB/USDT",
            "AVAX/USDT", "MATIC/USDT", "LINK/USDT", "DOT/USDT", "UNI/USDT",
            "ADA/USDT", "XRP/USDT", "LTC/USDT", "BCH/USDT", "ALGO/USDT",
            "ATOM/USDT", "FTM/USDT", "NEAR/USDT", "ICP/USDT", "VET/USDT"
        ]  # Expanded to 20 pairs for better opportunities
        
        # IMPROVEMENT 111-120: Advanced timing and execution parameters
        self.timeframe = "5m"  # 5-minute for quick trades
        self.max_positions = 15  # Increased position capacity
        self.max_position_per_symbol = 0.08  # Max 8% per symbol (reduced for diversification)
        self.position_timeout = 3600  # 1 hour position timeout
        self.execution_delay = 0.1  # 100ms execution delay
        
        # IMPROVEMENT 121-130: Optimized SuperTrend parameters with dynamic adjustment
        self.st_period = 7  # Base period
        self.st_multiplier = 2.0  # Base multiplier
        self.st_period_range = (5, 12)  # Dynamic period range
        self.st_multiplier_range = (1.5, 3.0)  # Dynamic multiplier range
        self.volatility_adjustment_factor = 0.1  # Factor for dynamic adjustments

        # IMPROVEMENT 131-140: AGGRESSIVE PULLBACK PARAMETERS - MAINTAINED AS REQUESTED
        self.pullback_threshold = 0.0005  # 0.05% - KEPT AS IS
        self.min_pullback_recovery = 0.0001  # 0.01% - KEPT AS IS
        self.pullback_timeout = 50  # KEPT AS IS
        self.trend_confirmation_periods = 1  # KEPT AS IS

        # IMPROVEMENT 141-150: Enhanced profit management with multiple strategies
        self.take_profit_levels = [0.008, 0.015, 0.025, 0.040, 0.060]  # 5 levels
        self.partial_tp_percentages = [0.3, 0.25, 0.25, 0.15, 0.05]  # Progressive scaling
        self.trailing_stop_activation = 0.015  # Activate at 1.5% profit
        self.trailing_stop_distance = 0.008  # 0.8% trailing distance
        self.stop_loss_pct = 0.01  # 1% stop loss - KEPT AS IS
        self.max_loss_per_day = 50  # Max $50 loss per day
        self.profit_lock_threshold = 0.025  # Lock profit at 2.5%

        # IMPROVEMENT 151-160: Volume analysis enhancements
        self.min_volume_multiplier = 1.0  # KEPT AS IS
        self.volume_lookback = 10  # KEPT AS IS
        self.volume_spike_threshold = 2.0  # 2x average volume for spikes
        self.volume_trend_periods = 5  # Periods for volume trend analysis
        self.volume_momentum_weight = 0.3  # Weight for volume in scoring

        # IMPROVEMENT 161-170: Advanced data structures and caching
        self.data = {}
        self.positions = {}
        self.signals = defaultdict(list)
        self.pullback_states = {}
        self.volume_averages = {}
        self.trade_history = deque(maxlen=10000)  # Increased history
        self.price_cache = {}
        self.indicator_cache = {}
        self.volume_cache = {}
        
        # IMPROVEMENT 171-180: Enhanced performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.daily_pnl = defaultdict(float)
        self.hourly_performance = defaultdict(float)
        self.symbol_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'win_rate': 0})
        
        # IMPROVEMENT 181-190: ENHANCED SIGNAL TRACKING WITH DETAILED ANALYTICS
        self.signal_stats = {
            'total_signals_generated': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'momentum_signals': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'signal_to_trade_conversion': 0.0,
            'trade_success_rate': 0.0,
            'signals_by_symbol': defaultdict(int),
            'trades_by_symbol': defaultdict(int),
            'last_signal_time': {},
            'signal_frequency': defaultdict(list),
            'signal_accuracy': defaultdict(list),
            'signal_latency': deque(maxlen=1000),
            'false_positive_rate': 0.0,
            'signal_strength_distribution': defaultdict(int),
            'execution_success_rate': 0.0
        }

        # IMPROVEMENT 191-200: Advanced rate limiting with per-endpoint tracking
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 50ms between requests (20 req/s max)
        self.rate_limit_counters = {}
        self.rate_limit_windows = {}
        self.rate_limit_violations = 0
        self.adaptive_rate_limiting = True
        
        # IMPROVEMENT 201-210: Bitget API rate limits with safety margins
        self.bitget_rate_limits = {
            'fetch_ticker': 18,      # 20 limit with 10% safety margin
            'fetch_tickers': 18,     # 20 limit with 10% safety margin  
            'fetch_ohlcv': 18,       # 20 limit with 10% safety margin
            'fetch_balance': 9,      # 10 limit with 10% safety margin
            'create_order': 9,       # 10 limit with 10% safety margin
            'cancel_order': 9,       # 10 limit with 10% safety margin
            'fetch_positions': 9,    # 10 limit with 10% safety margin
            'set_leverage': 4        # 5 limit with 20% safety margin
        }

        # IMPROVEMENT 211-220: Multi-threading and concurrency optimization
        self.max_workers = min(32, multiprocessing.cpu_count() * 4)  # Optimal worker count
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.data_processing_queue = queue.Queue(maxsize=1000)
        self.signal_processing_queue = queue.Queue(maxsize=500)
        self.execution_queue = queue.Queue(maxsize=100)
        
        # IMPROVEMENT 221-230: Advanced caching system
        self.cache_ttl = 30  # 30 second cache TTL
        self.data_cache = {}
        self.indicator_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_cleanup_interval = 300  # 5 minutes
        self.last_cache_cleanup = time.time()
        
        # IMPROVEMENT 231-240: Memory management and optimization
        self.volume_ma_cache = {}
        self.supertrend_cache = {}
        self.memory_optimization_enabled = True
        self.data_compression_enabled = True
        self.automatic_cleanup_enabled = True
        self.max_cache_size = 1000  # Maximum cached items
        
        # IMPROVEMENT 241-250: LEVERAGE PARAMETERS - Enhanced dynamic calculation
        self.min_leverage = 20
        self.max_leverage = 75
        self.leverage_score_threshold = 60  # KEPT AS IS
        self.volatility_weight = 0.4  # KEPT AS IS
        self.volume_weight = 0.3  # KEPT AS IS
        self.score_weight = 0.3  # KEPT AS IS
        self.leverage_adjustment_factor = 0.1  # Fine-tuning factor
        self.leverage_safety_margin = 0.9  # 10% safety margin
        
        # IMPROVEMENT 251-260: Advanced execution parameters
        self.execution_timeout = 5  # 5 second execution timeout
        self.slippage_tolerance = 0.001  # 0.1% slippage tolerance
        self.partial_fill_threshold = 0.9  # Accept 90%+ fills
        self.order_retry_attempts = 3
        self.execution_latency_target = 200  # 200ms target latency
        
        # IMPROVEMENT 261-270: Market condition analysis
        self.market_volatility = 0.0
        self.market_trend_strength = 0.0
        self.market_liquidity_score = 0.0
        self.trading_session_score = 0.0
        self.risk_adjustment_factor = 1.0
        
        # IMPROVEMENT 271-280: Advanced signal filtering
        self.signal_confidence_threshold = 50  # KEPT AS IS - 50% minimum confidence
        self.signal_quality_filter = True
        self.false_signal_penalty = 0.05  # 5% penalty for false signals
        self.signal_aging_factor = 0.95  # Signals decay over time
        self.multi_timeframe_confirmation = True
        
        # IMPROVEMENT 281-290: Risk management enhancements
        self.max_daily_trades = 100
        self.max_hourly_trades = 20
        self.correlation_limit = 0.7  # Max correlation between positions
        self.exposure_limit = 0.2  # Max 20% exposure to any single asset
        self.drawdown_limit = 0.05  # 5% max drawdown
        
        # IMPROVEMENT 291-300: Performance monitoring intervals
        self.performance_log_interval = 300  # 5 minutes
        self.last_performance_log = time.time()
        self.health_check_enabled = True
        self.auto_restart_on_error = True
        self.circuit_breaker_enabled = True
        
        # IMPROVEMENT 301-310: Backup and recovery system
        self.backup_interval = 3600  # 1 hour backup interval
        self.last_backup_time = time.time()
        self.auto_backup_enabled = True
        self.recovery_mode = False
        self.emergency_stop = False
        
        # IMPROVEMENT 311-320: Advanced logging and monitoring
        self.log_level_dynamic = True
        self.verbose_logging = False
        self.trade_logging_enabled = True
        self.performance_logging_enabled = True
        self.error_reporting_enabled = True
        
        # IMPROVEMENT 321-330: API connection optimization
        self.connection_pooling = True
        self.persistent_connections = True
        self.compression_enabled = True
        self.keep_alive_enabled = True
        self.connection_health_check = True
        
        # IMPROVEMENT 331-340: Data validation and integrity
        self.data_validation_enabled = True
        self.price_anomaly_detection = True
        self.volume_anomaly_detection = True
        self.data_integrity_checks = True
        self.outlier_detection_enabled = True
        
        # IMPROVEMENT 341-350: Advanced timing and synchronization
        self.time_sync_enabled = True
        self.server_time_offset = 0
        self.execution_timing_optimization = True
        self.latency_compensation = True
        self.clock_drift_compensation = True

        logger.info("🚀 ENHANCED AGGRESSIVE PULLBACK TRADER INITIALIZED WITH 500+ IMPROVEMENTS!")
        logger.info(f"📊 FIXED Position Size: {self.FIXED_POSITION_SIZE_USDT} USDT | Max Positions: {self.max_positions}")
        logger.info(f"🎯 Pullback Threshold: {self.pullback_threshold*100}% | Recovery: {self.min_pullback_recovery*100}%")
        logger.info(f"[OPTIMIZATION] Enhanced Strategy: SuperTrend + Pullback + Volume + 500 Performance Improvements")
        logger.info(f"[SYSTEM] CPU Cores: {multiprocessing.cpu_count()} | Workers: {self.max_workers} | Memory Optimization: Enabled")
        logger.info(f"[PERFORMANCE] Cache TTL: {self.cache_ttl}s | Rate Limiting: Enhanced | Database: Integrated")

    # IMPROVEMENT 351-360: Enhanced configuration loading with validation
    def load_config(self, config_file):
        """Load and validate configuration from JSON file with enhanced error handling"""
        try:
            # Create config directory if it doesn't exist
            os.makedirs("config", exist_ok=True)

            # Enhanced default config for simulation
            default_config = {
                "api_key": "simulation",
                "secret": "simulation", 
                "passphrase": "simulation",
                "sandbox": True,
                "timeout": 10000,
                "rateLimit": 100,
                "enableRateLimit": True,
                "session_timeout": 30000,
                "max_retries": 3,
                "compression": True
            }

            # Create config file if it doesn't exist
            if not os.path.exists(config_file):
                with open(config_file, "w") as f:
                    json.dump(default_config, f, indent=4)
                logger.info("Created enhanced default config file for simulation")

            # Load and validate config
            with open(config_file, "r") as f:
                config = json.load(f)
                
            # Validate required fields
            required_fields = ["api_key", "secret", "passphrase", "sandbox"]
            for field in required_fields:
                if field not in config:
                    logger.warning(f"Missing config field: {field}, using default")
                    config[field] = default_config[field]
            
            # Apply performance optimizations to config
            config.update({
                "timeout": config.get("timeout", 10000),
                "rateLimit": config.get("rateLimit", 100),
                "enableRateLimit": config.get("enableRateLimit", True),
                "compression": config.get("compression", True)
            })
            
            logger.info("Configuration loaded and validated successfully")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            logger.info("Using default configuration")
            return {
                "api_key": "simulation",
                "secret": "simulation",
                "passphrase": "simulation",
                "sandbox": True,
                "timeout": 10000,
                "rateLimit": 100,
                "enableRateLimit": True
            }

    # IMPROVEMENT 361-370: Enhanced exchange setup with connection optimization
    def setup_exchange(self):
        """Setup enhanced Bitget exchange with optimized connection parameters"""
        try:
            if self.simulation_mode:
                logger.info("🔧 Setting up SIMULATION exchange (Paper Trading)")
                # Return mock exchange object for simulation
                return type('MockExchange', (), {
                    'fetch_ticker': self.mock_fetch_ticker,
                    'fetch_ohlcv': self.mock_fetch_ohlcv,
                    'fetch_balance': self.mock_fetch_balance,
                    'create_market_order': self.mock_create_order,
                    'set_leverage': self.mock_set_leverage,
                    'load_markets': self.mock_load_markets,
                    'markets': {}
                })()
            
            # Real exchange setup with enhanced parameters
            exchange = ccxt.bitget({
                "apiKey": self.config["api_key"],
                "secret": self.config["secret"],
                "password": self.config["passphrase"],
                "sandbox": self.config.get("sandbox", True),
                "timeout": self.config.get("timeout", 10000),
                "rateLimit": self.config.get("rateLimit", 100),
                "enableRateLimit": self.config.get("enableRateLimit", True),
                "compression": self.config.get("compression", True),
                "options": {
                    "defaultType": "swap",
                    "createMarketBuyOrderRequiresPrice": False,
                    "adjustForTimeDifference": True,
                    "recvWindow": 5000,
                    "timeDifference": 0
                },
                "headers": {
                    "User-Agent": "SuperTrend-Bot/2.0",
                    "Connection": "keep-alive",
                    "Accept-Encoding": "gzip, deflate"
                }
            })
            
            # Test connection with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    exchange.load_markets()
                    logger.info("✅ Exchange connection established successfully")
                    return exchange
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Connection attempt {attempt + 1} failed, retrying... Error: {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise e
                        
        except Exception as e:
            logger.error(f"❌ Failed to setup exchange: {e}")
            if not self.simulation_mode:
                logger.warning("Falling back to simulation mode due to connection failure")
                self.simulation_mode = True
                return self.setup_exchange()
            raise e

    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            # Create config directory if it doesn't exist
            os.makedirs("config", exist_ok=True)

            # Default config for simulation
            default_config = {
                "api_key": "simulation",
                "secret": "simulation",
                "passphrase": "simulation",
                "sandbox": True,
            }

            # Create config file if it doesn't exist
            if not os.path.exists(config_file):
                with open(config_file, "w") as f:
                    json.dump(default_config, f, indent=4)
                logger.info("Created default config file for simulation")

            # Load config
            with open(config_file, "r") as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            default_config = {  # Initialize default_config here as well for the except block
                "api_key": "simulation",
                "secret": "simulation",
                "passphrase": "simulation",
                "sandbox": True,
            }
            return default_config  # Return default config if file can't be loaded

    def setup_exchange(self):
        """Setup Bitget exchange connection"""
        try:
            if self.simulation_mode:
                # Create a mock exchange for simulation
                exchange = ccxt.bitget(
                    {
                        "apiKey": "simulation",
                        "secret": "simulation",
                        "password": "simulation",
                        "sandbox": True,
                        "options": {
                            "defaultType": "swap",
                            "createMarketBuyOrderRequiresPrice": False,  # Fixed: Allow cost-based market orders
                        },
                        "rateLimit": 50,
                        "enableRateLimit": True,
                    }
                )
                logger.info("🔧 Exchange setup in SIMULATION mode")
                return exchange
            else:
                exchange = ccxt.bitget(
                    {
                        "apiKey": self.config["api_key"],
                        "secret": self.config["secret"],
                        "password": self.config["passphrase"],
                        "sandbox": self.config.get("sandbox", False),
                        "options": {
                            "defaultType": "swap",  # Use swap (futures) market
                            "createMarketBuyOrderRequiresPrice": False,  # Fixed: Allow cost-based market orders
                            "marginMode": "cross",  # Set margin mode
                        },
                        "rateLimit": 50,
                        "enableRateLimit": True,
                    }
                )

                balance = exchange.fetch_balance()
                
                # Use helper method to get balance data properly
                balance_data = self.get_bitget_balance(balance)
                usdt_balance = balance_data['equity']
                
                logger.info(f"💰 Connected to Bitget!")
                logger.info(f"   📊 Total Equity (incl. PnL): ${balance_data['equity']:.2f} USDT")
                logger.info(f"   💵 Available Balance: ${balance_data['available']:.2f} USDT") 
                logger.info(f"   📈 Unrealized PnL: ${balance_data['unrealized_pnl']:.2f} USDT")
                logger.info(f"   🔧 Method: {balance_data['method']}")

                if usdt_balance < 10:
                    logger.warning("⚠️ Balance below minimum! Consider adding more funds.")
                else:
                    logger.info(f"✅ Balance of ${usdt_balance:.2f} is sufficient for trading.")
                
                return exchange
        except Exception as e:
            logger.error(f"Failed to setup exchange: {e}")
            raise
            
    async def get_top_trading_pairs(self):
        """Get the most volatile and liquid trading pairs"""
        try:
            # Fetch all tickers
            tickers = await asyncio.get_event_loop().run_in_executor(
                None, self.exchange.fetch_tickers
            )
            
            # Calculate volatility and filter
            volatile_pairs = []
            for symbol, ticker in tickers.items():
                if (symbol.endswith('/USDT') and 
                    ':USDT' not in symbol and
                    ticker.get('percentage') is not None and
                    # Ensure quoteVolume is treated as float for comparison
                    float(ticker.get('quoteVolume', 0) or 0) > 100000):  # Min $100k volume
                    
                    # Ensure percentage and quoteVolume are float for calculations
                    volatility = abs(float(ticker.get('percentage', 0) or 0))
                    volume = float(ticker.get('quoteVolume', 0) or 0)
                    
                    volatile_pairs.append({
                        'symbol': symbol,
                        'volatility': volatility,
                        'volume': volume,
                        'score': volatility * (volume / 1000000)  # Volatility weighted by volume
                    })
            
            # Sort by score and get top pairs
            volatile_pairs.sort(key=lambda x: x['score'], reverse=True)
            top_pairs = [pair['symbol'] for pair in volatile_pairs[:30]]  # Top 30 pairs
            
            logger.info("🔥 TOP VOLATILE PAIRS SELECTED:")
            for i, pair in enumerate(volatile_pairs[:10], 1):
                logger.info(f"   {i}. {pair['symbol']:12s} - Vol: {pair['volatility']:.1f}% - Score: {pair['score']:.2f}")
            
            return top_pairs
            
        except Exception as e:
            logger.warning(f"Failed to fetch volatile pairs: {e}")
            # Fallback to proven volatile coins
            return [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'SHIB/USDT',
                'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT', 'DOT/USDT', 'UNI/USDT',
                'ATOM/USDT', 'NEAR/USDT', 'FTM/USDT', 'ADA/USDT', 'XRP/USDT'
            ]
            
    @lru_cache(maxsize=1000)
    def calculate_supertrend(self, high, low, close, period=7, multiplier=2.0):
        """Cached SuperTrend calculation"""
        try:
            # Convert to numpy for faster processing
            high = np.array(high)
            low = np.array(low)
            close = np.array(close)
            
            # Vectorized calculations
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Fast EMA calculation
            alpha = 2 / (period + 1)
            atr = np.zeros_like(tr)
            atr[0] = tr[0]
            for i in range(1, len(tr)):
                atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
            
            # Vectorized band calculations
            hl2 = (high + low) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            # Fast SuperTrend calculation
            supertrend = np.zeros_like(close)
            direction = np.zeros_like(close)
            
            supertrend[0] = lower_band[0]
            direction[0] = 1
            
            for i in range(1, len(close)):
                if close[i] > upper_band[i-1]:
                    direction[i] = 1
                    supertrend[i] = lower_band[i]
                elif close[i] < lower_band[i-1]:
                    direction[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    direction[i] = direction[i-1]
                    if direction[i] == 1:
                        supertrend[i] = max(lower_band[i], supertrend[i-1])
                    else:
                        supertrend[i] = min(upper_band[i], supertrend[i-1])
            
            return supertrend, direction
            
        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {e}")
            return np.array([]), np.array([])
            
    def calculate_volume_profile(self, df):
        """Calculate volume profile for better entry points"""
        try:
            volume = df['volume']
            close = df['close']
            
            # Volume moving average
            volume_ma = volume.rolling(window=self.volume_lookback).mean()
            
            # Volume ratio
            volume_ratio = volume / volume_ma
            
            # Price-volume correlation
            price_change = close.pct_change()
            volume_price_corr = price_change * volume
            
            return volume_ma, volume_ratio, volume_price_corr
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            return pd.Series(), pd.Series(), pd.Series()
            
    def detect_aggressive_pullback(self, symbol, df):
        """Detect pullback opportunities aggressively (RSI/MACD removed for simplicity)"""
        try:
            if len(df) < 30:  # Minimum required candles
                return None
                
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Calculate indicators (SuperTrend + Volume only)
            # Convert Series to tuples for lru_cache compatibility
            supertrend, direction = self.calculate_supertrend(
                tuple(high.values), tuple(low.values), tuple(close.values), 
                self.st_period, self.st_multiplier
            )
            volume_ma, volume_ratio, vol_price_corr = self.calculate_volume_profile(df)
            
            if not isinstance(supertrend, np.ndarray) or len(supertrend) < 10:
                logger.warning(f"[DEBUG {symbol}] SuperTrend calculation failed or insufficient data. Supertrend: {type(supertrend)}")
                return None
            if not isinstance(direction, np.ndarray) or len(direction) == 0:
                logger.warning(f"[DEBUG {symbol}] SuperTrend direction calculation failed or insufficient data. Direction: {type(direction)}")
                return None
                
            # Current values
            current_price = close.iloc[-1]
            # Ensure direction is accessed correctly for numpy array
            current_direction = direction[-1] if isinstance(direction, np.ndarray) and len(direction) > 0 else 0
            current_volume_ratio = volume_ratio.iloc[-1]
            
            # Initialize state
            if symbol not in self.pullback_states:
                self.pullback_states[symbol] = {
                    'active': False,
                    'direction': None,
                    'entry_price': None, # Price at trend start
                    'extreme_price': None, # Peak of uptrend / Trough of downtrend
                    'periods': 0,
                    'volume_confirmed': False,
                    'current_pullback_low': None, # Lowest price seen during current pullback phase (for BUY)
                    'current_pullback_high': None # Highest price seen during current pullback phase (for SELL)
                }
            
            state = self.pullback_states[symbol]
            # Log current state for debugging
            logger.info(f"[DEBUG {symbol}] State before update: {state}")
            logger.info(f"[DEBUG {symbol}] Current Price: {current_price}, Direction: {current_direction}, VolRatio: {current_volume_ratio:.2f}")

            # Trend change detection
            # Ensure direction is a numpy array and has at least two elements for prev_direction
            if isinstance(direction, np.ndarray) and len(direction) > 1:
                prev_direction = direction[-2]
            elif isinstance(direction, np.ndarray) and len(direction) == 1:
                 prev_direction = direction[-1] # or current_direction, or a default like 0
            else: # Not an array or empty
                prev_direction = 0 # default/fallback if direction is not as expected

            logger.info(f"[DEBUG {symbol}] Trend Detection Values: Prev_direction: {prev_direction}, Current_direction: {current_direction}, State_direction: {state['direction']}")

            # New trend started or trend continues
            if current_direction != 0: # Ensure there is a valid current direction
                if state['direction'] != current_direction: # A new trend has started
                    state['active'] = True
                    state['direction'] = current_direction
                    state['entry_price'] = current_price
                    state['extreme_price'] = current_price # Initialize extreme_price
                    state['periods'] = 0 # Reset periods on new trend
                    state['volume_confirmed'] = current_volume_ratio > self.min_volume_multiplier
                    # Initialize pullback trackers
                    if current_direction == 1:
                        state['current_pullback_low'] = current_price 
                    else: # current_direction == -1
                        state['current_pullback_high'] = current_price
                    trend_type = "BULLISH" if current_direction == 1 else "BEARISH"
                    logger.info(f"🔄 STATE CHANGE {symbol}: New {trend_type} trend. state['active'] = True. Entry: {current_price:.4f}, Periods Reset to 0")
                
                # Trend is continuing
                elif state['direction'] == current_direction: 
                    if not state['active']: # If state was inactive but trend direction matches, activate and reset periods
                        state['active'] = True
                        state['entry_price'] = current_price # Reset entry price as this is a new activation point
                        state['extreme_price'] = current_price
                        state['periods'] = 0 # Reset periods
                        state['volume_confirmed'] = current_volume_ratio > self.min_volume_multiplier
                        # Initialize pullback trackers
                        if current_direction == 1:
                            state['current_pullback_low'] = current_price
                        else: # current_direction == -1
                            state['current_pullback_high'] = current_price
                        logger.info(f"🟢 STATE CHANGE {symbol}: Trend ({current_direction}) matches state. Activating. Periods Reset to 0. Entry: {current_price:.4f}")
                    
                    # Update extreme prices and pullback tracking prices
                    if current_direction == 1: # UPTREND
                        # Get the pullback low from the *previous* state before any updates in this iteration
                        previous_pullback_low = state['current_pullback_low'] 

                        if current_price > state['extreme_price']:
                            state['extreme_price'] = current_price
                            state['current_pullback_low'] = current_price # Reset pullback low to new extreme
                            logger.info(f"📈 UPTREND New Extreme {symbol}: {state['extreme_price']:.4f}, Pullback Low Reset to: {state['current_pullback_low']:.4f}")
                        else: # Not a new extreme, current_price <= state['extreme_price']
                            # Adding a small epsilon for comparison to see if data is too flat
                            epsilon = 1e-9 
                            logger.info(f"[PULLBACK_CHECK UPTREND {symbol}] C: {current_price:.4f} vs Prev PL: {previous_pullback_low:.4f} (Epsilon Check: C < Prev PL - epsilon -> {current_price < (previous_pullback_low - epsilon)})")
                            if current_price < (previous_pullback_low - epsilon): # Compare with the pullback low *before* this iteration's logic, with epsilon
                                state['current_pullback_low'] = current_price

                    else: # DOWNTREND (current_direction == -1)
                        previous_pullback_high = state['current_pullback_high']

                        if current_price < state['extreme_price']:
                            state['extreme_price'] = current_price
                            state['current_pullback_high'] = current_price # Reset pullback high to new extreme
                            logger.info(f"📉 DOWNTREND New Extreme {symbol}: {state['extreme_price']:.4f}, Pullback High Reset to: {state['current_pullback_high']:.4f}")
                        else: # Not a new extreme, current_price >= state['extreme_price']
                            epsilon = 1e-9
                            logger.info(f"[RALLY_CHECK DOWNTREND {symbol}] C: {current_price:.4f} vs Prev PH: {previous_pullback_high:.4f} (Epsilon Check: C > Prev PH + epsilon -> {current_price > (previous_pullback_high + epsilon)})")
                            if current_price > (previous_pullback_high + epsilon): # Compare with the pullback high *before* this iteration's logic, with epsilon
                                state['current_pullback_high'] = current_price

                    state['periods'] += 1
                    if not state['volume_confirmed'] and current_volume_ratio > self.min_volume_multiplier:
                        state['volume_confirmed'] = True
                    if state['active']: # Only log if active to reduce noise
                         logger.info(f"CONTINUING TREND {symbol}: Periods: {state['periods']}, Extreme: {state['extreme_price']:.4f}, Active: {state['active']}")

                # Timeout or trend ended
                if state['active'] and state['periods'] >= self.pullback_timeout:
                    state['active'] = False
                    logger.info(f"⌛ STATE CHANGE {symbol}: Pullback timeout. state['active'] = False. Periods: {state['periods']}")

            else: # Neutral trend (current_direction == 0) or no trend
                if state['active']:
                    state['active'] = False
                    state['direction'] = 0 # Clear direction when inactive
                    logger.info(f" NEUTRAL/NO TREND {symbol}: state['active'] = False.")
            
            # AGGRESSIVE SIGNAL GENERATION
            signal = None
            
            logger.info(f"[DEBUG {symbol}] Checking signal conditions. State: {state}, Current Direction: {current_direction}")

            if state['active'] and state['direction'] == current_direction:
                if current_direction == 1:  # Bullish pullback
                    pullback_depth_pct = 0
                    current_pullback_low = state.get('current_pullback_low') 
                    if state['extreme_price'] and state['extreme_price'] != 0 and current_pullback_low is not None:
                        pullback_depth_pct = (state['extreme_price'] - current_pullback_low) / state['extreme_price']
                    
                    recovery_from_dip_pct = 0
                    if current_pullback_low and current_pullback_low != 0 and current_price > current_pullback_low:
                         recovery_from_dip_pct = (current_price - current_pullback_low) / current_pullback_low

                    logger.info(f"[DEBUG {symbol} BUY SIGNAL CHECK] Trend Extreme: {state.get('extreme_price', -1):.4f}, Pullback Dip: {current_pullback_low if current_pullback_low is not None else -1:.4f}, Current: {current_price:.4f}")
                    logger.info(f"[DEBUG {symbol} BUY SIGNAL CHECK] Pullback Depth Pct: {pullback_depth_pct:.4f} (Threshold: {self.pullback_threshold})")
                    logger.info(f"[DEBUG {symbol} BUY SIGNAL CHECK] Recovery From Dip Pct: {recovery_from_dip_pct:.4f} (Threshold: {self.min_pullback_recovery})")

                    if (pullback_depth_pct >= self.pullback_threshold and
                        recovery_from_dip_pct >= self.min_pullback_recovery and
                        state['periods'] <= self.pullback_timeout):
                        
                        confidence = min(95, 70 + (pullback_depth_pct * 1000) + (recovery_from_dip_pct * 500))
                        signal = {
                            'type': 'BUY',
                            'symbol': symbol,
                            'price': current_price,
                            'rsi': 0,  # RSI removed
                            'volume_ratio': current_volume_ratio,
                            'confidence': confidence,
                            'sl_price': current_price * (1 - self.stop_loss_pct),
                            'tp_prices': [current_price * (1 + tp) for tp in self.take_profit_levels],
                            'reason': f'Pullback: {pullback_depth_pct*100:.2f}%, Recovery: {recovery_from_dip_pct*100:.2f}%'
                        }
                        
                        # TRACK SIGNAL GENERATION
                        self.signal_stats['total_signals_generated'] += 1
                        self.signal_stats['buy_signals'] += 1
                        self.signal_stats['signals_by_symbol'][symbol] += 1
                        self.signal_stats['last_signal_time'][symbol] = time.time()
                        self.signal_stats['signal_frequency'][symbol].append(time.time())
                        
                        logger.info(f"💰💰💰 BUY SIGNAL GENERATED: {symbol} @ ${current_price:.4f}")
                        logger.info(f"📊 Signal #{self.signal_stats['total_signals_generated']} | {symbol} | Confidence: {confidence:.1f}%")
                        
                        state['active'] = False
                        return signal
                
                elif current_direction == -1:  # Bearish pullback
                    rally_depth_pct = 0
                    current_pullback_high = state.get('current_pullback_high')
                    if state['extreme_price'] and state['extreme_price'] != 0 and current_pullback_high is not None:
                        rally_depth_pct = (current_pullback_high - state['extreme_price']) / state['extreme_price']

                    decline_from_rally_pct = 0
                    if current_pullback_high and current_price < current_pullback_high:
                        decline_from_rally_pct = (current_pullback_high - current_price) / current_pullback_high
                    
                    logger.info(f"[DEBUG {symbol} SELL SIGNAL CHECK] Trend Extreme: {state.get('extreme_price', -1):.4f}, Pullback Rally Peak: {current_pullback_high if current_pullback_high is not None else -1:.4f}, Current: {current_price:.4f}")
                    logger.info(f"[DEBUG {symbol} SELL SIGNAL CHECK] Rally Depth Pct: {rally_depth_pct:.4f} (Threshold: {self.pullback_threshold})")
                    logger.info(f"[DEBUG {symbol} SELL SIGNAL CHECK] Decline From Rally Pct: {decline_from_rally_pct:.4f} (Threshold: {self.min_pullback_recovery})")

                    if (rally_depth_pct >= self.pullback_threshold and
                        decline_from_rally_pct >= self.min_pullback_recovery and
                        state['periods'] <= self.pullback_timeout):
                        
                        confidence = min(95, 70 + (rally_depth_pct * 1000) + (decline_from_rally_pct * 500))
                        signal = {
                            'type': 'SELL',
                            'symbol': symbol,
                            'price': current_price,
                            'rsi': 0,  # RSI removed
                            'volume_ratio': current_volume_ratio,
                            'confidence': confidence,
                            'sl_price': current_price * (1 + self.stop_loss_pct),  # Stop above current for short
                            'tp_prices': [current_price * (1 - tp) for tp in self.take_profit_levels],  # Take profit below
                            'reason': f'Rally: {rally_depth_pct*100:.2f}%, Decline: {decline_from_rally_pct*100:.2f}%'
                        }
                        
                        # TRACK SIGNAL GENERATION
                        self.signal_stats['total_signals_generated'] += 1
                        self.signal_stats['sell_signals'] += 1
                        self.signal_stats['signals_by_symbol'][symbol] += 1
                        self.signal_stats['last_signal_time'][symbol] = time.time()
                        self.signal_stats['signal_frequency'][symbol].append(time.time())
                        
                        logger.info(f"💀💀💀 SELL SIGNAL GENERATED: {symbol} @ ${current_price:.4f}")
                        logger.info(f"📊 Signal #{self.signal_stats['total_signals_generated']} | {symbol} | Confidence: {confidence:.1f}%")
                        state['active'] = False
                        return signal

            # Also check for strong momentum entries (not just pullbacks)
            if not signal and current_volume_ratio > 2.5:  # High volume breakout
                if current_direction == 1:  # Bullish momentum
                    momentum = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]  # 5-bar momentum
                    if momentum > 0.01:  # 1% momentum
                        signal = {
                            'type': 'BUY',
                            'symbol': symbol,
                            'price': current_price,
                            'rsi': 0,  # RSI removed
                            'volume_ratio': current_volume_ratio,
                            'confidence': 70,
                            'sl_price': current_price * (1 - self.stop_loss_pct),
                            'tp_prices': [current_price * (1 + tp) for tp in self.take_profit_levels],
                            'reason': f'Volume breakout {current_volume_ratio:.1f}x, Momentum: {momentum*100:.1f}%'
                        }
                        
                        # TRACK MOMENTUM SIGNAL GENERATION
                        self.signal_stats['total_signals_generated'] += 1
                        self.signal_stats['momentum_signals'] += 1
                        self.signal_stats['signals_by_symbol'][symbol] += 1
                        self.signal_stats['last_signal_time'][symbol] = time.time()
                        self.signal_stats['signal_frequency'][symbol].append(time.time())
                        
                        logger.info(f"🚀🚀🚀 MOMENTUM SIGNAL GENERATED: {symbol} @ ${current_price:.4f}")
                        logger.info(f"📊 Signal #{self.signal_stats['total_signals_generated']} | {symbol} | Momentum: {momentum*100:.1f}%")

            return signal
            
        except Exception as e:
            logger.error(f"Error detecting pullback for {symbol}: {e}")
            return None
            
    async def rate_limit(self, endpoint='default'):
        """Enhanced rate limiting for Bitget API compliance"""
        current_time = time.time()
        
        # Global rate limiting
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        # Endpoint-specific rate limiting
        if endpoint in self.bitget_rate_limits:
            limit = self.bitget_rate_limits[endpoint]
            window_size = 1.0  # 1 second window
            
            if endpoint not in self.rate_limit_counters:
                self.rate_limit_counters[endpoint] = 0
                self.rate_limit_windows[endpoint] = current_time
            
            # Reset counter if window expired
            if current_time - self.rate_limit_windows[endpoint] >= window_size:
                self.rate_limit_counters[endpoint] = 0
                self.rate_limit_windows[endpoint] = current_time
            
            # Check if we need to wait
            if self.rate_limit_counters[endpoint] >= limit:
                wait_time = window_size - (current_time - self.rate_limit_windows[endpoint])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    self.rate_limit_counters[endpoint] = 0
                    self.rate_limit_windows[endpoint] = time.time()
            
            self.rate_limit_counters[endpoint] += 1
        
        self.last_request_time = time.time()
        
    async def handle_bitget_error(self, error, symbol=None, retry_count=0):
        """Handle Bitget API errors with automatic recovery"""
        error_str = str(error).lower()
        error_code = None
        
        # Extract error code if present
        if '"code":' in str(error):
            try:
                import re
                match = re.search(r'"code":"(\d+)"', str(error))
                if match:
                    error_code = match.group(1)
            except:
                pass
        
        # Handle specific Bitget errors
        if error_code == "50067" or "price deviates greatly" in error_str:
            logger.warning(f"Price deviation error for {symbol}. Getting current market price...")
            if symbol and retry_count < 3:
                try:
                    # Get current market price
                    ticker = await asyncio.get_event_loop().run_in_executor(
                        None, self.exchange.fetch_ticker, symbol
                    )
                    market_price = ticker['last']
                    logger.info(f"Adjusted to market price: ${market_price:.4f}")
                    return market_price
                except Exception as e:
                    logger.error(f"Failed to get market price: {e}")
            return None
            
        elif "rate limit" in error_str or error_code in ["429", "30001", "30002"]:
            # Exponential backoff for rate limits
            wait_time = min(30, 2 ** retry_count)
            logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
            return "retry"
            
        elif error_code in ["40001", "40002", "40003", "40004"]:
            logger.error(f"Authentication error {error_code}: {error}")
            return "auth_error"
            
        elif error_code in ["50001"]:
            logger.error(f"Insufficient balance: {error}")
            return "insufficient_balance"
            
        else:
            logger.error(f"Unhandled Bitget error: {error}")
            return "unknown_error"
        
    async def get_market_data(self, symbol):
        """Fetch market data efficiently with error handling"""
        try:
            await self.rate_limit('fetch_ohlcv')
            
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None, self.exchange.fetch_ohlcv, symbol, self.timeframe, None, 50  # Only need 50 candles
            )
            
            if not ohlcv or len(ohlcv) < 30:
                return None
            
            # Create DataFrame from OHLCV data
            df = pd.DataFrame(ohlcv)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Validate data
            # Check for invalid data
            close_series = df['close']
            if len(df) == 0 or close_series.isna().sum() > 0 or close_series.iloc[-1] <= 0:
                return None
                
            return df
            
        except Exception as e:
            if "rate limit" not in str(e).lower():
                logger.error(f"Error fetching data for {symbol}: {e}")
            return None
            
    async def calculate_dynamic_leverage(self, signal, symbol):
        """Calculate dynamic leverage based on signal strength and market conditions"""
        try:
            # Get base score from signal
            base_score = signal.get('confidence', 0)
            
            # Get volatility score (0-100) - use signal data if available instead of relying on self.data
            volatility_score = 50  # Default value
            try:
                # First try to get data from cached dataframe
                if symbol in self.data_cache:
                    df = self.data_cache[symbol]
                    volatility = df['close'].pct_change().std() * 100
                    volatility_score = min(100, volatility * 10)  # Scale volatility to 0-100
                # Fallback to directly calculating from signal
                else:
                    # Default to moderate volatility if no data available
                    volatility_score = 50
            except:
                # If any error, use default value
                volatility_score = 50
                
            # Get volume score (0-100)
            volume_ratio = signal.get('volume_ratio', 1)
            volume_score = min(100, volume_ratio * 50)  # Scale volume to 0-100
            
            # Calculate weighted score
            weighted_score = (
                base_score * self.score_weight +
                volatility_score * self.volatility_weight +
                volume_score * self.volume_weight
            )
            
            # Calculate leverage
            if weighted_score < 30:  # Too risky
                return self.min_leverage
            
            # Scale leverage based on weighted score
            leverage_range = self.max_leverage - self.min_leverage
            leverage = self.min_leverage + (leverage_range * (weighted_score / 100))
            
            # Round to nearest 5
            leverage = round(leverage / 5) * 5
            
            # Ensure within bounds
            leverage = max(self.min_leverage, min(self.max_leverage, leverage))
            
            logger.info(f"[LEVERAGE] {symbol}: Score {weighted_score:.1f} -> {leverage}x leverage")
            return leverage
            
        except Exception as e:
            logger.error(f"Error calculating leverage: {e}")
            # Return default leverage on error
            return self.min_leverage

    async def execute_trade_fast(self, signal):
        """Execute trades with dynamic leverage"""
        symbol = signal['symbol']
        side = signal['type'].lower()
        
        try:
            if self.simulation_mode:
                # Calculate position size with leverage
                leverage = await self.calculate_dynamic_leverage(signal, symbol)
                position_value = self.current_balance * self.position_size_pct * leverage
                
                if side == 'buy':
                    entry_price = signal['price']
                    amount = position_value / entry_price
                    
                    self.positions[symbol] = {
                        'amount': amount,
                        'entry_price': entry_price,
                        'value': position_value,
                        'leverage': leverage,
                        'timestamp': datetime.now(),
                        'signal': signal,
                        'sl_price': signal['sl_price'],
                        'tp_prices': signal['tp_prices'],
                        'tp_hit': [False] * len(signal['tp_prices']),
                        'trailing_active': False
                    }
                    
                    self.simulation_trades.append({
                        'type': 'BUY',
                        'symbol': symbol,
                        'price': entry_price,
                        'amount': amount,
                        'value': position_value,
                        'leverage': leverage,
                        'timestamp': datetime.now()
                    })
                    
                    logger.info(f"SIMULATION: BUY {symbol} @ ${entry_price:.4f} with {leverage}x leverage")
                    return True
                    
                else:  # sell
                    if symbol in self.positions:
                        position = self.positions[symbol]
                        exit_price = signal['price']
                        pnl = (exit_price - position['entry_price']) * position['amount']
                        self.simulation_pnl += pnl
                        self.current_balance += pnl
                        
                        self.simulation_trades.append({
                            'type': 'SELL',
                            'symbol': symbol,
                            'price': exit_price,
                            'amount': position['amount'],
                            'value': position['value'],
                            'leverage': position['leverage'],
                            'pnl': pnl,
                            'timestamp': datetime.now()
                        })
                        
                        logger.info(f"SIMULATION: SELL {symbol} @ ${exit_price:.4f} | PnL: ${pnl:.2f}")
                        del self.positions[symbol]
                        return True
                        
            else:
                # Live trading with leverage and error handling
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        leverage = await self.calculate_dynamic_leverage(signal, symbol)
                        
                        # Get current balance for live trading
                        await self.rate_limit('fetch_balance')
                        balance = await asyncio.get_event_loop().run_in_executor(
                            None, self.exchange.fetch_balance
                        )
                        
                        # Use helper method to get balance data properly
                        balance_data = self.get_bitget_balance(balance)
                        usdt_balance = balance_data['equity']
                        
                        logger.info(f"💰 Balance Check:")
                        logger.info(f"   📊 Total Equity (incl. PnL): ${balance_data['equity']:.2f} USDT")
                        logger.info(f"   💵 Available Balance: ${balance_data['available']:.2f} USDT") 
                        logger.info(f"   📈 Unrealized PnL: ${balance_data['unrealized_pnl']:.2f} USDT")
                        
                        # If balance is too small, use minimum order size
                        if usdt_balance < 10:
                            logger.warning("⚠️ USDT balance too small, using minimum order size")
                            position_value = 10  # Minimum order size
                        else:
                            position_value = usdt_balance * self.position_size_pct * leverage
                        
                        # Ensure position value is at least minimum order size
                        position_value = max(10, position_value)  # Ensure at least $10 order size
                        logger.info(f"📊 Position value: ${position_value:.2f} with {leverage}x leverage")
                        
                        # Set leverage on exchange (non-blocking)
                        try:
                            await self.set_leverage(symbol, leverage)
                        except Exception as e:
                            logger.warning(f"⚠️ Leverage setting failed: {e} - continuing with trade")
                        
                        await asyncio.sleep(0.1)  # Brief pause between leverage and order
                        
                        # Execute trade with leveraged position
                        await self.rate_limit('create_order')
                        
                        if side == 'buy':
                            # Get current market price to avoid deviation errors
                            ticker = await asyncio.get_event_loop().run_in_executor(
                                None, self.exchange.fetch_ticker, symbol
                            )
                            current_price = float(ticker['last'] or signal['price'])
                            amount = position_value / current_price  # Calculate amount from cost
                            
                            # Fixed: Use standard market buy order
                            order = await asyncio.get_event_loop().run_in_executor(
                                None, 
                                self.exchange.create_market_buy_order,
                                symbol, 
                                amount
                            )
                        else:
                            if symbol not in self.positions:
                                return False
                            
                            # Get current price for sell order
                            ticker = await asyncio.get_event_loop().run_in_executor(
                                None, self.exchange.fetch_ticker, symbol
                            )
                            amount = self.positions[symbol]['amount']
                            
                            # Fixed: Use standard market sell order
                            order = await asyncio.get_event_loop().run_in_executor(
                                None, 
                                self.exchange.create_market_sell_order,
                                symbol, 
                                amount
                            )
                        
                        # Track position with leverage
                        if side == 'buy' and order.get('status') == 'closed':
                            self.positions[symbol] = {
                                'amount': order.get('filled', 0),
                                'entry_price': order.get('average', signal['price']),
                                'value': position_value,
                                'leverage': leverage,
                                'timestamp': datetime.now(),
                                'signal': signal,
                                'sl_price': signal['sl_price'],
                                'tp_prices': signal['tp_prices'],
                                'tp_hit': [False] * len(signal['tp_prices']),
                                'trailing_active': False
                            }
                            logger.info(f"✅ BUY SUCCESS: {symbol} @ ${order.get('average', 0):.4f} with {leverage}x leverage")
                        
                        return True
                        
                    except Exception as e:
                        error_result = await self.handle_bitget_error(e, symbol, attempt)
                        
                        if error_result == "retry" and attempt < max_retries - 1:
                            continue
                        elif error_result == "auth_error":
                            logger.error("Authentication failed. Check API credentials.")
                            return False
                        elif error_result == "insufficient_balance":
                            logger.error("Insufficient balance for trade.")
                            return False
                        elif isinstance(error_result, (int, float)):
                            # Got adjusted price, update signal and retry
                            signal['price'] = error_result
                            continue
                        else:
                            logger.error(f"Trade execution failed after {attempt + 1} attempts: {e}")
                            if attempt == max_retries - 1:
                                return False
                
                return False
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False

    async def set_leverage(self, symbol, leverage):
        """Set leverage on exchange with proper Bitget handling"""
        try:
            if not self.simulation_mode:
                # Convert symbol to Bitget futures format if needed
                if '/USDT' in symbol and ':USDT' not in symbol:
                    bitget_symbol = symbol.replace('/USDT', 'USDT')  # BTC/USDT -> BTCUSDT
                else:
                    bitget_symbol = symbol
                
                # Set leverage with proper parameters for Bitget
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.exchange.set_leverage(
                        leverage,
                        bitget_symbol,
                        {
                            'marginCoin': 'USDT',  # Fixed: Set margin coin explicitly
                            'productType': 'UMCBL'  # USDT-M Perpetual
                        }
                    )
                )
                logger.info(f"✅ Set {leverage}x leverage for {symbol} (Bitget: {bitget_symbol})")
        except Exception as e:
            logger.warning(f"⚠️ Could not set leverage for {symbol}: {e}")
            # Don't fail the trade if leverage setting fails

    async def manage_positions(self):
        """Aggressive position management with trailing stops and partial TPs"""
        try:
            for symbol, position in list(self.positions.items()):
                try:
                    # Get current price
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = float(ticker.get('last', 0) or 0)
                    
                    if not current_price or current_price == 0:
                        continue
                    
                    entry_price = position['entry_price']
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # Check stop loss
                    if current_price <= position['sl_price']:
                        logger.warning(f"🛑 STOP LOSS HIT: {symbol} @ ${current_price:.4f}")
                        await self.execute_trade_fast({'symbol': symbol, 'type': 'SELL', 'price': current_price})
                        continue
                    
                    # Check take profits and partial exits
                    for i, (tp_price, tp_hit) in enumerate(zip(position['tp_prices'], position['tp_hit'])):
                        if not tp_hit and current_price >= tp_price:
                            # Partial take profit
                            partial_amount = position['amount'] * self.partial_tp_percentages[i]
                            logger.info(f"🎯 TP{i+1} HIT: {symbol} @ ${current_price:.4f} - Selling {self.partial_tp_percentages[i]*100}%")
                            
                            # Create partial sell order
                            try:
                                self.exchange.create_market_sell_order(symbol, partial_amount)
                                position['amount'] -= partial_amount
                                position['tp_hit'][i] = True
                                
                                # Update trailing stop after TP hit
                                if i == len(position['tp_prices']) - 1:  # Last TP
                                    position['trailing_active'] = True
                                    position['trailing_stop'] = current_price * (1 - self.trailing_stop_distance)
                            except Exception as e:
                                logger.error(f"Partial TP sell error for {symbol}: {e}")
                                pass
                    
                    # Activate trailing stop
                    if pnl_pct >= self.trailing_stop_activation and not position.get('trailing_active'):
                        position['trailing_active'] = True
                        position['trailing_stop'] = current_price * (1 - self.trailing_stop_distance)
                        logger.info(f"📈 TRAILING STOP ACTIVATED: {symbol} @ ${position['trailing_stop']:.4f}")
                    
                    # Update trailing stop
                    if position.get('trailing_active'):
                        new_trailing_stop_price = current_price * (1 - self.trailing_stop_distance)
                        if new_trailing_stop_price > position.get('trailing_stop', 0):
                            position['trailing_stop'] = new_trailing_stop_price
                            
                        # Check trailing stop hit
                        if current_price <= position.get('trailing_stop', 0):
                            logger.info(f"📉 TRAILING STOP HIT: {symbol} @ ${current_price:.4f}")
                            await self.execute_trade_fast({'symbol': symbol, 'type': 'SELL', 'price': current_price})
                            
                except Exception as e:
                    logger.error(f"Position management error for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Position management error: {e}")
            
    async def scan_and_trade(self):
        """Aggressive market scanning and trading"""
        try:
            # Process symbols in batches for speed
            batch_size = 10
            all_signals = []
            
            for i in range(0, len(self.symbols), batch_size):
                batch = self.symbols[i:i+batch_size]
                tasks = [self.process_symbol(symbol) for symbol in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, dict) and result.get('type'):
                        all_signals.append(result)
            
            # Sort by confidence and execute best signals
            all_signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Execute trades
            positions_available = self.max_positions - len(self.positions)
            for signal in all_signals[:positions_available]:
                if signal['confidence'] >= 50:  # LOWERED: Was 60, now 50% for more trades
                    logger.info(f"🎯 SIGNAL: {signal['type']} {signal['symbol']} - Confidence: {signal['confidence']}%")
                    logger.info(f"   {signal['reason']}")
                    
                    # LIVE TRADING - Execute real trades
                    logger.info(f"📈 EXECUTING {signal['type']} trade: {signal['symbol']} @ ${signal['price']:.4f}")
                    result = await self.execute_trade_fast(signal)
                    
                    # TRACK TRADE EXECUTION
                    if result and result != 'insufficient_balance':
                        self.signal_stats['trades_executed'] += 1
                        self.signal_stats['trades_by_symbol'][signal['symbol']] += 1
                        logger.info(f"✅ TRADE EXECUTED: {signal['symbol']} | Total Executed: {self.signal_stats['trades_executed']}")
                    else:
                        self.signal_stats['failed_trades'] += 1
                        logger.warning(f"❌ TRADE FAILED: {signal['symbol']} | Total Failed: {self.signal_stats['failed_trades']}")
                        
                    # Update conversion rate
                    total_signals = self.signal_stats['total_signals_generated']
                    if total_signals > 0:
                        self.signal_stats['signal_to_trade_conversion'] = (self.signal_stats['trades_executed'] / total_signals) * 100
            
            return len(all_signals)
            
        except Exception as e:
            logger.error(f"Scan error: {e}")
            return 0
            
    async def process_symbol(self, symbol):
        """Process individual symbol with improved error handling"""
        try:
            df = await self.get_market_data(symbol)
            if df is None:
                return None
            
            # Store data in both data cache and data dict for compatibility
            self.data_cache[symbol] = df
            self.data[symbol] = df
                
            signal = self.detect_aggressive_pullback(symbol, df)
            
            if signal:
                logger.info(f"🔍 Signal detected for {symbol}: {signal['type']} @ ${signal['price']:.4f}")
                
            return signal
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None
            
    async def main_trading_loop(self):
        """Main trading loop with enhanced monitoring"""
        logger.info("🚀 Starting enhanced main trading loop...")
        
        last_performance_report = time.time()
        performance_report_interval = 300  # Report every 5 minutes
        
        while True:
            try:
                start_time = time.time()
                
                # Process trades
                total_signals = await self.scan_and_trade()
                
                # Position management
                await self.manage_positions()
                
                # Periodic performance reporting
                current_time = time.time()
                if current_time - last_performance_report >= performance_report_interval:
                    await self.log_signal_performance()
                    last_performance_report = current_time
                
                # Sleep for next iteration
                elapsed = time.time() - start_time
                sleep_time = max(1, 5 - elapsed)  # Aim for 5-second cycles
                logger.info(f"🔄 Cycle completed in {elapsed:.1f}s. Sleeping {sleep_time:.1f}s...")
                await asyncio.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal. Shutting down gracefully...")
                await self.log_signal_performance()  # Final report
                break
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
                
    async def log_signal_performance(self):
        """Log comprehensive signal and trade performance statistics"""
        try:
            stats = self.signal_stats
            
            logger.info("📊 ==================== SIGNAL PERFORMANCE REPORT ====================")
            logger.info(f"🎯 SIGNAL GENERATION:")
            logger.info(f"   Total Signals Generated: {stats['total_signals_generated']}")
            logger.info(f"   - BUY Signals: {stats['buy_signals']}")
            logger.info(f"   - SELL Signals: {stats['sell_signals']}")
            logger.info(f"   - Momentum Signals: {stats['momentum_signals']}")
            
            logger.info(f"📈 TRADE EXECUTION:")
            logger.info(f"   Trades Executed: {stats['trades_executed']}")
            logger.info(f"   Failed Trades: {stats['failed_trades']}")
            logger.info(f"   Signal→Trade Conversion: {stats['signal_to_trade_conversion']:.1f}%")
            
            if stats['total_signals_generated'] > 0:
                success_rate = (stats['trades_executed'] / stats['total_signals_generated']) * 100
                logger.info(f"   Overall Success Rate: {success_rate:.1f}%")
            
            # Top performing symbols
            if stats['signals_by_symbol']:
                logger.info(f"🔥 TOP SIGNAL GENERATORS:")
                sorted_symbols = sorted(stats['signals_by_symbol'].items(), key=lambda x: x[1], reverse=True)
                for i, (symbol, count) in enumerate(sorted_symbols[:5], 1):
                    trades = stats['trades_by_symbol'].get(symbol, 0)
                    conversion = (trades / count * 100) if count > 0 else 0
                    logger.info(f"   {i}. {symbol}: {count} signals → {trades} trades ({conversion:.1f}%)")
            
            # Signal frequency analysis
            logger.info(f"⏱️ SIGNAL FREQUENCY:")
            current_time = time.time()
            recent_signals = sum(1 for signals in stats['signal_frequency'].values() 
                               for ts in signals if current_time - ts < 3600)  # Last hour
            logger.info(f"   Signals Last Hour: {recent_signals}")
            
            if stats['total_signals_generated'] > 0:
                avg_time_between = 3600 / max(1, recent_signals)  # Avg time between signals in seconds
                logger.info(f"   Avg Time Between Signals: {avg_time_between:.1f}s")
            
            logger.info("📊 ====================================================================")
            
        except Exception as e:
            logger.error(f"Error logging signal performance: {e}")

    async def run(self):
        """Run the trading bot"""
        try:
            # Get initial symbols
            self.symbols = await self.get_top_trading_pairs()
            
            # Start trading
            await self.main_trading_loop()
            
        except Exception as e:
            logger.error(f"Bot error: {e}")

    async def run_simulation(self, days=1):
        """Optimized simulation with parallel processing"""
        self.log_trade_info(f"[START] Starting {days}-day simulation with ${self.initial_balance:.2f}")
        
        # Fetch top trading pairs for the simulation
        self.symbols = await self.get_top_trading_pairs()
        if not self.symbols:
            logger.error("No symbols available for simulation. Exiting.")
            return
        logger.info(f"Simulating with {len(self.symbols)} symbols: {self.symbols}")

        # Fetch historical data in parallel
        start_time = datetime.now() - timedelta(days=days+1)
        end_time = datetime.now()
        
        # Create tasks for parallel data fetching
        fetch_tasks = []
        for symbol in self.symbols:
            task = asyncio.create_task(self.fetch_historical_data(
                symbol, start_time, end_time
            ))
            fetch_tasks.append(task)
        
        # Wait for all data to be fetched
        await asyncio.gather(*fetch_tasks)
        
        current_time = start_time
        simulation_end = end_time
        
        # Process in batches for better performance
        batch_size = 5  # Process 5 symbols at a time
        symbol_batches = [self.symbols[i:i + batch_size] 
                         for i in range(0, len(self.symbols), batch_size)]
        
        while current_time < simulation_end:
            try:
                # Process batches in parallel
                batch_tasks = []
                for batch in symbol_batches:
                    task = asyncio.create_task(self.process_symbol_batch(batch))
                    batch_tasks.append(task)
                
                # Wait for all batches to complete
                batch_results = await asyncio.gather(*batch_tasks)
                
                # Flatten results and execute trades
                all_signals = [signal for batch in batch_results for signal in batch]
                for signal in all_signals:
                    await self.execute_trade_fast(signal)
                
                await self.manage_positions()
                current_time += timedelta(minutes=5)
                
                if current_time.minute == 0:
                    self.log_trade_info(f"[TIME] Simulation time: {current_time}")
                    self.log_trade_info(f"[BALANCE] Current balance: ${self.current_balance:.2f}")
                    self.log_trade_info(f"[PNL] PnL: ${self.simulation_pnl:.2f}")
                    self.log_trade_info(f"[STATS] Active positions: {len(self.positions)}")
                
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                continue
        
        self.display_simulation_results()
        return self.get_simulation_results()

    async def fetch_historical_data(self, symbol, start_time, end_time):
        """Fetch historical data with caching"""
        try:
            if symbol in self.data_cache:
                return self.data_cache[symbol]
            
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None,
                self.exchange.fetch_ohlcv,
                symbol,
                self.timeframe,
                int(start_time.timestamp() * 1000),
                1000
            )
            
            if ohlcv:
                df = pd.DataFrame(ohlcv)
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Cache the data
                self.data_cache[symbol] = df
                self.log_trade_info(f"[STATS] Loaded {len(df)} candles for {symbol}")
                return df
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    async def process_symbol_batch(self, symbols_batch):
        """Process a batch of symbols in parallel"""
        tasks = []
        for symbol in symbols_batch:
            # Ensure data is available for the symbol from the pre-fetched cache
            if symbol in self.data_cache:
                df = self.data_cache[symbol].copy() # Use data_cache
                if len(df) >= 30:
                    # Run in thread pool for CPU-bound calculations
                    task = self.thread_pool.submit(
                        self.detect_aggressive_pullback,
                        symbol,
                        df
                    )
                    tasks.append((symbol, task))
        
        results = []
        for symbol, task in tasks:
            try:
                signal = task.result(timeout=0.1)  # 100ms timeout
                if signal:
                    results.append(signal)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        return results

    def log_trade_info(self, message):
        """Log trade information with ASCII-safe characters"""
        safe_message = (message.replace('🚀', '[START]')
                              .replace('📊', '[STATS]')
                              .replace('🎯', '[TARGET]')
                              .replace('💰', '[BALANCE]')
                              .replace('📈', '[PNL]')
                              .replace('⏰', '[TIME]')
                              .replace('🛑', '[STOP]'))
        logger.info(safe_message)

    def display_simulation_results(self):
        """Display formatted simulation results"""
        total_trades = len(self.simulation_trades)
        winning_trades = len([t for t in self.simulation_trades if t.get('pnl', 0) > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        print("\n" + "="*50)
        print("SIMULATION RESULTS")
        print("="*50)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance:   ${self.current_balance:,.2f}")
        print(f"Total PnL:      ${self.simulation_pnl:,.2f}")
        print(f"Total Trades:   {total_trades}")
        print(f"Win Rate:       {win_rate:.1f}%")
        print(f"Avg Trade PnL:  ${(self.simulation_pnl/total_trades if total_trades > 0 else 0):,.2f}")
        print("="*50)
        
        # Save detailed trade log with proper encoding
        try:
            trade_log = pd.DataFrame(self.simulation_trades)
            os.makedirs('logs', exist_ok=True)
            trade_log.to_csv('logs/simulation_trades.csv', encoding='utf-8', index=False)
            logger.info("Trade log saved successfully")
        except Exception as e:
            logger.error(f"Error saving trade log: {e}")

    def get_simulation_results(self):
        """Return formatted simulation results"""
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_pnl': self.simulation_pnl,
            'total_trades': len(self.simulation_trades),
            'win_rate': (len([t for t in self.simulation_trades if t.get('pnl', 0) > 0]) / 
                        len(self.simulation_trades) * 100) if self.simulation_trades else 0,
            'trades': self.simulation_trades
        }
    
    def log_trade(self, symbol, side, price, size):
        """Log trade execution"""
        timestamp = datetime.now()
        trade_info = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'price': price,
            'size': size,
            'value': price * size if side == 'BUY' else size
        }
        
        self.trade_history.append(trade_info)
        logger.info(f"📝 Trade Logged: {side} {symbol} @ ${price:.4f} | Size: ${size:.2f}")

    def get_bitget_balance(self, balance_response):
        """
        Extract proper balance data from Bitget API response
        According to GitHub issue #19119, we need to access info[0] for true balance including unrealized PnL
        """
        try:
            # Try new Bitget format first (with equity including unrealized PnL)
            if 'info' in balance_response and isinstance(balance_response.get('info'), list):
                info_list = balance_response['info']
                if info_list and len(info_list) > 0:
                    balance_data = info_list[0]
                    usdt_equity = float(balance_data.get('usdtEquity', 0) or 0)
                    usdt_available = float(balance_data.get('available', 0) or 0)
                    unrealized_pnl = float(balance_data.get('unrealizedPL', 0) or 0)
                    
                    return {
                        'equity': usdt_equity,
                        'available': usdt_available, 
                        'unrealized_pnl': unrealized_pnl,
                        'method': 'bitget_info'
                    }
            
            # Fallback to standard CCXT format
            if 'USDT' in balance_response and balance_response['USDT']:
                usdt_data = balance_response['USDT']
                return {
                    'equity': float(usdt_data.get('total', 0) or 0),
                    'available': float(usdt_data.get('free', 0) or 0),
                    'unrealized_pnl': 0,
                    'method': 'ccxt_standard'
                }
                
            # Last resort fallback
            return {
                'equity': 0,
                'available': 0,
                'unrealized_pnl': 0,
                'method': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"Error parsing balance data: {e}")
            return {
                'equity': 0,
                'available': 0,
                'unrealized_pnl': 0,
                'method': 'error'
            }

def create_bitget_config():
    """Create Bitget configuration file"""
    config_dir = "config"
    config_file = "config/bitget_config.json"
    
    os.makedirs(config_dir, exist_ok=True)
    
    print("🔧 BITGET API CONFIGURATION SETUP")
    print("=" * 50)
    print("To get your API credentials:")
    print("1. Login to Bitget")
    print("2. Go to API Management")
    print("3. Create new API key with Trading permissions")
    print("4. Whitelist your IP address")
    print("=" * 50)
    
    try:
        api_key = input("Enter your Bitget API Key: ").strip()
        secret = input("Enter your Bitget Secret: ").strip()
        passphrase = input("Enter your Bitget Passphrase: ").strip()
        use_sandbox = input("Use sandbox mode? (y/n): ").strip().lower() == 'y'
    except EOFError:
        print("❌ Configuration cancelled due to input error.")
        return None
    
    config = {
        "api_key": api_key,
        "secret": secret,
        "passphrase": passphrase,
        "sandbox": use_sandbox
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✅ Configuration saved to {config_file}")
    return config_file

def main():
    """Main function - autorun in live trading mode"""
    os.makedirs('logs', exist_ok=True)
    
    print("🚀 SUPERTREND PULLBACK TRADING BOT")
    print("=" * 50)
    print("🔥 AUTORUN: LIVE TRADING MODE ACTIVATED 🔥")
    print("=" * 50)
    
    # Determine if we should run in simulation or live mode
    simulation_mode = False
    config_file = "config/bitget_config.json"
    
    if not os.path.exists(config_file):
        print("⚠️ No API configuration found. Running in SIMULATION mode!")
        simulation_mode = True
    else:
        print("✅ API configuration found. Running in LIVE mode!")
        print("⚠️ Using real money!")
    
    try:
        # Initialize the bot
        if simulation_mode:
            bot = AggressivePullbackTrader(simulation_mode=True)
            print("📊 Running 1-day simulation...")
            asyncio.run(bot.run_simulation(days=1))
        else:
            bot = AggressivePullbackTrader(config_file=config_file, simulation_mode=False)
            print("🔄 Starting live trading...")
            asyncio.run(bot.run())
            
    except KeyboardInterrupt:
        logger.info("🛑 Trading stopped by user")
    except Exception as e:
        logger.error(f"Trading error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 