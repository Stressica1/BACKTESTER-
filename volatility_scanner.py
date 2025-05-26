import ccxt
import pandas as pd
import numpy as np
import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
import ta
from statsmodels.tsa.stattools import adfuller
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiohttp
import json

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Fast calculation functions using pure numpy
def fast_atr(high, low, close, period=14):
    """Fast ATR calculation using pure numpy"""
    n = len(high)
    if n < period:
        return np.nan
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr[-1]

def fast_volatility(returns, period=20):
    """Fast volatility calculation using numpy"""
    if len(returns) < period:
        return np.nan
    return np.std(returns[-period:]) * 100

def fast_rsi(closes, period=14):
    """Fast RSI calculation using numpy"""
    if len(closes) < period + 1:
        return 50.0
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fast_bb_squeeze(closes, period=20, std_dev=2):
    """Fast Bollinger Band squeeze calculation"""
    if len(closes) < period:
        return 0.5
    
    recent_closes = closes[-period:]
    sma = np.mean(recent_closes)
    std = np.std(recent_closes)
    
    bb_width = (2 * std_dev * std) / sma * 100
    
    # Compare to historical average (simplified)
    if len(closes) >= period * 2:
        hist_closes = closes[-period*2:-period]
        hist_sma = np.mean(hist_closes)
        hist_std = np.std(hist_closes)
        hist_bb_width = (2 * std_dev * hist_std) / hist_sma * 100
        
        if hist_bb_width > 0:
            return 1 - min(bb_width / hist_bb_width, 1)
    
    return 0.5

class VolatilityScanner:
    """Ultra-fast advanced scanner that identifies the most volatile and appealing coins on Bitget USDT-M futures"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, passphrase: str = None, testnet: bool = True):
        """Initialize the scanner with Bitget API credentials"""
        self.exchange = ccxt.bitget({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,  # Bitget requires passphrase
            'enableRateLimit': False,  # Disable to go faster
            'options': {
                'defaultType': 'swap',  # For USDT-M futures
                'adjustForTimeDifference': True
            }
        })
        
        # Set testnet if applicable
        if testnet and 'test' in self.exchange.urls:
            self.exchange.urls['api'] = self.exchange.urls['test']
            self.exchange.set_sandbox_mode(True)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Timeframes to analyze
        self.timeframes = ['4h']  # Focus on single timeframe for speed
        
        # Volatility calculation parameters
        self.atr_period = 14
        self.vol_period = 20
        self.price_change_periods = [1, 3, 7, 14]  # days
        
        # Additional technical parameters
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bollinger_period = 20
        self.bollinger_std = 2
        self.volume_sma_period = 20
        self.mean_reversion_period = 30
        
        # Ultra-fast rate limiting - minimal delays
        self.last_api_call = 0
        self.min_api_interval = 0.05  # 50ms between API calls (20x faster)
        
        # Enhanced caching with persistence
        self.data_cache = {}
        self.cache_duration = 180  # 3 minutes cache
        
        # Batch processing settings
        self.max_concurrent_requests = 20  # Process 20 symbols simultaneously
        self.batch_size = 10
        
        # Connection pool for aiohttp
        self.session = None
        
        # Pre-computed constants for faster calculations
        self.volatility_regimes = {
            'low': (0, 3),        # 0-3% daily volatility
            'medium': (3, 7),     # 3-7% daily volatility
            'high': (7, 15),      # 7-15% daily volatility
            'extreme': (15, 100)  # 15%+ daily volatility
        }
          # Market cycle detection
        self.market_cycle_periods = 90  # days to analyze for market cycle
    
    async def rate_limit(self):
        """Apply minimal rate limiting to API calls"""
        now = time.time()
        elapsed = now - self.last_api_call
        if elapsed < self.min_api_interval:
            await asyncio.sleep(self.min_api_interval - elapsed)
        self.last_api_call = time.time()

    async def fetch_all_market_data_batch(self, symbols: List[str], timeframe: str = '4h', lookback_days: int = 7) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols in parallel with aggressive batching"""
        tasks = []
        results = {}
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def fetch_single(symbol):
            async with semaphore:
                try:
                    data = await self.fetch_market_data_fast(symbol, timeframe, lookback_days)
                    if data is not None:
                        return symbol, data
                except Exception as e:
                    logger.debug(f"Error fetching {symbol}: {e}")
                    return symbol, None
        
        # Create all tasks
        tasks = [fetch_single(symbol) for symbol in symbols]
        
        # Execute all tasks in parallel
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in completed_tasks:
            if isinstance(result, tuple) and result[1] is not None:
                symbol, data = result
                results[symbol] = data
        
        logger.info(f"Fetched data for {len(results)}/{len(symbols)} symbols")
        return results

    async def fetch_market_data_fast(self, symbol: str, timeframe: str = '4h', lookback_days: int = 7) -> pd.DataFrame:
        """Ultra-fast market data fetching with minimal processing"""
        try:
            # Check cache first with longer cache time
            cache_key = f"{symbol}_{timeframe}_{lookback_days}"
            if cache_key in self.data_cache:
                cache_timestamp, df = self.data_cache[cache_key]
                # Use cache for up to 3 minutes
                if (datetime.now() - cache_timestamp).total_seconds() < self.cache_duration:
                    return df
            
            # Minimal rate limiting
            await self.rate_limit()
            
            # Calculate lookback in milliseconds
            since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
            
            # Fetch OHLCV data with minimal retries
            try:
                ohlcv = await asyncio.to_thread(
                    self.exchange.fetch_ohlcv,
                    symbol,
                    timeframe,
                    since=since,
                    limit=500  # Reduced limit for faster response
                )
            except Exception as e:
                return None
            
            if not ohlcv or len(ohlcv) < 20:
                return None
            
            # Fast DataFrame creation with minimal processing
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Only calculate essential data
            df['returns'] = df['close'].pct_change()
            df.attrs['timeframe'] = timeframe
            
            # Store in cache
            self.data_cache[cache_key] = (datetime.now(), df)
            
            return df
            
        except Exception as e:
            return None

    async def get_all_usdt_m_pairs(self) -> List[str]:
        """Fetch all available USDT-M pairs from Bitget"""
        try:
            await self.rate_limit()
            logger.info("Fetching markets from Bitget...")
            markets = await asyncio.to_thread(self.exchange.fetch_markets)
            
            # Debug: Check what markets are available
            logger.info(f"Total markets retrieved: {len(markets)}")
            
            # More flexible filtering to handle different exchange formats
            usdt_pairs = []
            for market in markets:
                symbol = market['symbol']
                
                # Check for active status
                if not market.get('active', False):
                    continue
                    
                # Check for swap/future type
                if market.get('type') not in ['swap', 'future']:
                    continue
                    
                # Match different USDT pair formats (BTCUSDT, BTC/USDT, BTC/USDT:USDT, etc)
                is_usdt_pair = (
                    (':USDT' in symbol) or 
                    (symbol.endswith('USDT')) or 
                    (symbol.endswith('/USDT')) or
                    (market.get('quote') == 'USDT' and market.get('settle') == 'USDT')
                )
                
                if is_usdt_pair:
                    usdt_pairs.append(symbol)
                    logger.debug(f"Found USDT-M pair: {symbol}")
            
            if not usdt_pairs:
                # Fallback to hard-coded common pairs if no pairs found
                logger.warning("No USDT-M pairs found via API filtering, using fallback list")
                usdt_pairs = [
                    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
                    "BNB/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT", "MATIC/USDT:USDT",
                    "DOT/USDT:USDT", "SHIB/USDT:USDT", "LTC/USDT:USDT", "AVAX/USDT:USDT",
                    "LINK/USDT:USDT", "UNI/USDT:USDT", "ATOM/USDT:USDT"
                ]
            
            logger.info(f"Found {len(usdt_pairs)} active USDT-M pairs on Bitget")
            return usdt_pairs
            
        except Exception as e:
            logger.error(f"Error fetching USDT-M pairs: {str(e)}")
            # Return a minimal list of common pairs as fallback
            fallback_pairs = [
                "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"
            ]
            logger.info(f"Using {len(fallback_pairs)} fallback pairs due to error")
            return fallback_pairs

    async def fetch_market_data(self, symbol: str, timeframe: str, 
                              lookback_days: int = 30) -> pd.DataFrame:
        """Fetch market data for a specific symbol and timeframe"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{lookback_days}"
            if cache_key in self.data_cache:
                cache_timestamp, df = self.data_cache[cache_key]
                # If cache is recent (less than 5 minutes old), use it
                if (datetime.now() - cache_timestamp).total_seconds() < 300:
                    return df
            
            await self.rate_limit()
            # Calculate lookback in milliseconds
            since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
            
            # Fetch OHLCV data with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ohlcv = await asyncio.to_thread(
                        self.exchange.fetch_ohlcv,
                        symbol,
                        timeframe,
                        since=since,
                        limit=1000
                    )
                    break
                except ccxt.RateLimitExceeded:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol} on {timeframe}: {str(e)}")
                    return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Make sure the dataframe is sorted by time
            df = df.sort_index()
            
            # Calculate additional data
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Ensure we have enough data
            if len(df) < 30:
                logger.warning(f"Not enough data for {symbol} on {timeframe}, only {len(df)} candles")
                return None
            
            # Store in cache
            self.data_cache[cache_key] = (datetime.now(), df)
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} on {timeframe}: {str(e)}")
            return None

    def calculate_volatility_metrics_fast(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Ultra-fast volatility metrics calculation using optimized functions"""
        if df is None or len(df) < self.vol_period:
            return None
        
        try:
            # Convert to numpy arrays for speed
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            
            current_price = closes[-1]
            
            # Fast ATR calculation using numba
            atr = fast_atr(highs, lows, closes, self.atr_period)
            atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
            
            # Fast volatility using numba
            returns = np.diff(closes) / closes[:-1]  # Calculate returns directly
            daily_volatility = fast_volatility(returns, self.vol_period)
            
            # Fast RSI using numba
            rsi = fast_rsi(closes, self.rsi_period)
            
            # Fast BB squeeze using numba
            bb_squeeze = fast_bb_squeeze(closes, self.bollinger_period, self.bollinger_std)
            
            # Quick volume metrics
            volume = volumes[-1]
            avg_volume = np.mean(volumes[-self.volume_sma_period:]) if len(volumes) >= self.volume_sma_period else volume
            vol_change_pct = ((volume - avg_volume) / avg_volume) * 100 if avg_volume > 0 else 0
            relative_volume = volume / avg_volume if avg_volume > 0 else 1.0
            
            # Quick price changes (simplified for speed)
            price_changes = {}
            tf_multiplier = {'5m': 288, '15m': 96, '1h': 24, '4h': 6, '1d': 1}
            tf = df.attrs.get('timeframe', '4h')
            
            for days in [1, 3, 7]:  # Reduced for speed
                periods = days * tf_multiplier.get(tf, 6)
                if len(closes) > periods:
                    past_price = closes[-periods-1]
                    change_pct = ((current_price - past_price) / past_price) * 100
                    price_changes[f"{days}d_change"] = change_pct
            
            # Quick Bollinger Band metrics
            if len(closes) >= self.bollinger_period:
                bb_middle = np.mean(closes[-self.bollinger_period:])
                bb_std = np.std(closes[-self.bollinger_period:])
                bb_upper = bb_middle + self.bollinger_std * bb_std
                bb_lower = bb_middle - self.bollinger_std * bb_std
                bb_width = ((bb_upper - bb_lower) / bb_middle) * 100 if bb_middle > 0 else 0
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            else:
                bb_width = 0
                bb_position = 0.5
            
            # Quick volume volatility
            if len(volumes) >= self.vol_period:
                volume_returns = np.diff(volumes) / volumes[:-1]
                volume_volatility = np.std(volume_returns[-self.vol_period:]) * 100
            else:
                volume_volatility = 0
            
            # Quick mean reversion score
            if len(closes) >= self.mean_reversion_period:
                price_sma = np.mean(closes[-self.mean_reversion_period:])
                price_std = np.std(closes[-self.mean_reversion_period:])
                z_score = (current_price - price_sma) / price_std if price_std > 0 else 0
                mean_reversion_score = min(abs(z_score) / 3, 1) if z_score != 0 else 0
                mean_reversion_direction = -1 if z_score > 0 else 1 if z_score < 0 else 0
            else:
                z_score = 0
                mean_reversion_score = 0
                mean_reversion_direction = 0
            
            # Latest candle range
            latest_candle_range_pct = ((highs[-1] - lows[-1]) / lows[-1]) * 100 if lows[-1] > 0 else 0
            
            # Volatility regime (simplified)
            volatility_regime = "medium"
            if daily_volatility < 3:
                volatility_regime = "low"
            elif daily_volatility > 15:
                volatility_regime = "extreme"
            elif daily_volatility > 7:
                volatility_regime = "high"
            
            # Volume trend (simplified)
            if len(volumes) >= 10:
                recent_vol_avg = np.mean(volumes[-5:])
                prev_vol_avg = np.mean(volumes[-10:-5])
                volume_trend = ((recent_vol_avg - prev_vol_avg) / prev_vol_avg) * 100 if prev_vol_avg > 0 else 0
            else:
                volume_trend = 0
            
            # Simplified metrics for speed
            metrics = {
                'symbol': symbol,
                'last_price': current_price,
                
                # Core volatility metrics
                'atr': atr,
                'atr_pct': atr_pct,
                'daily_volatility': daily_volatility,
                'volatility_regime': volatility_regime,
                
                # Price action metrics
                'latest_range_pct': latest_candle_range_pct,
                'bb_width': bb_width,
                'bb_position': bb_position,
                'bb_squeeze': bb_squeeze,
                
                # Technical indicators
                'rsi': rsi,
                
                # Volume metrics
                'volume': volume,
                'avg_volume': avg_volume,
                'volume_change_pct': vol_change_pct,
                'relative_volume': relative_volume,
                'volume_volatility': volume_volatility,
                'volume_trend': volume_trend,
                
                # Mean reversion metrics
                'z_score': z_score,
                'mean_reversion_score': mean_reversion_score,
                'mean_reversion_direction': mean_reversion_direction,
                  # Add price changes
                **price_changes,
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics for {symbol}: {str(e)}")
            return None

    def calculate_appeal_score(self, metrics: Dict[str, float]) -> float:
        """Calculate an enhanced overall appeal score based on multiple metrics"""
        if not metrics:
            return 0
        
        # Expanded weights for different metrics
        weights = {
            # Volatility metrics (50% of score)
            'atr_pct': 0.10,                 # Higher ATR % means more volatility
            'daily_volatility': 0.10,         # Standard deviation of daily returns
            'bb_squeeze': 0.10,               # Bollinger Band squeeze (higher = more compressed and ready to expand)
            'latest_range_pct': 0.05,         # Latest candle range as % of price
            'volatility_premium': 0.05,       # Short-term vs long-term volatility premium
            'bb_width': 0.05,                 # Bollinger Band width as % of price
            'volume_volatility': 0.05,        # Volatility of volume
            
            # Momentum metrics (25% of score)
            'mean_reversion_score': 0.10,     # Mean reversion potential (higher = more likely to revert)
            'volume_change_pct': 0.05,        # Recent volume change
            'relative_volume': 0.05,          # Current volume relative to average
            'volume_trend': 0.05,             # Recent volume trend
            
            # Trend metrics (15% of score)
            'trend_strength': 0.05,           # Strength of current trend
            'macd_direction': 0.05,           # MACD direction (bullish/bearish)
            '1d_change': 0.025,               # Short-term price change
            '3d_change': 0.025,               # Medium-term price change
            
            # Technical metrics (10% of score)
            'rsi': 0.10,                      # RSI (adjusted for mean reversion or trend following)
        }
        
        # Normalize price changes
        for days in metrics.get('price_change_periods', [1, 3, 7, 14]):
            key = f"{days}d_change"
            if key in metrics and key in weights:
                # Convert to a 0-1 score (more positive is better, consider +/- 20% as the range)
                metrics[f"{key}_score"] = (metrics[key] + 20) / 40
                metrics[f"{key}_score"] = max(0, min(1, metrics[f"{key}_score"]))
        
        # Calculate RSI score, optimizing for either trend following or mean reversion
        # For trend following: higher RSI = better for uptrends, lower RSI = better for downtrends
        # For mean reversion: extreme RSI (high or low) = better opportunity
        rsi = metrics.get('rsi', 50)
        
        # Determine if we should score for mean reversion or trend following based on ADX
        trend_strength = metrics.get('trend_strength', 0)
        
        if trend_strength > 0.3:  # Strong trend, use trend following scoring
            # If trend is up (positive price change), higher RSI is better
            # If trend is down (negative price change), lower RSI is better
            recent_change = metrics.get('1d_change', 0)
            if recent_change > 0:  # Uptrend
                metrics['rsi_score'] = rsi / 100  # Higher RSI is better
            else:  # Downtrend
                metrics['rsi_score'] = (100 - rsi) / 100  # Lower RSI is better
        else:  # Weak trend, use mean reversion scoring
            # Extreme RSI values are better for mean reversion
            metrics['rsi_score'] = abs(rsi - 50) / 50  # 0 = neutral, 1 = extreme
        
        # Calculate appeal score
        score = 0
        used_weights_sum = 0
        
        for metric, weight in weights.items():
            if metric == 'rsi' and 'rsi_score' in metrics:
                score += metrics['rsi_score'] * weight
                used_weights_sum += weight
            elif metric.endswith('_change') and f"{metric}_score" in metrics:
                score += metrics[f"{metric}_score"] * weight
                used_weights_sum += weight
            elif metric in metrics and not pd.isna(metrics[metric]):
                # Normalize metrics to 0-1 scale based on typical ranges
                norm_ranges = {
                    'atr_pct': (0, 10),               # 0-10% ATR
                    'daily_volatility': (0, 15),       # 0-15% daily volatility
                    'bb_width': (0, 30),               # 0-30% BB width
                    'bb_squeeze': (0, 1),              # Already 0-1
                    'latest_range_pct': (0, 15),       # 0-15% candle range
                    'volatility_premium': (-5, 5),     # -5% to +5% premium
                    'mean_reversion_score': (0, 1),    # Already 0-1
                    'volume_change_pct': (-50, 200),   # -50% to +200% volume change
                    'relative_volume': (0, 3),         # 0-3x relative volume
                    'volume_trend': (-50, 100),        # -50% to +100% trend
                    'trend_strength': (0, 1),          # Already 0-1
                    'volume_volatility': (0, 100),     # 0-100% volume volatility
                    'macd_direction': (-1, 1),         # -1 to +1, normalize to 0-1
                }
                
                if metric in norm_ranges:
                    min_val, max_val = norm_ranges[metric]
                    value = metrics[metric]
                    
                    # Special handling for mean reversion direction
                    if metric == 'macd_direction':
                        # Convert from [-1, 0, 1] to [0, 0.5, 1]
                        norm_value = (value + 1) / 2
                    else:
                        # Standard normalization
                        norm_value = (value - min_val) / (max_val - min_val)
                    
                    # Clip to 0-1 range
                    norm_value = max(0, min(1, norm_value))
                    score += norm_value * weight
                    used_weights_sum += weight
        
        # Normalize score based on weights actually used
        if used_weights_sum > 0:
            score = score / used_weights_sum
          # Add bonus for extreme opportunities (very high volatility + significant mean reversion potential)
        if metrics.get('daily_volatility', 0) > 10 and metrics.get('mean_reversion_score', 0) > 0.8:
            score *= 1.2  # 20% bonus
        
        # Cap at 1.0
        return min(score, 1.0)

    async def scan_all_markets(self, timeframe: str = '4h', top_n: int = 20, min_volume: float = 100000) -> List[Dict[str, Any]]:
        """Scan all markets using ultra-fast method by default"""
        return await self.scan_all_markets_ultra_fast(timeframe, top_n, min_volume)

    async def scan_all_markets_ultra_fast(self, timeframe: str = '4h', top_n: int = 20, min_volume: float = 100000) -> List[Dict[str, Any]]:
        """Ultra-fast market scanning with parallel processing and optimized calculations"""
        # Get all USDT-M pairs
        pairs = await self.get_all_usdt_m_pairs()
        
        if not pairs:
            logger.error("No pairs found to scan! Returning empty result.")
            return []
        
        logger.info(f"Starting ultra-fast volatility scan for {len(pairs)} pairs on {timeframe} timeframe")
        
        # Batch fetch all market data in parallel
        start_time = time.time()
        all_data = await self.fetch_all_market_data_batch(pairs, timeframe, lookback_days=7)
        fetch_time = time.time() - start_time
        logger.info(f"Fetched data for {len(all_data)} pairs in {fetch_time:.2f} seconds")
        
        # Process all metrics calculations in parallel using thread pool
        results = []
        
        def calculate_metrics_wrapper(args):
            symbol, df = args
            if df is not None:
                return self.calculate_volatility_metrics_fast(df, symbol)
            return None
        
        # Use ThreadPoolExecutor for CPU-bound metric calculations
        with ThreadPoolExecutor(max_workers=8) as executor:
            start_calc_time = time.time()
            metric_results = list(executor.map(calculate_metrics_wrapper, all_data.items()))
            calc_time = time.time() - start_calc_time
            logger.info(f"Calculated metrics for {len(metric_results)} pairs in {calc_time:.2f} seconds")
        
        # Filter and score results
        for metrics in metric_results:
            if metrics and metrics.get('avg_volume', 0) >= min_volume:
                # Add appeal score
                metrics['appeal_score'] = self.calculate_appeal_score_fast(metrics)
                # Add timeframe for reference
                metrics['timeframe'] = timeframe
                # Add timestamp
                metrics['timestamp'] = datetime.now().isoformat()
                results.append(metrics)
        
        # Sort by appeal score, descending
        results.sort(key=lambda x: x.get('appeal_score', 0), reverse=True)
        
        # Return top N results
        top_results = results[:top_n]
        total_time = time.time() - start_time
        logger.info(f"Completed ultra-fast scan in {total_time:.2f}s, found {len(results)} valid pairs, returning top {len(top_results)}")
        
        return top_results

    def calculate_appeal_score_fast(self, metrics: Dict[str, float]) -> float:
        """Ultra-fast appeal score calculation with simplified weights"""
        if not metrics:
            return 0
        
        # Simplified scoring for speed
        score = 0
        
        # Volatility components (60%)
        daily_vol = min(metrics.get('daily_volatility', 0) / 10, 1) * 0.20
        atr_pct = min(metrics.get('atr_pct', 0) / 5, 1) * 0.15
        bb_squeeze = metrics.get('bb_squeeze', 0) * 0.15
        latest_range = min(metrics.get('latest_range_pct', 0) / 8, 1) * 0.10
        
        # Volume components (25%)
        vol_change = min(abs(metrics.get('volume_change_pct', 0)) / 100, 1) * 0.10
        rel_volume = min(metrics.get('relative_volume', 1) / 3, 1) * 0.15
        
        # Mean reversion (15%)
        mean_rev = metrics.get('mean_reversion_score', 0) * 0.15
        
        score = daily_vol + atr_pct + bb_squeeze + latest_range + vol_change + rel_volume + mean_rev
        
        return min(score, 1.0)

    async def get_volatility_rankings(self, top_n: int = 20, min_volume: float = 1000000) -> Dict[str, List[Dict[str, Any]]]:
        """Get volatility rankings for multiple timeframes"""
        results = {}
        
        for timeframe in self.timeframes:
            results[timeframe] = await self.scan_all_markets(timeframe, top_n, min_volume)
            
        return results
        
    def predict_future_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Use historical volatility patterns to predict future volatility
        Note: This is a simplified prediction and not a full GARCH model
        """
        try:
            # Extract returns
            returns = df['returns'].dropna()
            
            if len(returns) < 30:
                return {}
                
            # Calculate recent volatility (last 10 periods)
            recent_vol = returns[-10:].std() * 100
            
            # Calculate volatility of volatility (how stable is volatility?)
            # Higher values mean volatility itself is volatile
            rolling_vol = returns.rolling(window=5).std() * 100
            vol_of_vol = rolling_vol.std()
            
            # Estimate volatility persistence (autocorrelation of squared returns)
            # Higher values mean volatility tends to cluster
            squared_returns = returns**2
            vol_persistence = squared_returns.autocorr(lag=1)
            
            # Simple volatility forecast based on recent trend
            vol_5d = returns[-5:].std() * 100
            vol_10d = returns[-10:].std() * 100
            vol_20d = returns[-20:].std() * 100
            
            # If recent volatility is increasing, predict it continues
            vol_trend = (vol_5d - vol_10d) / vol_10d if vol_10d > 0 else 0
            
            # Predicted change in volatility
            pred_vol_change = vol_trend * (1 + abs(vol_persistence))
            
            # Predicted volatility
            predicted_vol = recent_vol * (1 + pred_vol_change)
            
            # Confidence score (0-1, higher = more confident)
            # Based on stability of volatility and sample size
            prediction_confidence = 1 - min(vol_of_vol / recent_vol, 1) if recent_vol > 0 else 0
            
            return {
                'recent_volatility': recent_vol,
                'volatility_of_volatility': vol_of_vol,
                'volatility_persistence': vol_persistence,
                'volatility_trend': vol_trend,
                'predicted_volatility': predicted_vol,
                'prediction_confidence': prediction_confidence,
            }
        except Exception as e:
            logger.error(f"Error predicting volatility: {str(e)}")
            return {}

# Function to run the scanner
async def scan_for_volatile_coins(api_key: str = None, api_secret: str = None, 
                                testnet: bool = True, top_n: int = 20, min_volume: float = 1000000) -> Dict[str, List[Dict[str, Any]]]:
    """Run the enhanced volatility scanner and return results"""
    scanner = VolatilityScanner(api_key, api_secret, testnet)
    return await scanner.get_volatility_rankings(top_n, min_volume)

# Synchronous wrapper for the scanner
def run_volatility_scan(api_key: str = None, api_secret: str = None, 
                      testnet: bool = True, top_n: int = 20, min_volume: float = 1000000) -> Dict[str, List[Dict[str, Any]]]:
    """Synchronous wrapper to run the enhanced volatility scanner"""
    return asyncio.run(scan_for_volatile_coins(api_key, api_secret, testnet, top_n, min_volume))

if __name__ == "__main__":
    # Run independently for testing
    results = run_volatility_scan(top_n=10)
    
    # Print results
    for timeframe, coins in results.items():
        print(f"\nTop volatile coins for {timeframe} timeframe:")
        for i, coin in enumerate(coins, 1):
            print(f"{i}. {coin['symbol']} - Appeal Score: {coin['appeal_score']:.2f}, "
                 f"Vol: {coin['daily_volatility']:.2f}%, ATR%: {coin['atr_pct']:.2f}%")