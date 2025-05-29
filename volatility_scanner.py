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
import requests
from requests.adapters import HTTPAdapter
import scipy.stats # Placeholder for potential future use
from scipy.stats import norm

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Define constants for default values if metrics are missing
DEFAULT_FUNDING_RATE = 0.0001  # Neutral assumption
DEFAULT_OPEN_INTEREST = 100000  # Arbitrary, to avoid zero division if used as divisor
DEFAULT_MARK_INDEX_SPREAD = 0.0005 # 0.05%
DEFAULT_LAUNCH_TIME = time.time() - (365 * 24 * 60 * 60) # Assume 1 year old
DEFAULT_TAKER_FEE = 0.0006 # 0.06%
DEFAULT_MAKER_FEE = 0.0002 # 0.02%
DEFAULT_MAINT_MARGIN_RATE = 0.005 # 0.5%

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
        # Temporary session before defining max_concurrent_requests
        temp_session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        temp_session.mount('https://', adapter)
        temp_session.mount('http://', adapter)
        # Initialize CCXT exchange with custom HTTP session to increase pool size
        self.exchange = ccxt.bitget({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,  # Bitget requires passphrase
            'enableRateLimit': False,  # Disable to go faster
            'session': temp_session,
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
        self.timeframes = ['5m', '15m', '30m']  # Analyze 5, 15, and 30 minute intervals
        
        # Volatility calculation parameters
        self.atr_period = 14
        self.vol_period = 20
        self.price_change_periods = [1, 3, 7, 14]  # days
        
        # Additional technical parameters
        self.rsi_period = 14
        self.bollinger_period = 20
        self.bollinger_std = 2
        self.volume_sma_period = 20
        self.mean_reversion_period = 30
        
        # Ultra-fast rate limiting - minimal delays
        self.last_api_call = 0
        self.min_api_interval = 0.2  # 200ms between API calls (safer)
        self.max_concurrent_requests = 8  # Lower concurrency to avoid rate limits
        
        # Enhanced caching with persistence
        self.data_cache = {}
        self.cache_duration = 180  # 3 minutes cache
        
        # Batch processing settings
        self.batch_size = 10
        
        # Connection pool for aiohttp
        self.session = None
        
        # Override CCXT HTTP session with increased pool size to avoid 'pool is full' warnings
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=self.max_concurrent_requests, pool_maxsize=self.max_concurrent_requests)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        # Assign to CCXT exchange
        self.exchange.session = session
        
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
            cache_key = f"{symbol}_{timeframe}_{lookback_days}"
            if cache_key in self.data_cache:
                cache_timestamp, df = self.data_cache[cache_key]
                if (datetime.now() - cache_timestamp).total_seconds() < self.cache_duration:
                    return df
            await self.rate_limit()
            since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
            max_retries = 5
            for attempt in range(max_retries):
            try:
                ohlcv = await asyncio.to_thread(
                    self.exchange.fetch_ohlcv,
                    symbol,
                    timeframe,
                    since=since,
                        limit=500
                )
                    break
                except ccxt.RateLimitExceeded:
                    logger.warning(f"[429] Rate limit exceeded for {symbol} {timeframe}, retrying (attempt {attempt+1})...")
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                    if '429' in str(e):
                        logger.warning(f"[429] Too Many Requests for {symbol} {timeframe}, retrying (attempt {attempt+1})...")
                        await asyncio.sleep(2 ** attempt)
                        continue
                return None
            else:
                logger.error(f"Failed to fetch {symbol} {timeframe} after {max_retries} retries due to rate limits.")
                return None
            if not ohlcv or len(ohlcv) < 20:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            df['returns'] = df['close'].pct_change()
            df.attrs['timeframe'] = timeframe
            self.data_cache[cache_key] = (datetime.now(), df)
            return df
        except Exception as e:
            logger.error(f"Error fetching fast market data for {symbol}: {str(e)}")
            return None

    async def get_all_usdt_m_pairs_with_leverage(self) -> List[Tuple[str, float]]:
        """Fetch all available USDT-M pairs and their max leverage from Bitget"""
        try:
            await self.rate_limit()
            logger.info("Fetching markets from Bitget...")
            markets = await asyncio.to_thread(self.exchange.fetch_markets)
            pairs_with_leverage = []
            for market in markets:
                symbol = market['symbol']
                if not market.get('active', False):
                    continue
                if market.get('type') not in ['swap', 'future']:
                    continue
                is_usdt_pair = (
                    (':USDT' in symbol) or 
                    (symbol.endswith('USDT')) or 
                    (symbol.endswith('/USDT')) or
                    (market.get('quote') == 'USDT' and market.get('settle') == 'USDT')
                )
                if is_usdt_pair:
                    max_leverage = market.get('limits', {}).get('leverage', {}).get('max', 1)
                    pairs_with_leverage.append((symbol, max_leverage))
            logger.info(f"Found {len(pairs_with_leverage)} active USDT-M pairs with leverage on Bitget")
            return pairs_with_leverage
        except Exception as e:
            logger.error(f"Error fetching USDT-M pairs with leverage: {str(e)}")
            return []

    async def fetch_market_data(self, symbol: str, timeframe: str, lookback_days: int = 30) -> pd.DataFrame:
        """Fetch market data for a specific symbol and timeframe"""
        try:
            cache_key = f"{symbol}_{timeframe}_{lookback_days}"
            if cache_key in self.data_cache:
                cache_timestamp, df = self.data_cache[cache_key]
                if (datetime.now() - cache_timestamp).total_seconds() < 300:
                    return df
            await self.rate_limit()
            since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
            max_retries = 5
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
                    logger.warning(f"[429] Rate limit exceeded for {symbol} {timeframe}, retrying (attempt {attempt+1})...")
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    if '429' in str(e):
                        logger.warning(f"[429] Too Many Requests for {symbol} {timeframe}, retrying (attempt {attempt+1})...")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    logger.error(f"Error fetching data for {symbol} on {timeframe}: {str(e)}")
                    return None
            else:
                logger.error(f"Failed to fetch {symbol} {timeframe} after {max_retries} retries due to rate limits.")
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            if len(df) < 30:
                logger.warning(f"Not enough data for {symbol} on {timeframe}, only {len(df)} candles")
                return None
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
            
            # OBV calculation
            if len(closes) > 1:
                obv = 0
                obv_arr = np.zeros_like(closes)
                for i in range(1, len(closes)):
                    if closes[i] > closes[i-1]:
                        obv += volumes[i]
                    elif closes[i] < closes[i-1]:
                        obv -= volumes[i]
                    obv_arr[i] = obv
                obv_val = obv_arr[-1]
            else:
                obv_val = 0

            # Momentum (10-period ROC)
            momentum_period = 10
            if len(closes) > momentum_period:
                momentum = (closes[-1] - closes[-momentum_period-1]) / closes[-momentum_period-1]
            else:
                momentum = 0
            
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
                'obv': obv_val,
                'momentum': momentum,
                
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
                }
                
                if metric in norm_ranges:
                    min_val, max_val = norm_ranges[metric]
                    value = metrics[metric]
                    
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
        # Get all USDT-M pairs with leverage
        pairs_with_leverage = await self.get_all_usdt_m_pairs_with_leverage()
        # Filter for leverage >= 25x
        pairs = [s for s, lev in pairs_with_leverage if lev >= 25]
        leverage_map = {s: lev for s, lev in pairs_with_leverage}
        if not pairs:
            logger.error("No pairs found to scan after leverage filter! Returning empty result.")
            return []
        logger.info(f"Starting ultra-fast volatility scan for {len(pairs)} pairs on {timeframe} timeframe")
        # Batch fetch all market data in parallel
        start_time = time.time()
        all_data = await self.fetch_all_market_data_batch(pairs, timeframe, lookback_days=7)
        fetch_time = time.time() - start_time
        logger.info(f"Fetched data for {len(all_data)} pairs in {fetch_time:.2f} seconds")
        results = []
        def calculate_metrics_wrapper(args):
            symbol, df = args
            if df is not None:
                metrics = self.calculate_volatility_metrics_fast(df, symbol)
                if metrics:
                    metrics['leverage'] = leverage_map.get(symbol, 1)
                return metrics
            return None
        with ThreadPoolExecutor(max_workers=8) as executor:
            start_calc_time = time.time()
            metric_results = list(executor.map(calculate_metrics_wrapper, all_data.items()))
            calc_time = time.time() - start_calc_time
            logger.info(f"Calculated metrics for {len(metric_results)} pairs in {calc_time:.2f} seconds")
        for metrics in metric_results:
            if metrics and metrics.get('avg_volume', 0) >= min_volume and metrics.get('leverage', 1) >= 25:
                # Add full enhanced score and tier
                ranker = EnhancedMarketRanker()
                perf = {
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'profit_factor': metrics.get('profit_factor', 1),
                    'win_rate': metrics.get('win_rate', 0.5),
                    'max_drawdown': metrics.get('max_drawdown', 0.2),
                    'consistency': metrics.get('consistency', 0.5),
                }
                mtf_signals = metrics.get('mtf_signals', None)
                scored = ranker.score_market(metrics, mtf_signals, perf)
                metrics['score'] = scored['score']
                metrics['tier'] = scored['tier']
                metrics['mtf_trend'] = scored['mtf_trend']
                results.append(metrics)
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
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

class EnhancedMarketRanker:
    """
    Enhanced market ranking system with ultra-modern visual experience
    """
    # ANSI color codes
    COLOR_RESET = '\033[0m'
    COLOR_BOLD = '\033[1m'
    COLOR_UNDERLINE = '\033[4m'
    COLOR_BLINK = '\033[5m'

    # Enhanced visual elements
    TIERS = [
        (90, "GODLIKE UNICORN BUSSY TIER 🌈✨💫"),
        (80, "ULTRA PREMIUM BUSSY TIER 💎🔥⚡"),
        (70, "PREMIUM TIER ✨🌟💫"),
        (60, "HIGH POTENTIAL TIER 🔥🚀💪"),
        (50, "STRONG PERFORMER TIER ⚡💯🎯"),
        (40, "SOLID MARKET TIER 🥇💫✨"),
        (30, "AVERAGE MARKET TIER 🥈📊📈"),
        (20, "BASIC MARKET TIER 🥉📉📊"),
        (10, "LOW QUALITY TIER 🪫⚠️📉"),
        (0,  "PURE SHIT TIER 💩🚫❌")
    ]

    # Enhanced metric emojis with dynamic states
    METRIC_EMOJIS = {
        'volume': ['💰', '💎', '💵', '💸', '🤑'],
        'volatility': ['🌋', '⚡', '🔥', '💥', '🌪️'],
        'leverage': ['🔥', '💪', '⚡', '🚀', '💫'],
        'tick_size': ['🎯', '🎨', '🎪', '🎭', '🎪'],
        'liquidity': ['🌊', '💧', '🌊', '💦', '🌊'],
        'obv': ['📊', '📈', '📉', '📊', '📈'],
        'momentum': ['🚀', '💫', '⚡', '🔥', '💪'],
        'rsi': ['📈', '📊', '📉', '📈', '📊'],
        'supertrend': ['🔮', '✨', '🌟', '💫', '⭐'],
        'sharpe': ['📏', '📐', '📊', '📈', '📉'],
        'profit_factor': ['💹', '📈', '💯', '🎯', '✨'],
        'win_loss': ['🏆', '🎯', '💪', '🔥', '⚡'],
        'drawdown': ['📉', '⚠️', '🚫', '❌', '💩'],
        'consistency': ['🔄', '⚡', '💫', '✨']
    }

    # Enhanced ANSI colors with gradients
    METRIC_COLORS = {
        'volume': '\033[38;2;255;215;0m',      # Gold gradient
        'volatility': '\033[38;2;255;0;255m',  # Magenta gradient
        'leverage': '\033[38;2;255;0;0m',      # Red gradient
        'tick_size': '\033[38;2;0;191;255m',   # Deep sky blue
        'liquidity': '\033[38;2;0;255;255m',   # Cyan gradient
        'obv': '\033[38;2;0;255;0m',           # Green gradient
        'momentum': '\033[38;2;255;69;0m',     # Orange-red gradient
        'rsi': '\033[38;2;30;144;255m',        # Dodger blue
        'supertrend': '\033[38;2;147;112;219m', # Medium purple
        'sharpe': '\033[38;2;50;205;50m',      # Lime green
        'profit_factor': '\033[38;2;255;215;0m', # Gold
        'win_loss': '\033[38;2;0;255;127m',    # Spring green
        'drawdown': '\033[38;2;255;0;0m',      # Red
        'consistency': '\033[38;2;0;191;255m'  # Deep sky blue
    }

    # Animation frames for loading states
    LOADING_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def __init__(self, mtf_weights=None, base_max_score=100):
        if mtf_weights is None:
            self.mtf_weights = {'15m': 0.15, '1h': 0.25, '4h': 0.30, '1d': 0.30}
        else:
            self.mtf_weights = mtf_weights
        self.base_max_score = base_max_score
        self.metrics_breakdown = {}
        self.perf_data = {}

        # Define metric configurations: (weight, is_positive_correlation_to_score, normalization_type, params)
        # Normalization types: 'direct_value', 'percentage_of_max', 'percentile', 'inverse_ratio', 'log_transform', 'range_map'
        # params for 'range_map': [input_min, input_max, output_min, output_max] (output typically 0-1 or 0-10 for sub-scores)
        # params for 'percentile': (not directly used here, handled by overall percentile scaling if chosen)
        self.metric_configs = {
            'volatility': {'weight': 15, 'positive': True, 'norm_type': 'range_map', 'params': [0, 0.2, 0, 10]}, # 0-20% daily vol maps to 0-10
            'volume_24h': {'weight': 15, 'positive': True, 'norm_type': 'log_transform', 'params': [100000, 10]}, # Scale by log, base_value, scale_factor
            'leverage': {'weight': 5, 'positive': True, 'norm_type': 'direct_value', 'params': [125, 10]}, # Max leverage, scale_factor (max_leverage/params[0] * params[1])
            'tick_size_impact': {'weight': 10, 'positive': False, 'norm_type': 'range_map', 'params': [0, 0.001, 0, 10]}, # 0-0.1% tick size as % of price maps to 0-10 (lower is better)
            'liquidity_score': {'weight': 10, 'positive': True, 'norm_type': 'direct_value', 'params': [1, 10]}, # Assume liquidity score is 0-1, maps to 0-10
            'obv_slope': {'weight': 10, 'positive': True, 'norm_type': 'range_map', 'params': [-45, 45, 0, 10]}, # OBV slope angle maps to 0-10
            'momentum': {'weight': 10, 'positive': True, 'norm_type': 'range_map', 'params': [-0.1, 0.1, 0, 10]}, # -10% to +10% momentum maps to 0-10
            'rsi': {'weight': 5, 'positive': False, 'norm_type': 'rsi_deviation', 'params': [50, 10]}, # Deviate from 50, max_score_contribution

            # New Metrics
            'funding_rate_impact': {'weight': 5, 'positive': False, 'norm_type': 'range_map', 'params': [0, 0.001, 0, 10]}, # Absolute funding rate 0 to 0.1% maps to score 0-10 (lower abs is better)
            'open_interest': {'weight': 5, 'positive': True, 'norm_type': 'log_transform', 'params': [1000000, 10]}, # Log transform for open interest in quote currency
            'mark_index_spread_impact': {'weight': 3, 'positive': False, 'norm_type': 'range_map', 'params': [0, 0.002, 0, 10]}, # 0 to 0.2% spread maps to 0-10 (lower is better)
            'contract_age_bonus': {'weight': 2, 'positive': True, 'norm_type': 'range_map', 'params': [0, 365*2, 0, 5]}, # 0 to 2 years age maps to 0-5 bonus points
            'fee_impact': {'weight': 3, 'positive': False, 'norm_type': 'range_map', 'params': [0.0002, 0.001, 0, 10]}, # Taker fee 0.02% to 0.1% maps to 0-10 (lower is better)
            'maint_margin_impact': {'weight': 2, 'positive': False, 'norm_type': 'range_map', 'params': [0.004, 0.02, 0, 10]} # Maint margin 0.4% to 2% maps to 0-10 (lower is better)
        }
        self.total_weight = sum(config['weight'] for config in self.metric_configs.values())
        self.animation_index = 0
        self.last_update = time.time()

    def get_animated_emoji(self, metric: str, value: float) -> str:
        emojis = self.METRIC_EMOJIS.get(metric, ['❓'])
        # Normalize value to 0-1 for emoji index
        norm = min(max(float(value) / 20, 0), 1)
        index = min(int(norm * (len(emojis)-1)), len(emojis) - 1)
        return emojis[index]

    def get_gradient_color(self, metric: str, value: float) -> str:
        base_color = self.METRIC_COLORS.get(metric, '\033[38;2;255;255;255m')
        # Optionally, could modulate color intensity by value
        return base_color

    def create_progress_bar(self, value: float, width: int = 20) -> str:
        value = min(max(value, 0), 1)
        filled = int(width * value)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {value*100:.1f}%"

    def create_metric_line(self, metric: str, value: float, score: float) -> str:
        emoji = self.get_animated_emoji(metric, score)
        color = self.get_gradient_color(metric, score)
        progress = self.create_progress_bar(score/20)  # Normalize to 0-1 range
        return f"{color}{emoji} {metric.title():<12} {progress}{self.COLOR_RESET}"

    def create_tier_banner(self, tier: str, score: float) -> str:
        banner = f"\n{self.COLOR_BOLD}{'='*60}{self.COLOR_RESET}\n"
        banner += f"{self.COLOR_BOLD}{' ' * 15}{tier}{self.COLOR_RESET}\n"
        banner += f"{self.COLOR_BOLD}{' ' * 20}Score: {score:.2f}{self.COLOR_RESET}\n"
        banner += f"{self.COLOR_BOLD}{'='*60}{self.COLOR_RESET}\n"
        return banner

    def create_visual_breakdown(self, metrics: Dict, breakdown: Dict) -> str:
        visual_lines = []
        for metric, score in breakdown.items():
            value = metrics.get(metric, score)
            try:
                visual_lines.append(self.create_metric_line(metric, value, score))
            except Exception:
                visual_lines.append(f"{metric}: {score:.1f}")
        return "\n".join(visual_lines)

    def _normalize_metric(self, value, norm_type, params, metric_name=""):
        try:
            if value is None: # Handle None values upfront for most types
                # For range_map, None should map to output_min
                if norm_type == 'range_map' and params and len(params) == 4:
                    return params[2] # output_min
                return 0 # Default score for None if not range_map

            if norm_type == 'direct_value': 
                # Assumes value is already scaled appropriately (e.g. a pre-calculated 0-10 score component)
                # params can be [max_expected_value_for_full_score, score_multiplier]
                # e.g. if value is 0-1, params=[1, 10] maps it to 0-10.
                # If value is max_leverage, params=[125, 10] -> (value/125)*10
                # If value is max_leverage, params=[125, 10] -> (value/125)*10
                if params and len(params) == 2:
                    max_val, multiplier = params
                    return (value / max_val) * multiplier if max_val != 0 else 0
                return value # Return as is if no params
            
            elif norm_type == 'percentage_of_max': 
                max_possible, score_multiplier = params
                if max_possible == 0: return 0
                return (value / max_possible) * score_multiplier
            
            elif norm_type == 'log_transform': 
                base_val, scale_factor = params
                if value <= 0: return 0 # Log is undefined for non-positive
                # Adding 1 inside log to ensure that if value == base_val, log result is not 0 before scaling
                # Max(1, value / base_val) handles cases where value might be less than base_val, preventing log of sub-1 values from being negative.
                normalized_value = np.log(max(1, value / base_val) + 1) 
                return normalized_value * scale_factor
            
            elif norm_type == 'range_map': 
                input_min, input_max, output_min, output_max = params
                if input_max == input_min: return output_min # Avoid division by zero
                clamped_value = max(input_min, min(input_max, value))
                return output_min + (clamped_value - input_min) * (output_max - output_min) / (input_max - input_min)
            
            elif norm_type == 'rsi_deviation': 
                mid_point, max_score_contribution = params
                deviation = abs(value - mid_point)
                # Score is inversely proportional to deviation. Max deviation is mid_point (e.g. 50 for RSI from 0-100 scale)
                # (1 - deviation / mid_point) gives 1 for no deviation, 0 for max deviation.
                score = max_score_contribution * (1 - (deviation / mid_point)) if mid_point != 0 else 0
                return max(0, score) # Ensure score is not negative
            
            elif norm_type == 'inverse_ratio': 
                # Example: (1 / (value_normalized_to_0_1 + epsilon)) * scale_factor
                # This type is tricky and often better handled by range_map with inverted logic (positive=False in config)
                # For instance, for fees: map fee_range [0.0002, 0.001] to score_range [10, 0]
                # So if 'positive' is False in metric_config, the interpretation of range_map already handles inverse.
                # We can remove 'inverse_ratio' or make it very specific if needed.
                # For now, let's assume range_map with positive=False covers this.
                return 0 # Placeholder - prefer range_map for inverse relationships
            
            else:
                # Fallback for unknown norm_type, or if direct value is intended without explicit type
                return value 
        except Exception as e:
            print(f"Error normalizing metric {metric_name} with value {value} (type {type(value)}), norm_type '{norm_type}', params {params}: {e}")
            return 0 # Return 0 score on error

    def score_market(self, market_data, mtf_signals, perf_data):
        self.metrics_breakdown = {}
        self.perf_data = perf_data # Store perf_data if needed later
        total_raw_weighted_score = 0 # Sum of (normalized_metric_score * weight)
        
        # Ensure market_data and its 'info' field exist
        market_info = market_data.get('info', {})
        ticker_info = market_data.get('ticker', {}).get('info', {})
        
        # Robustly get current_price, falling back to perf_data or 1.0
        current_price = market_data.get('ticker', {}).get('last')
        if current_price is None or current_price == 0:
            current_price = self.perf_data.get('current_price') # Check perf_data next
        if current_price is None or current_price == 0:
            current_price = ticker_info.get('lastPr') # Check ticker_info as another fallback
        if current_price is None or current_price == 0:
            current_price = 1.0 # Final fallback to avoid division by zero
        try:
            current_price = float(current_price)
        except (ValueError, TypeError):
            current_price = 1.0

        # --- Standard Metrics ---
        volatility = self.perf_data.get('volatility', 0)
        self.metrics_breakdown['Volatility (%)'] = f"{volatility*100:.2f}"
        cfg = self.metric_configs['volatility']
        total_raw_weighted_score += self._normalize_metric(volatility, cfg['norm_type'], cfg['params'], 'volatility') * cfg['weight']

        volume_24h_quote = market_data.get('ticker', {}).get('quoteVolume', 0)
        if volume_24h_quote is None: volume_24h_quote = 0
        try: volume_24h_quote = float(volume_24h_quote)
        except: volume_24h_quote = 0
        self.metrics_breakdown['Volume 24h (Quote)'] = f"{volume_24h_quote:,.0f}"
        cfg = self.metric_configs['volume_24h']
        total_raw_weighted_score += self._normalize_metric(volume_24h_quote, cfg['norm_type'], cfg['params'], 'volume_24h') * cfg['weight']

        leverage = market_data.get('limits', {}).get('leverage', {}).get('max', market_info.get('maxLeverage', 1.0))
        if leverage is None: leverage = 1.0
        try: leverage = float(leverage)
        except: leverage = 1.0
        self.metrics_breakdown['Max Leverage'] = f"{leverage:.0f}x"
        cfg = self.metric_configs['leverage']
        total_raw_weighted_score += self._normalize_metric(leverage, cfg['norm_type'], cfg['params'], 'leverage') * cfg['weight']

        tick_size = market_data.get('precision', {}).get('price', 0.000001) # Price increment
        if tick_size is None: tick_size = 0.000001
        try: tick_size = float(tick_size)
        except: tick_size = 0.000001
        tick_size_as_percentage = (tick_size / current_price) if current_price != 0 else 0
        self.metrics_breakdown['Tick Size Impact'] = f"{tick_size_as_percentage*100:.4f}% (of price)"
        cfg = self.metric_configs['tick_size_impact']
        total_raw_weighted_score += self._normalize_metric(tick_size_as_percentage, cfg['norm_type'], cfg['params'], 'tick_size_impact') * cfg['weight']

        liquidity_score_val = self.perf_data.get('liquidity_score', 0) # Assume this is pre-calculated 0-1
        self.metrics_breakdown['Liquidity Score (0-1)'] = f"{liquidity_score_val:.2f}"
        cfg = self.metric_configs['liquidity_score']
        total_raw_weighted_score += self._normalize_metric(liquidity_score_val, cfg['norm_type'], cfg['params'], 'liquidity_score') * cfg['weight']

        obv_slope = self.perf_data.get('obv_slope', 0)
        self.metrics_breakdown['OBV Slope (degrees)'] = f"{obv_slope:.2f}"
        cfg = self.metric_configs['obv_slope']
        total_raw_weighted_score += self._normalize_metric(obv_slope, cfg['norm_type'], cfg['params'], 'obv_slope') * cfg['weight']

        momentum = self.perf_data.get('momentum', 0) # e.g., 1-day price change
        self.metrics_breakdown['Momentum (1d Change %)'] = f"{momentum*100:.2f}"
        cfg = self.metric_configs['momentum']
        total_raw_weighted_score += self._normalize_metric(momentum, cfg['norm_type'], cfg['params'], 'momentum') * cfg['weight']
        
        rsi = self.perf_data.get('rsi', 50)
        self.metrics_breakdown['RSI'] = f"{rsi:.2f}"
        cfg = self.metric_configs['rsi']
        total_raw_weighted_score += self._normalize_metric(rsi, cfg['norm_type'], cfg['params'], 'rsi') * cfg['weight']

        # --- New Metrics ---
        funding_rate_str = ticker_info.get('capitalRate', market_info.get('fundFeeRate'))
        funding_rate = DEFAULT_FUNDING_RATE # Default from constants
        if funding_rate_str is not None:
            try: funding_rate = float(funding_rate_str)
            except (ValueError, TypeError):
                # If conversion fails, funding_rate remains DEFAULT_FUNDING_RATE
                pass 
        self.metrics_breakdown['Funding Rate (%)'] = f"{funding_rate*100:.4f}"
        cfg_fr = self.metric_configs['funding_rate_impact']
        total_raw_weighted_score += self._normalize_metric(abs(funding_rate), cfg_fr['norm_type'], cfg_fr['params'], 'funding_rate_impact') * cfg_fr['weight']

        open_interest_contracts_str = ticker_info.get('openInterest')
        open_interest_contracts = DEFAULT_OPEN_INTEREST # Default
        if open_interest_contracts_str is not None:
            try: open_interest_contracts = float(open_interest_contracts_str)
            except (ValueError, TypeError): pass 
        
        contract_size_str = market_data.get('contractSize', market_info.get('lotSize'))
        contract_size = 1.0 # Default
        if contract_size_str is not None:
            try: contract_size = float(contract_size_str)
            except (ValueError, TypeError): pass
        if contract_size == 0: contract_size = 1.0 # Avoid division by zero

        open_interest_quote = open_interest_contracts * contract_size * current_price
        self.metrics_breakdown['Open Interest (Quote)'] = f"{open_interest_quote:,.0f}"
        cfg_oi = self.metric_configs['open_interest']
        total_raw_weighted_score += self._normalize_metric(open_interest_quote, cfg_oi['norm_type'], cfg_oi['params'], 'open_interest') * cfg_oi['weight']

        mark_price_str = ticker_info.get('markPrice')
        index_price_str = ticker_info.get('indexPrice')
        mark_price, index_price = current_price, current_price # Default to current_price if specific values are missing
        if mark_price_str is not None: 
            try: mark_price = float(mark_price_str)
            except (ValueError, TypeError): pass
        if index_price_str is not None: 
            try: index_price = float(index_price_str)
            except (ValueError, TypeError): pass
        
        mark_index_spread_abs = abs(mark_price - index_price) / index_price if index_price != 0 else 0
        self.metrics_breakdown['Mark/Index Spread (%)'] = f"{mark_index_spread_abs*100:.4f}"
        cfg_mis = self.metric_configs['mark_index_spread_impact']
        total_raw_weighted_score += self._normalize_metric(mark_index_spread_abs, cfg_mis['norm_type'], cfg_mis['params'], 'mark_index_spread_impact') * cfg_mis['weight']
        
        launch_time_str = market_info.get('launchTime') # Expects ms string from Bitget
        launch_timestamp_s = DEFAULT_LAUNCH_TIME # Default
        if launch_time_str is not None:
            try: launch_timestamp_s = int(launch_time_str) / 1000 # Convert ms to s
            except (ValueError, TypeError): pass
        
        contract_age_days = (time.time() - launch_timestamp_s) / (24 * 60 * 60)
        self.metrics_breakdown['Contract Age (Days)'] = f"{contract_age_days:.0f}"
        cfg_age = self.metric_configs['contract_age_bonus']
        total_raw_weighted_score += self._normalize_metric(contract_age_days, cfg_age['norm_type'], cfg_age['params'], 'contract_age_bonus') * cfg_age['weight']

        taker_fee_str = market_data.get('taker', market_info.get('takerFeeRate'))
        taker_fee = DEFAULT_TAKER_FEE # Default
        if taker_fee_str is not None:
            try: taker_fee = float(taker_fee_str)
            except (ValueError, TypeError): pass

        self.metrics_breakdown['Taker Fee (%)'] = f"{taker_fee*100:.4f}"
        cfg_fee = self.metric_configs['fee_impact']
        total_raw_weighted_score += self._normalize_metric(taker_fee, cfg_fee['norm_type'], cfg_fee['params'], 'fee_impact') * cfg_fee['weight']

        maint_margin_rate_str = market_info.get('maintainMarginRate')
        maint_margin_rate = DEFAULT_MAINT_MARGIN_RATE # Default
        if maint_margin_rate_str is not None:
            try: maint_margin_rate = float(maint_margin_rate_str)
            except (ValueError, TypeError): pass
        self.metrics_breakdown['Maint. Margin Rate (%)'] = f"{maint_margin_rate*100:.2f}"
        cfg_mmr = self.metric_configs['maint_margin_impact']
        total_raw_weighted_score += self._normalize_metric(maint_margin_rate, cfg_mmr['norm_type'], cfg_mmr['params'], 'maint_margin_impact') * cfg_mmr['weight']
        
        # --- MTF Scoring & Finalization ---
        mtf_overall_trend = "Neutral"
        weighted_mtf_numeric_score = 0 # for trend calculation

        if mtf_signals: # mtf_signals is expected to be a dict like {'15m': {'trend': 'Up', ...}, ...}
            self.metrics_breakdown['MTF Signals'] = {}
            for tf, signal_data in mtf_signals.items():
                trend = "Neutral" # Default trend for a timeframe if not specified
                if isinstance(signal_data, dict):
                    trend = signal_data.get('trend', "Neutral")
                elif isinstance(signal_data, str): # Allow direct trend string e.g., mtf_signals={'1h': 'Up'}
                    trend = signal_data
                
                self.metrics_breakdown['MTF Signals'][tf] = trend
                
                tf_weight = self.mtf_weights.get(tf, 0)
                trend_value = 0
                if trend == 'Up': trend_value = 1
                elif trend == 'Strong Up': trend_value = 1 # Treat Strong Up as Up for numeric scoring for now
                elif trend == 'Down': trend_value = -1
                elif trend == 'Strong Down': trend_value = -1 # Treat Strong Down as Down
                weighted_mtf_numeric_score += trend_value * tf_weight
            
            # Determine overall MTF trend string based on the weighted numeric score
            # Sum of weights can be > 1 or < 1 depending on config. For example, 0.15+0.25+0.3+0.3 = 1.0
            # Thresholds for trend categories can be relative to sum of positive/negative weights
            sum_of_weights = sum(self.mtf_weights.values())
            if sum_of_weights == 0: sum_of_weights = 1 # Avoid division by zero if weights are all zero

            if weighted_mtf_numeric_score >= 0.7 * sum_of_weights: mtf_overall_trend = "Very Strong Up"
            elif weighted_mtf_numeric_score >= 0.3 * sum_of_weights: mtf_overall_trend = "Strong Up"
            elif weighted_mtf_numeric_score > 0: mtf_overall_trend = "Up"
            elif weighted_mtf_numeric_score <= -0.7 * sum_of_weights: mtf_overall_trend = "Very Strong Down"
            elif weighted_mtf_numeric_score <= -0.3 * sum_of_weights: mtf_overall_trend = "Strong Down"
            elif weighted_mtf_numeric_score < 0: mtf_overall_trend = "Down"
            # If weighted_mtf_numeric_score is 0, it remains "Neutral"

        self.metrics_breakdown['MTF Overall Trend'] = mtf_overall_trend

        # Determine the maximum possible raw weighted score based on metric configurations
        max_possible_raw_weighted_score = 0
        for metric_key, cfg_item in self.metric_configs.items():
            try:
                max_sub_score_contribution = 0
                norm_type = cfg_item.get('norm_type') # Use .get for safety
                params = cfg_item.get('params', [])    # Use .get for safety

                if norm_type == 'range_map':
                    if len(params) > 3:
                        max_sub_score_contribution = params[3] 
                    else:
                        max_sub_score_contribution = 10 # Fallback for malformed params
                elif norm_type in ['rsi_deviation', 'direct_value', 'log_transform', 'percentage_of_max']:
                    if len(params) > 1:
                        max_sub_score_contribution = params[1]
                    else: 
                        max_sub_score_contribution = 10 # Fallback for malformed params
                else: 
                    max_sub_score_contribution = 10 # Fallback for unknown norm_type
                
                current_weight = cfg_item.get('weight', 0) # Use .get for safety
                max_possible_raw_weighted_score += max_sub_score_contribution * current_weight
            except Exception as e_cfg_loop:
                print(f"Error processing metric_config for {metric_key}: {e_cfg_loop}. Using default contribution.")
                max_possible_raw_weighted_score += 5 * cfg_item.get('weight', 1)

        if max_possible_raw_weighted_score == 0: 
            # Fallback if all weights are zero or configs are malformed
            max_possible_raw_weighted_score = self.total_weight * 10 if self.total_weight > 0 else 100 

        final_score = (total_raw_weighted_score / max_possible_raw_weighted_score) * self.base_max_score if max_possible_raw_weighted_score != 0 else 0
        final_score = max(0, min(self.base_max_score, final_score)) # Clamp to [0, base_max_score]

        tier = "PURE SHIT TIER"
        if final_score >= 90: tier = "GODLIKE UNICORN BUSSY TIER"
        elif final_score >= 80: tier = "GIGA CHAD TIER"
        elif final_score >= 70: tier = "TOP G TIER"
        elif final_score >= 60: tier = "DECENT TIER"
        elif final_score >= 50: tier = "MID AF TIER"
        elif final_score >= 40: tier = "LOW TIER"
        elif final_score >= 30: tier = "BOTTOM OF THE BARREL TIER"
        
        self.metrics_breakdown['Calculated Score'] = f"{final_score:.2f} / {self.base_max_score}"
        self.metrics_breakdown['Tier'] = tier
        
        summary_str = self._generate_summary_string(market_data.get('symbol', 'N/A'), final_score, tier, mtf_overall_trend)

        return {
            'symbol': market_data.get('symbol', 'N/A'),
            'score': final_score,
            'tier': tier,
            'mtf_trend': mtf_overall_trend,
            'metrics_breakdown': self.metrics_breakdown,
            'perf_data': self.perf_data, # Pass along perf_data for inspection if needed
            'summary': summary_str
        }

    def _generate_summary_string(self, symbol, score, tier, mtf_trend):
        # ... existing code ...

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
    results = run_volatility_scan(top_n=10, testnet=False)
    ranker = EnhancedMarketRanker()
    # Print results with full visual scoring
    for timeframe, coins in results.items():
        print(f"\nTop volatile coins for {timeframe} timeframe:")
        for i, coin in enumerate(coins, 1):
            # Use the ranker to get the full visual summary
            perf = {
                'sharpe_ratio': coin.get('sharpe_ratio', 0),
                'profit_factor': coin.get('profit_factor', 1),
                'win_rate': coin.get('win_rate', 0.5),
                'max_drawdown': coin.get('max_drawdown', 0.2),
                'consistency': coin.get('consistency', 0.5),
            }
            mtf_signals = coin.get('mtf_signals', None)
            result = ranker.score_market(coin, mtf_signals, perf)
            ranker.print_market_analysis(result)